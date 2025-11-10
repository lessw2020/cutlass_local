import argparse
import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128

# Pipeline stage configuration
ab_stages = 4
acc_stages = 1

def get_smem_stage_bytes(dtype, smem_layout):
    """Calculate bytes for a single smem layout stage (excluding stage dimension).
    
    Args:
        dtype: Element data type (e.g., cutlass.Float16)
        smem_layout: Shared memory layout (ComposedLayout with stages)
    
    Returns:
        Number of bytes for one stage of the layout
    """
    return cute.size_in_bytes(dtype, cute.select(smem_layout, mode=[0, 1, 2]))

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages*2]
    tmem_holding_buff: cutlass.Int32

@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TiledMma, 
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor, 
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    # Current thread/warp/block coordinates
    tx, ty, tz = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    bx, by, bz = cute.arch.block_idx()
    mma_coord_mnk = (bx, by, None)

    # 1. Prepare args
    # allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    sm_storage = smem.allocate(SharedStorage)

    # tile A and B into SMEM
    sA = smem.allocate_tensor(
        element_type = io_dtype,
        layout = a_smem_layout.outer,
        byte_alignment = 128,
        swizzle = a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type = io_dtype,
        layout = b_smem_layout.outer,
        byte_alignment = 128,
        swizzle = b_smem_layout.inner,
    )

    # allocate all TMEM columns
    tmem_alloc_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads = threads_per_cta,
    )
    tmem = utils.TmemAllocator(
        sm_storage.tmem_holding_buff,
        barrier_for_retrieve = tmem_alloc_barrier,
    )

    num_tmem_columns = 512
    tmem.allocate(num_tmem_columns)

    # TMA
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
    
    # Pipeline configuration
    num_tma_stage_copy_bytes = (
        get_smem_stage_bytes(io_dtype, a_smem_layout) +
        get_smem_stage_bytes(io_dtype, b_smem_layout)
    )

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count = num_tma_stage_copy_bytes,
        barrier_storage = sm_storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()

    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages = acc_stages,
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage = sm_storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    # Partition tensors for MMA and make fragments
    # blockM, blockK, RestK
    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    # blockN, blockK, RestK
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    # bM, bN
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1,1,None))
    thr_mma = tiled_mma.get_slice(0)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    # Partition tensors for TMA; This requires the tensors partitioned for MMA
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 3),
        cute.group_modes(tCgA, 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 3),
        cute.group_modes(tCgB, 0, 3),
    )

    # CTA wide sync before retrieving the pointer to the start of the allocated TMEM
    # Only warp 0 does the allocation so we need to sync before retrieving the TMEM start address
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    # Swap the pointer in tCtAcc
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile_cnt = 4
    # (EpiTile)
    epi_tiler = (
        (cute.size(tCtAcc, mode=[0,0]), cute.size(tCtAcc, mode=[0,1]) // subtile_cnt),
    )

    # (EpiTile, NumTiles)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    # (EpiTile, NumTiles)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)

    # Every thread loads 32x128 bits
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        cutlass.Float32,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None,0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tx)

    # (TmemCpy,NumTmemCpy,NumTiles)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    # (TmemCpy,NumTmemCpy,NumTiles)
    tDgC = tmem_thr_copy.partition_D(gC_epi)

    # (TmemCpy,NumTmemCpy)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    # (TmemCpy,NumTmemCpy)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    # 2 Main Loop! 
    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx ==0:
        # wait for empty accumulator buffer
        acc_empty = acc_producer.acquire_and_advance()
        for k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages-2):
            # TMA loads
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tAgA[(None, ab_empty.count)],
                tAsA[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b,
                tBgB[(None, ab_empty.count)],
                tBsB[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )

            # execute one K-block worth of MMA instructions
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(tCrA, mode=[2])
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    tCrA[k_block_coord],
                    tCrB[k_block_coord],
                    tCtAcc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
        acc_empty.commit()

    # 3. Epilogue
    # Release TMEM allocation lock
    tmem.relinquish_alloc_permit()
    # Wait for the accumulator buffer to be full
    acc_full = acc_consumer.wait_and_advance()
    # TMEM -> RMEM -> GEMM
    # Sub-tiling for better instruction-level parallelism
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()

    # deallocate tmem
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)

@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    op = tcgen05.MmaF16BF16Op(
        io_dtype,
        acc_dtype,
        mma_inst_shape_mnk,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    
    # construct smem layouts for A and B
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a.element_type,
        ab_stages,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b.element_type,
        ab_stages,
    )
    print(f"a_smem_layout = {a_smem_layout}")
    print(f"b_smem_layout = {b_smem_layout}")
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])
    print(f"a_smem_layout_one_stage = {a_smem_layout_one_stage}")
    print(f"b_smem_layout_one_stage = {b_smem_layout_one_stage}")
    print(f"a_smem_layout = {a_smem_layout}")
    print(f"b_smem_layout = {b_smem_layout}")

    # construct tma load atoms
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_smem_layout_one_stage,
        mma_tiler_mnk,
        tiled_mma,
    )
    print(f"a_tma_atom = {a_tma_atom}")
    print(f"b_tma_atom = {b_tma_atom}")
    print(f"a_tma_tensor = {a_tma_tensor}")
    print(f"b_tma_tensor = {b_tma_tensor}")

    # Pretty prints kernel attributes useful for debugging
    print(f"a            = {cute.pretty_str(a)}")
    print(f"b            = {cute.pretty_str(b)}")
    print(f"c            = {cute.pretty_str(c)}")
    print(f"tiled_mma    = {cute.pretty_str(tiled_mma)}")
    print(f"a_tma_atom   = {cute.pretty_str(a_tma_atom)}")
    print(f"b_tma_atom   = {cute.pretty_str(b_tma_atom)}")
    print(f"a_tma_tensor = {cute.pretty_str(a_tma_tensor)}")
    print(f"b_tma_tensor = {cute.pretty_str(b_tma_tensor)}")

    # Launch the kernel
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    gemm_kernel(
        tiled_mma,
        a_tma_atom,
        a_tma_tensor,
        b_tma_atom,
        b_tma_tensor,
        c,
        a_smem_layout, 
        b_smem_layout,
    ).launch(
        grid=grid_shape,
        block=(threads_per_cta, 1, 1),
    )

def run_dense_gemm(
    mnk: Tuple[int, int, int],
    tolerance: float,
):
    print("===================================================================")
    print("Running Blackwell fp16 GEMM example 0 with:")
    print(f"  mnk:       {mnk}")
    print(f"  tolerance: {tolerance}")
    print("===================================================================")
    print()
    
    m,n,k = mnk
    torch.manual_seed(2020)

    
    # Make K-major tensors (torch tensors are row-major)
    def make_tensors(mn, k, dtype):
        shape = (mn, k)
        return (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=dtype, device="cuda")
        )

    a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
    b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
    c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))
    a_tensor = (
        from_dlpack(a, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    b_tensor = (
        from_dlpack(b, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=k)
    )
    c_tensor = (
        from_dlpack(c, assumed_align=32)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=n)
    )

    # Entry point to the host JIT function
    host_function(
        a_tensor,
        b_tensor,
        c_tensor,
        no_cache=True,
    )

    # Compute reference result and verify
    ref = (torch.einsum("mk,nk->mn", a.to(torch.float32), b.to(torch.float32))).cpu()

    torch.testing.assert_close(
        c.cpu(), ref.to(cutlass_torch.dtype(io_dtype)), atol=tolerance, rtol=1e-05
    )


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str):
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required to run this example")

    parser = argparse.ArgumentParser(description="Blackwell fp16 GEMM example 0")
    parser.add_argument(
        "--mnk",
        type=parse_comma_separated_ints,
        default=[8192, 8192, 8192],
        help="MNK dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    args = parser.parse_args()
    if len(args.mnk) != 3:
        parser.error("--mnk must contain exactly 3 values")
    if args.mnk[0] % mma_tiler_mnk[0] != 0 or args.mnk[1] % mma_tiler_mnk[1] != 0:
        parser.error("m n must be divisible by mma_tiler_mn")

    run_dense_gemm(
        args.mnk,
        args.tolerance,
    )
    print("PASS")