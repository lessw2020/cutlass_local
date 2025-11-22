import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.torch as cutlass_torch
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack


# Constants
io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32

mma_tcgen05_inst_shape_mnk = (128, 256, 16)  # shape of single tcgen05.mma instruction
mma_cta_tiler_mnk = (128, 256, 64)  # block shape that 1 or 2 ctas will process
threads_per_cta = 128

# Pipeline stage configuration
ab_stages = 4   # number of stages for A and B matrices
acc_stage = 1   # number of stages for accumulation before writing to global memory

# host function that constructs the tiled MMA and SMEM layouts, and launches the kernel
@cute.jit
def host_function(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    kernel: cutlass.Constexpr,
):
    # Construct tiled MMA
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

    # Construct SMEM layouts for A and B
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
    a_smem_layout_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_layout_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])

    # Construct TMA load atoms
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

    # Launch the kernel
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    kernel(
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

def prep_tmem(smem_storage: cute.SharedStorage, tmem_barrier_id: int=1, threads_per_cta: int=128, num_tmem_cols: int=512):
        # Allocate all TMEM columns
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=tmem_barrier_id,
            num_threads=threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            smem_storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        
        tmem.allocate(num_tmem_cols)
        return tmem

# Make K-major tensors (torch tensors are row-major)
def make_tensors(mn, k, dtype):
    shape = (mn, k)
    return (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=dtype, device="cuda")
    )


# define problem size
m, n, k = 8192, 8192, 8192


a = make_tensors(m, k, cutlass_torch.dtype(io_dtype))
b = make_tensors(n, k, cutlass_torch.dtype(io_dtype))
c = make_tensors(m, n, cutlass_torch.dtype(io_dtype))

a_tensor = (
    from_dlpack(a)
    .mark_layout_dynamic(leading_dim=1)
)
b_tensor = (
    from_dlpack(b)
    .mark_layout_dynamic(leading_dim=1)
)
c_tensor = (
    from_dlpack(c)
    .mark_layout_dynamic(leading_dim=1)
)