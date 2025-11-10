import argparse
from typing import Optional, Tuple, Type, Union

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack


def _compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
    use_tma_store: bool,
    c_smem_layout: Union[cute.Layout, None],
) -> Tuple[int, int, int]:
    """Computes the number of stages for A/B/C operands based on heuristics.

    :param tiled_mma: The tiled MMA object defining the core computation.
    :type tiled_mma: cute.TiledMma
    :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
    :type mma_tiler_mnk: tuple[int, int, int]
    :param a_dtype: Data type of operand A.
    :type a_dtype: type[cutlass.Numeric]
    :param b_dtype: Data type of operand B.
    :type b_dtype: type[cutlass.Numeric]
    :param c_dtype: Data type of operand C (output).
    :type c_dtype: type[cutlass.Numeric]
    :param smem_capacity: Total available shared memory capacity in bytes.
    :type smem_capacity: int
    :param occupancy: Target number of CTAs per SM (occupancy).
    :type occupancy: int
    :param use_tma_store: Whether TMA store is enabled.
    :type use_tma_store: bool
    :param c_smem_layout: Layout of C operand in shared memory, or None if not using TMA store.
    :type c_smem_layout: Union[cute.Layout, None]

    :return: A tuple containing the computed number of stages for:
             (ACC stages, A/B operand stages, C stages)
    :rtype: tuple[int, int, int]
    """
    # Default ACC stages
    num_acc_stage = 2

    # Default C stages
    num_c_stage = 2 if use_tma_store else 0

    # Calculate smem layout and size for one stage of A, B, and C with 1-stage
    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
        tiled_mma, mma_tiler_mnk, a_dtype, 1
    )
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
        tiled_mma, mma_tiler_mnk, b_dtype, 1
    )

    ab_bytes_per_stage = cute.size_in_bytes(
        a_dtype, a_smem_layout_stage_one
    ) + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
    mbar_helpers_bytes = 1024

    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout)
    c_bytes = c_bytes_per_stage * num_c_stage

    # Calculate A/B stages:
    # Start with total smem per CTA (capacity / occupancy)
    # Subtract reserved bytes and initial C stages bytes
    # Divide remaining by bytes needed per A/B stage
    num_ab_stage = (
        smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
    ) // ab_bytes_per_stage

    # Refine epilogue stages:
    # Calculate remaining smem after allocating for A/B stages and reserved bytes
    # Add remaining unused smem to epilogue
    if use_tma_store:
        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
    return num_acc_stage, num_ab_stage, num_c_stage

class PersistentAlphaDenseGemmKernel:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_tma_store: bool,
    ):
        """ Init config """
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.use_tma_store = use_tma_store
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        # warp specialization
        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # barrier specialization
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
    
    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """
        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
    
        # Compute mma/cluster/tile shapes
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # compute epilogue subtile
        if cutlass.const_expr(self.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]
        
        c_smem_layout = None
        if cutlass.const_expr(self.use_tma_store):
            c_smem_layout = sm100_utils.make_smem_layout_epi(
                self.c_dtype,
                self.c_layout,
                self.epi_tile,
                1,
            )
        
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = _compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.use_tma_store,
            c_smem_layout,  

        )

        # A/B/C smem layout
        self.a_smem_layout_staged = sm100.utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100.utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.c_smem_layout_staged = None
        if self.use_tma_store:
            self.c_smem_layout_staged = sm100.utils.make_smem_layout_epi(
                self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
            )
        
        # compute num tmem columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage,
        )
    
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes before smem/grid/tma computation
        - Setup TMA load/store atoms and tensors
        - Compute grid size with regard to hardware constraints
        - Define shared storage for kernel
        - Launch the kernel synchronously
        """
        # Setup static attributes before smem/grid/tma computation  
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        
        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id_.shape)


        def get_tma_ab_ops(cluster_shape_mn: Tuple[int, int], tiled_mma: cute.TiledMma):
            a_tma_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cluster_shape_mn, tiled_mma.thr_id,
            )
            
            b_tma_op = sm100_utils.cluster_shape_to_tma_atom_B(
            cluster_shape_mn, tiled_mma.thr_id,
            )
            return a_tma_op, b_tma_op

        
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        
        # TMA load for A
        # ensures the TMA operation is properly configured for the thread count 
        # and data distribution pattern of the specific MMA instruction
        # slice out single stage of A/B smem layout
        # for TMA transfer
        # or - where does one buffer of A data go in shared memory?
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        # Shape: (N, K, CTA_count, num_stages)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        
        a_tma_op, b_tma_op = get_tma_ab_ops(self.cluster_shape_mn, tiled_mma)
        
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_tma_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if a.element_type is cutlass.Float32 else None
            ),
        )

        # TMA load for B
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_tma_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if b.element_type is cutlass.Float32 else None
            ),
        )
        
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        
        # TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        if cutlass.const_expr(self.use_tma_store):
            epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0,1])

            tma_atom_c, tma_tensor_c = cute.nvgpu.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c,
                epi_smem_layout,
                self.epi_tile,
            )

        # grid size
        self.tile_sched_params, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, 
            self.cluster_shape_mn, max_active_clusters,
        )

        # Launch! 
        self.kernel(
            tiled_mma, 
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c if self.use_tma_store else c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,

        ).launch(
            grid = grid,
            block = [self.threads_per_cta, 1, 1],
            cluster = (*self.cluster_shape_mn, 1),
            stream = stream,
        )
        return
    
    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout, 
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent alpha dense GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        # prefetch tma descriptors
        def alpha_prefetch_tma_descriptors(tma_atom_list: list[cute.CopyAtom]):
            for tma_atom in tma_atom_list:
                cpasync.prefetch_descriptor(tma_atom)

        if warp_idx == self.tma_warp_id:
            alpha_prefetch_tma_descriptors([tma_atom_a, tma_atom_b])
            
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)
        
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # setup cta/thread coords
        # coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # coords inside cta
        tidx, _, _ = cute.arch.thread_idx()
        
        # setup smem/tmem
        # define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage *2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
        
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # init mainloop ab_pipeline barrier and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b -1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage = storage.ab_full_mbar_ptr.data_ptr(),
            num_stages = self.num_ab_stage,
            producer_group = ab_pipeline_producer_group,
            consumer_group = ab_pipeline_consumer_group,
            tx_count = self.num_tma_load_bytes,
            cta_layout_vmnk = cluster_layout_vmnk,
        ).make_participants()

        # init mainloop acc_pipeline barrier and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage = storage.acc_full_mbar_ptr.data_ptr(),
            num_stages = self.num_acc_stage,
            producer_group = acc_pipeline_producer_group,
            consumer_group = acc_pipeline_consumer_group,
            cta_layout_vmnk = cluster_layout_vmnk,
        )

        # init tmem alloc barrier
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id = self.tmem_alloc_sync_bar_id,
            num_threads = 32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id = self.tmem_dealloc_sync_bar_id,
                num_threads = 32 * len(self.epilog_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve = tmem_alloc_barrier,
            allocator_warp_id = self.epilog_warp_id[0],
            is_two_cta = use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr,
        )
        # Cluster arrive after barrier init
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()
        
        # setup smem tensor A/B/C/D
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type = self.a_dtype,
            layout = a_smem_layout_staged.outer,
            byte_alignment = 128,
            swizzle = a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type = self.b_dtype,
            layout = b_smem_layout_staged.outer,
            byte_alignment = 128,
            swizzle = b_smem_layout_staged.inner,
        )
        
        # compute multicast mask for A/B buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, 
                mcast_mode=2,
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, 
                mcast_mode=1,
            )
        
        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        
       # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
       )

        k_tile_cnt = cute.size(gA_mkl, mode=[3]) # RestK

        # partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)
        
        # partition global/shared tensor for TMA load A/B
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0,0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
    
        # partition shared/tmem tensor for TiledMMA_A/B/C
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        
        # Cluster wait before tensor memory alloc
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()
        
        # Alloc tensor memory buffer
        tmem.allocate(self.num_tmem_alloc_cols)
        
        # specialized TMA load warp
        if warp_idx == self.tma_warp_id:
            # persistent tile scheduling loop
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            
            while work_tile.is_valid_tile:
                # get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                
                # slice to per mma tile index
                # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)



def run (
    mnkl: Tuple[int, int, int, int],
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int]= (2,1),   
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    tolerance: float = 1e-01,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
):
    m, n, k, l = mnkl
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    a_tensor, b_tensor, c_tensor, a_torch_cpu, b_torch_cpu, c_torch_cpu, c_torch_gpu = (
        create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype)
    )

    gemm = PersistentAlphaDenseGemmKernel(
        acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store
    )

    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    '''compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
    '''

    def generate_tensors():
        a_tensor, _ = cutlass_torch.cute_tensor_like(
            a_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_tensor, _ = cutlass_torch.cute_tensor_like(
            b_torch_cpu, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        c_tensor, _ = cutlass_torch.cute_tensor_like(
            c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
        )

def mycute_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Tuple[int, int] = (2,1),
    use_2cta_instrs: bool = True,
    use_tma_store: bool = True,
    stream: cuda.CUstream = None,
) -> torch.Tensor:
    
    if not A.is_cuda:
        A = A.cuda()
    if not B.is_cuda:
        B = B.cuda()
    if not stream:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    
    L, M, K = A.shape
    _, N, K_B = B.shape

    if K != K_B:
        raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_B}")
    
    torch2cute_dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.TFloat32,
    }
    
    if A.dtype not in torch2cute_dtype_map:
        raise ValueError(f"Unsupported dtype: {A.dtype}. Supported: float16, bfloat16, float32")
    
    ab_dtype = torch2cute_dtype_map[A.dtype]
    c_dtype = cutlass.Float16 if A.dtype == torch.float16 else \
             cutlass.BFloat16 if A.dtype == torch.bfloat16 else \
             cutlass.Float32
    acc_dtype = cutlass.Float32
    if A.dtype not in torch2cute_dtype_map:
        raise ValueError(f"Unsupported dtype: {A.dtype}. Supported: float16, bfloat16, float32")
    
    ab_dtype = torch2cute_dtype_map[A.dtype]
    c_dtype = cutlass.Float16 if A.dtype == torch.float16 else \
             cutlass.Float32

    A = A.permute(1,2,0)
    B = B.permute(1,2,0)

    D_torch_cpu = cutlass_torch.matrix(1, M, N, False, c_dtype)  # is_mode0_major=False means N-major
    D_torch = D_torch_cpu.cuda()  # Shape is (M, N, L)
    
    
    A_cute = from_dlpack(A.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    B_cute = from_dlpack(B.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=1)
    C_cute = from_dlpack(D_torch, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    gemm = PersistentAlphaDenseGemmKernel(
        acc_dtype, use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, use_tma_store
    )
    
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    
    compiled_gemm = cute.compile(
        gemm,
        A_cute,
        B_cute,
        C_cute,
        max_active_clusters,
        stream,
    )
    
    compiled_gemm(A_cute, B_cute, C_cute, stream)
    
    torch.cuda.synchronize()
    
    D_torch = C_cute.permute(2,0,1).contiguous()
    return D_torch
    

def verify_results(output_torch, input_torch):
    # Verify results
    print("\nVerifying results...")
    torch.cuda.synchronize()
    
    result = output_torch.cpu()
    reference = input_torch.cpu()
    
    max_diff = torch.max(torch.abs(result - reference)).item()
    print(f"Max difference: {max_diff}")
    
    if torch.allclose(result, reference, atol=1e-5, rtol=1e-3):
        print("✓ PASS: Results match!")
    else:
        print("✗ FAIL: Results do not match!")
        print(f"Reference sample:\n{reference[:4, :4]}")
        print(f"Result sample:\n{result[:4, :4]}")
    
    print("=" * 80)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example of Alpha Dense Persistent GEMM on Blackwell."
    )
    parser.add_argument("--mnkl", type=Tuple[int, int, int, int], default=(256, 256, 512, 1), help="mnkl dimensions")
    parser.add_argument("--mma_tiler_mn", type=Tuple[int, int], default=(128, 128), help="Mma tile shape")
    parser.add_argument("--cluster_shape_mn", type=Tuple[int, int], default=(1, 1), help="Cluster shape")
    parser.add_argument("--ab_dtype", type=cutlass.dtype, default=cutlass.Float16, help="AB dtype")
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.Float16, help="C dtype")
    parser.add_argument("--acc_dtype", type=cutlass.dtype, default=cutlass.Float32, help="Accumulator dtype")
    parser.add_argument(
        "--use_2cta_instrs",
        action="store_true",
        help="Enable 2CTA MMA instructions feature",
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument(
        "--use_tma_store", action="store_true", help="Use tma store or not"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-01, help="Tolerance for validation"
    )
    parser.add_argument(
        "--warmup_iterations", type=int, default=0, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")

    if len(args.mma_tiler_mn) != 2:
        parser.error("--mma_tiler_mn must contain exactly 2 values")

    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    '''exec_time_us = run(
        args.mnkl,
        args.ab_dtype,
        args.c_dtype,
        args.d_dtype,
        args.acc_dtype,
        args.epi_dtype,
        args.alpha,
        args.beta,
        args.mma_tiler_mn,
        args.cluster_shape_mn,
        args.use_2cta_instrs,
        args.use_tma_store,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
    )
    '''

