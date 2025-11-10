"""
Simple TMA Copy Kernel Example for Blackwell (SM100)

This example demonstrates the basic usage of Tensor Memory Access (TMA) operations
as used in the dense_gemm_persistent.py kernel, but in a simplified copy context.

Key TMA concepts demonstrated:
1. TMA descriptor creation and prefetching
2. TMA load from global memory (GMEM) to shared memory (SMEM)
3. Barrier synchronization with TMA
4. TMA store from shared memory back to global memory

Requirements:
- CUDA-capable GPU (Blackwell/SM100 architecture preferred)
- PyTorch with CUDA support
- CUTLASS Python bindings
- CuteDSL
"""

try:
    import torch
    import cuda.bindings.driver as cuda
    
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.torch as cutlass_torch
    from cutlass.cute.nvgpu import cpasync
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease ensure you have installed:")
    print("  - PyTorch with CUDA support")
    print("  - CUTLASS Python bindings")
    print("  - CuteDSL")
    raise


class SimpleTMACopyKernel:
    """
    A simple kernel that copies data from input to output using TMA operations.
    
    Flow:
    1. TMA Load: Global Memory (input) -> Shared Memory
    2. TMA Store: Shared Memory -> Global Memory (output)
    
    :param tile_shape: Shape of the tile to copy (M, N)
    :type tile_shape: Tuple[int, int]
    :param cluster_shape: Cluster dimensions for parallel processing
    :type cluster_shape: Tuple[int, int]
    """
    
    def __init__(self, tile_shape=(128, 128), cluster_shape=(1, 1)):
        self.tile_shape = tile_shape
        self.cluster_shape = cluster_shape
        self.num_stages = 2  # Pipeline stages for double buffering
        self.threads_per_cta = 128  # Threads per CTA
        
    @cute.jit
    def __call__(
        self,
        input_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """
        Execute the TMA copy operation.
        
        :param input_tensor: Input tensor to copy from
        :type input_tensor: cute.Tensor
        :param output_tensor: Output tensor to copy to
        :type output_tensor: cute.Tensor
        :param stream: CUDA stream for execution
        :type stream: cuda.CUstream
        """
        # Get tensor properties
        self.dtype = input_tensor.element_type
        m, n = input_tensor.shape[0], input_tensor.shape[1]
        
        # Create shared memory layout for staging
        # Shape: (TILE_M, TILE_N, STAGES)
        # In CuTe, shapes are just tuples - no make_shape() function
        smem_layout = cute.make_layout(
            (self.tile_shape[0], self.tile_shape[1], self.num_stages)
        )
        
        # Setup TMA load descriptor (Global -> Shared)
        # Get one stage of the smem layout by slicing the last dimension
        smem_layout_one_stage = cute.make_layout(
            (self.tile_shape[0], self.tile_shape[1])
        )
        
        # Use a simple copy operation for TMA
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        
        # Create TMA atom and partitioned tensors for loading
        tma_atom_load, tma_tensor_load = cpasync.make_tiled_tma_atom(
            tma_load_op,
            input_tensor,
            smem_layout_one_stage,
            self.tile_shape
        )
        
        # Setup TMA store descriptor (Shared -> Global)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
        
        tma_atom_store, tma_tensor_store = cpasync.make_tiled_tma_atom(
            tma_store_op,
            output_tensor,
            smem_layout_one_stage,
            self.tile_shape
        )
        
        # Calculate grid dimensions
        num_tiles_m = (m + self.tile_shape[0] - 1) // self.tile_shape[0]
        num_tiles_n = (n + self.tile_shape[1] - 1) // self.tile_shape[1]
        
        grid = (num_tiles_m * num_tiles_n, 1, 1)
        
        # Launch kernel - don't pass tma_tensor_* to avoid serialization issues
        # Pass original tensors instead and partition inside kernel
        self.kernel(
            tma_atom_load,
            input_tensor,
            tma_atom_store,
            output_tensor,
            self.tile_shape,
            self.num_stages,
            (num_tiles_m, num_tiles_n),
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape, 1),
            stream=stream,
        )
        
    @cute.kernel
    def kernel(
        self,
        tma_atom_load: cute.CopyAtom,
        input_tensor: cute.Tensor,
        tma_atom_store: cute.CopyAtom,
        output_tensor: cute.Tensor,
        tile_shape: tuple,
        num_stages: int,
        num_tiles_mn: tuple,
    ):
        """
        GPU kernel that performs TMA copy operations.
        """
        # Get thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        
        # Compute which tile this block is responsible for
        tile_idx = bidx
        num_tiles_m, num_tiles_n = num_tiles_mn
        tile_m = tile_idx // num_tiles_n
        tile_n = tile_idx % num_tiles_n
        
        # Reconstruct smem_layout inside kernel to avoid serialization issues
        smem_layout = cute.make_layout((tile_shape[0], tile_shape[1], num_stages))
        
        #
        # Setup shared memory
        #
        @cute.struct
        class SharedStorage:
            # Barriers for TMA load synchronization
            load_barriers: cute.struct.MemRange[cutlass.Int64, num_stages]
            # Barriers for TMA store synchronization  
            store_barriers: cute.struct.MemRange[cutlass.Int64, num_stages]
        
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        
        # Allocate shared memory tensor for data
        # Shape: (TILE_M, TILE_N, STAGES)
        smem_tensor = smem.allocate_tensor(
            element_type=self.dtype,
            layout=smem_layout,
            byte_alignment=128,
        )
        
        #
        # Prefetch TMA descriptors (important for performance)
        #
        if tidx == 0:
            cpasync.prefetch_descriptor(tma_atom_load)
            cpasync.prefetch_descriptor(tma_atom_store)
        
        #
        # Setup pipeline for load operations
        #
        load_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        load_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        
        # Calculate transaction size for TMA load
        tx_count = cute.size_in_bytes(
            self.dtype, 
            smem_tensor[(None, None, 0)]
        )
        
        # Create TMA pipeline
        load_producer, load_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.load_barriers.data_ptr(),
            num_stages=num_stages,
            producer_group=load_producer_group,
            consumer_group=load_consumer_group,
            tx_count=tx_count,
        ).make_participants()
        
        # Synchronize threads after barrier init
        cute.arch.sync_threads()
        
        #
        # Partition global tensors for this tile using local_tile
        #
        # Create local tile views for this CTA's region of work
        input_tile = cute.local_tile(
            input_tensor,
            tile_shape,
            (tile_m, tile_n)
        )
        output_tile = cute.local_tile(
            output_tensor,
            tile_shape,
            (tile_m, tile_n)
        )
        
        #
        # Only thread 0 performs TMA operations
        #
        if tidx == 0:
            # Initialize pipeline
            load_producer.reset()
            
            # Acquire barrier for first stage
            handle = load_producer.acquire_and_advance()
            
            # Issue TMA load from global to shared memory
            cute.copy(
                tma_atom_load,
                input_tile,
                smem_tensor[(None, None, handle.index)],
                tma_bar_ptr=handle.barrier,
            )
            
        #
        # Wait for TMA load to complete
        #
        load_consumer.reset()
        handle = load_consumer.wait_and_advance()
        
        # At this point, data is in shared memory at stage handle.index
        # In a real kernel, you would process the data here
        
        #
        # Issue TMA store from shared memory to global memory
        #
        if tidx == 0:
            cute.copy(
                tma_atom_store,
                smem_tensor[(None, None, handle.index)],
                output_tile,
            )
            
            # Signal that we're done with this stage
            handle.release()
        
        # Synchronize before kernel exit
        cute.arch.sync_threads()


def run_tma_copy_example():
    """
    Run the TMA copy example with a simple test case.
    """
    print("=" * 80)
    print("Simple TMA Copy Kernel Example")
    print("=" * 80)
    
    # Problem size
    m, n = 256, 256
    tile_shape = (128, 128)
    cluster_shape = (1, 1)
    dtype = cutlass.Float16
    
    print(f"\nProblem size: M={m}, N={n}")
    print(f"Tile shape: {tile_shape}")
    print(f"Cluster shape: {cluster_shape}")
    print(f"Data type: {dtype}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run this example!")
    
    # Create input and output tensors
    torch.manual_seed(42)
    input_torch = torch.randn(m, n, dtype=torch.float16, device='cuda')
    output_torch = torch.zeros_like(input_torch)
    
    # Convert to CUTE tensors
    input_tensor, _ = cutlass_torch.cute_tensor_like(
        input_torch, dtype, is_dynamic_layout=True, assumed_align=16
    )
    output_tensor, output_torch = cutlass_torch.cute_tensor_like(
        output_torch, dtype, is_dynamic_layout=True, assumed_align=16
    )
    
    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    # Create and compile kernel
    print("\nCompiling kernel...")
    copy_kernel = SimpleTMACopyKernel(tile_shape=tile_shape, cluster_shape=cluster_shape)
    compiled_kernel = cute.compile(copy_kernel, input_tensor, output_tensor, current_stream)
    
    # Run kernel
    print("Executing kernel...")
    compiled_kernel(input_tensor, output_tensor, current_stream)
    
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
    run_tma_copy_example()