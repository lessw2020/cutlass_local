#!/usr/bin/env python3

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

class SimpleMbarrierKernel:
    """
    A toy example demonstrating mbarrier usage in CUTE DSL.
    Shows a simple producer-consumer pattern where:
    - Producer threads (first half) write data to shared memory
    - Consumer threads (second half) read data from shared memory  
    - mbarrier ensures all producers finish before consumers start
    """
    
    def __init__(self, num_threads_per_block=64):
        self.num_threads_per_block = num_threads_per_block
        self.num_elements = num_threads_per_block  # One element per thread
        
    @cute.jit
    def __call__(self, input_tensor: cute.Tensor, output_tensor: cute.Tensor, stream: cuda.CUstream):
        """
        Launch the kernel that demonstrates mbarrier synchronization
        """
        # Define shared storage
        @cute.struct
        class SharedStorage:
            # Barrier storage - needs memory for the barrier state
            sync_barrier: cutlass.Int64
            # Shared memory buffer for data exchange
            shared_data: cute.struct.MemRange[Float32, self.num_elements]
        
        self.shared_storage = SharedStorage
        
        # Launch kernel
        grid_size = 1  # Single block for simplicity
        block_size = [self.num_threads_per_block, 1, 1]
        
        self.kernel(input_tensor, output_tensor).launch(
            grid=[grid_size, 1, 1],
            block=block_size,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )
    
    @cute.kernel
    def kernel(self, input_tensor: cute.Tensor, output_tensor: cute.Tensor):
        """
        GPU kernel demonstrating mbarrier usage
        """
        # Get thread and block information
        tidx, _, _ = cute.arch.thread_idx()
        
        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        
        # Get barrier and shared data pointers
        barrier_ptr = storage.sync_barrier
        shared_data = storage.shared_data.get_tensor((self.num_elements,))
        
        # Determine if this thread is a producer or consumer
        half_threads = self.num_threads_per_block // 2
        is_producer = tidx < half_threads
        
        # Initialize barrier (only one thread should do this)
        if tidx == 0:
            # We expect all threads to participate in the barrier
            cute.arch.mbarrier_init(barrier_ptr, self.num_threads_per_block)
        
        # Make sure barrier is initialized before proceeding
        cute.arch.sync_warp()
        
        
        if is_producer:
            # PRODUCER PHASE: Write data to shared memory
            # Each producer writes to their corresponding location
            producer_idx = tidx
            input_val = input_tensor[producer_idx]
            shared_data[producer_idx] = input_val * 2.0  # Simple transformation
            
            # Ensure write is visible to other threads
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, 
                space=cute.arch.SharedSpace.shared_cta
            )
            
            # Signal that this producer is done
            cute.arch.mbarrier_arrive(barrier_ptr, 0)
            
            # Producer can also wait if needed (for demonstration)
            cute.arch.mbarrier_wait(barrier_ptr, 0)

            # Write result to output after synchronization to match expected values
            shared_val = shared_data[producer_idx]
            output_tensor[tidx] = shared_val + 1.0
            
        else:
            # CONSUMER PHASE: Wait for all producers, then read data
            # First, arrive at barrier (consumers participate too)
            cute.arch.mbarrier_arrive(barrier_ptr, 0)
            
            # Wait for all threads (producers + consumers) to arrive
            cute.arch.mbarrier_wait(barrier_ptr, 0)
            
            # Now safe to read from shared memory
            consumer_idx = tidx - half_threads  # Map to producer's index
            shared_val = shared_data[consumer_idx]
            
            # Write result to output (each consumer writes to their output location)
            output_tensor[tidx] = shared_val + 1.0  # Another transformation


def run_mbarrier_example():
    """
    Run the mbarrier demonstration
    """
    print("Running mbarrier toy example...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping example")
        return
    
    # Setup
    num_threads = 64
    kernel = SimpleMbarrierKernel(num_threads)
    
    # Create input and output tensors
    input_data = torch.arange(num_threads//2, dtype=torch.float32, device='cuda')
    output_data = torch.zeros(num_threads, dtype=torch.float32, device='cuda')
    
    # Ensure CUDA driver context is initialized
    cutlass.cuda.initialize_cuda_context()

    # Wrap framework tensors via DLPack; layout inferred dynamically
    input_cute = from_dlpack(input_data).mark_layout_dynamic()
    output_cute = from_dlpack(output_data).mark_layout_dynamic()

    # Get CUDA stream
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Compile (manages its own MLIR context) and run
    compiled_kernel = cute.compile(kernel, input_cute, output_cute, current_stream)
    compiled_kernel(input_cute, output_cute, current_stream)
    
    # Wait for completion
    torch.cuda.synchronize()
    
    # Check results
    print("Input (first half of threads):", input_data.cpu().numpy())
    print("Output (all threads):")
    output_cpu = output_data.cpu().numpy()
    
    # Expected: producers write input*2 to shared memory
    # consumers read shared data and add 1
    # So output should be: [input*2+1 for producers] + [input*2+1 for consumers]
    for i in range(num_threads//2):
        expected_producer = input_data[i].item() * 2 + 1  # Producer transformation + consumer offset
        expected_consumer = input_data[i].item() * 2 + 1  # Same data, consumer adds 1
        print(f"  Thread {i} (producer): {output_cpu[i]:.1f}, expected: {expected_producer:.1f}")
        print(f"  Thread {i+num_threads//2} (consumer): {output_cpu[i+num_threads//2]:.1f}, expected: {expected_consumer:.1f}")
    
    print("\nDemonstration complete!")
    print("Key points:")
    print("1. mbarrier_init() set up barrier for all 64 threads")
    print("2. All threads called mbarrier_arrive() to signal completion of their phase")
    print("3. All threads called mbarrier_wait() to wait until everyone arrived")
    print("4. Only after all 64 threads arrived did any thread proceed past the barrier")
    print("5. This ensured producers finished writing before consumers started reading")


if __name__ == "__main__":
    run_mbarrier_example()
