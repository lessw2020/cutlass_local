import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from math import ceil
import torch
import torch.nn as nn

def benchmark(callable, mX, mW, mY,  
              num_tokens, hidden_dim, epsilon):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(mX, mW, mY),
        warmup_iterations=100,
        iterations=1000,
    )

    # Calculate metrics
    # ----------------
    dtype = mX.element_type

    # Calculate total bytes transferred:
    # - 1 Read and 1 Write
    # - Each element is dtype.width bits
    bytes_per_element = dtype.width // 8
    total_bytes = hidden_dim * num_tokens * 2 * bytes_per_element

    # Calculate achieved bandwidth
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s
    tbs = achieved_bandwidth / 1000  # Convert GB/s to TB/s

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")
    print(f"Memory throughput: {tbs:.4f} TB/s")

@cute.kernel
def rms_norm_kernel(mX: cute.Tensor, mW: cute.Tensor, mY: cute.tensor, threads_per_block: cutlass.Constexpr, 
                    num_tokens: cutlass.Constexpr, hidden_dim: cutlass.Constexpr, epsilon: cutlass.Constexpr
):
  allocator = cutlass.utils.SmemAllocator()
  layout = cute.make_layout((threads_per_block))
  scalar_layout = cute.make_layout((1))

  sdata = allocator.allocate_tensor(cutlass.Float32, layout=layout, byte_alignment=16, swizzle=None)
  squared_reduce = allocator.allocate_tensor(cutlass.Float32, layout=scalar_layout)

  tidx, _, _ = cute.arch.thread_idx()
  bidx, _, _ = cute.arch.block_idx()
   
  block_sum = 0.0
  for i in range(tidx, hidden_dim, threads_per_block, unroll_full=True):
    x_ = mX[(bidx, i)]
    block_sum += x_ * x_ 
  
  sdata[tidx] = block_sum
  cute.arch.sync_threads()

  if tidx < 128:
    sdata[tidx] += sdata[tidx + 128]
  cute.arch.sync_threads()
  
  if tidx < 64:
    sdata[tidx] += sdata[tidx + 64]
  cute.arch.sync_threads()
  
  if tidx < 32:
    sdata[tidx] += sdata[tidx + 32]
    res = cute.arch.warp_reduction_sum(sdata[tidx], threads_in_group=32)
    if tidx == 0:
      squared_reduce[0] = cute.math.rsqrt(res/hidden_dim + epsilon, fastmath=True)

  cute.arch.sync_threads()
  
  rms = squared_reduce[0]
  for i in range(tidx, hidden_dim, threads_per_block, unroll_full=True):
    mY[(bidx, i)] = mX[(bidx, i)] * rms * mW[i] 

@cute.jit
def rms_norm_(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor, 
              num_tokens: cutlass.Constexpr, hidden_dim: cutlass.Constexpr, epsilon: cutlass.Constexpr
):
  threads_per_block = 256
  rms_norm_kernel(mX, mW, mY, threads_per_block, num_tokens, hidden_dim, epsilon).launch(
    grid=(num_tokens,1,1),
    block=(threads_per_block,1,1)
  )

if __name__ == "__main__":
    num_tokens = 65536
    hidden_dim = 1024 
    eps = 1e-5

    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)    
    w = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    y = torch.zeros(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)

    rms_norm = nn.RMSNorm(hidden_dim, eps=eps)
    rms_norm.weight.data = w 
    y_ref = rms_norm(x)

    mX = from_dlpack(x, assumed_align=16)
    mW = from_dlpack(w, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)

    rms_norm_(mX, mW, mY, num_tokens, hidden_dim, eps)
    torch.testing.assert_close(y_ref, y)

    compiled = cute.compile(rms_norm_, mX, mW, mY, num_tokens, hidden_dim, eps)
    benchmark(compiled, mX, mW, mY, num_tokens, hidden_dim, eps)