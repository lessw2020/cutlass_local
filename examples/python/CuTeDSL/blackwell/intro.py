
import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack
import torch
from functools import partial
import time


@cute.kernel
def naive_elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor
    m, n = gA.shape
    ni = thread_idx % n
    mi = thread_idx // n

    # Map logical index to physical address via tensor layout
    a_val = gA[mi, ni]
    b_val = gB[mi, ni]

    # Perform element-wise addition
    gC[mi, ni] = a_val + b_val

@cute.jit
def naive_elementwise_add(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor
):
    num_threads_per_block = 256

    m, n = mA.shape
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(grid=((m * n) // num_threads_per_block, 1, 1),
                  block=(num_threads_per_block, 1, 1))



def benchmark(callable, *, num_warmups, num_iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    for _ in range(num_warmups):
        callable()

    start_event.record(stream=torch.cuda.current_stream())
    for _ in range(num_iterations):
        callable()
    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iterations

    print(f"Average execution time: {avg_time:.4f} ms")
    print(f"Throughput: {(3 * a.numel() * 2) / (avg_time / 1000) / 1e9:.2f} GB/s")

if __name__ == "__main__":
    print("Running naive elementwise add")
    M, N = 2048, 2048

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)
    b_ = from_dlpack(b, assumed_align=16)
    c_ = from_dlpack(c, assumed_align=16)

    # Compile kernel with timing
    _t0 = time.perf_counter()
    naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)
    _t1 = time.perf_counter()
    print(f"cute.compile time: {(_t1 - _t0) * 1000:.2f} ms")
    naive_elementwise_add_(a_, b_, c_)

    # verify correctness
    torch.testing.assert_close(c, a + b)

    benchmark(partial(naive_elementwise_add_, a_, b_, c_), num_warmups=5, num_iterations=100)

'''Running naive elementwise add
cute.compile time: 148.01 ms
Average execution time: 0.0124 ms
Throughput: 2031.76 GB/s
'''