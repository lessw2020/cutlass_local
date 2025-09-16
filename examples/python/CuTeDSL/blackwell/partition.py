
import argparse
from typing import Optional, Type, Tuple, Union
import time
import torch
import cuda.bindings.driver as cuda


import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack

@cute.kernel
def copy_kernel(S: cute.Tensor, D: cute.Tensor):
    
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    num = cute.size(S, mode=[2])
    block_start = bidx * bdimx + tidx
    x = block_start // num
    y = block_start % num
    # Predicated work to avoid out-of-range tile access (no early return)
    #valid = (x < cute.size(S, mode=[1])) and (y < cute.size(S, mode=[2]))
    #if valid:
    block_coordinate = ((None, None), x, y)
    tile_S = S[block_coordinate]
    tile_D = D[block_coordinate]

    fragment = cute.make_fragment_like(tile_S)
    fragment.store(tile_S.load())
    tile_D.store(fragment.load())
    #return
    

@cute.jit
def launch_copy(
    tensor_S: cute.Tensor, 
    tensor_D: cute.Tensor, 
    block_shape: cute.Shape,  
    num_threads: cutlass.Constexpr[cutlass.Int32],
):
    print("Tensors:")
    print(f"tensor_S = {tensor_S}")
    print(f"tensor_D = {tensor_D}")

    # Tile (m, n) by (M, N) to obtain ((M, N), m', n')
    # , where M' and N' are the number of block tiles
    tiled_tensor_S = cute.tiled_divide(tensor_S, block_shape)  # (M, N), m', n')
    tiled_tensor_D = cute.tiled_divide(tensor_D, block_shape)  # (M, N), m', n')

    print("Block Tile Tensor:")
    print(f"tiled_tensor_S = {tiled_tensor_S}")
    print(f"tiled_tensor_D = {tiled_tensor_D}")

    total_tiles = (
        cute.size(tiled_tensor_D, mode=[1]) * cute.size(tiled_tensor_D, mode=[2])
    )
    grid_x = (total_tiles + num_threads - 1) // num_threads  # ceil-div
    grid_dim = (grid_x, 1, 1)
    block_dim = (num_threads, 1, 1)

    print("Grid and Block Configuration:")
    print(f"grid_dim = {grid_dim}")
    print(f"block_dim = {block_dim}")

    copy_kernel(tiled_tensor_S, tiled_tensor_D).launch(grid=grid_dim, block=block_dim)
    return

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    print(f"Starting...")

    tensor_shape = (8192, 8192)
    block_shape = (1, 16)
    num_threads = 256

    S = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    D = torch.zeros(8192, 8192, device="cuda", dtype=torch.bfloat16)

    tensor_S = from_dlpack(S, assumed_align=16)
    tensor_D = from_dlpack(D, assumed_align=16)

    # Measure compile time (host wall time)
    # t0 = time.perf_counter()
    '''compiled_launch_copy = cute.compile(
        launch_copy,
        tensor_S,
        tensor_D,
        block_shape,
        num_threads,
    )'''
    # compile_ms = (time.perf_counter() - t0) * 1e3
    # print(f"Compile time: {compile_ms:.3f} ms")

    # Measure kernel execution time with CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    launch_copy(tensor_S, tensor_D, block_shape, num_threads)
    end_evt.record()
    torch.cuda.synchronize()
    kernel_ms = start_evt.elapsed_time(end_evt)
    print(f"Kernel time: {kernel_ms:.3f} ms")

    torch.testing.assert_close(S, D)
    print("Sucess!")
    print(S[0:4, 0:4])
    print(D[0:4, 0:4])