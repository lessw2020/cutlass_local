
import argparse
from typing import Optional, Type, Tuple, Union
import time
import torch
import cuda.bindings.driver as cuda
import os

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute import testing as cute_testing

@cute.kernel
def copy_kernel(S: cute.Tensor, D: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    num = cute.size(S, mode=[2])
    block_start = bidx * bdimx + tidx
    x = block_start // num
    y = block_start % num
    # Slice into the tiled tensors
    block_coordinate = ((None, None), x, y)
    tile_S = S[block_coordinate]
    tile_D = D[block_coordinate]

    #print("Block Tile:")
    #print(f"tile_S = {tile_S}")
    #print(f"tile_D = {tile_D}")

    fragment = cute.make_fragment_like(tile_S)

    #print("Fragment:")
    #print(f"fragment = {fragment}")

    fragment.store(tile_S.load())
    tile_D.store(fragment.load())
    return
    

@cute.jit
def launch_copy(
    tensor_S: cute.Tensor,
    tensor_D: cute.Tensor,
    block_shape: cute.Shape,
    num_threads: cutlass.Constexpr[cutlass.Int32],
    stream: cuda.CUstream,
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

    copy_kernel(tiled_tensor_S, tiled_tensor_D).launch(
        grid=grid_dim, block=block_dim, stream=stream
    )
    return

@cute.kernel
def copy_kernel2(S: cute.Tensor, D: cute.Tensor, ThreadLayout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Map 1D grid to 2D tile coordinates
    tiles_n = cute.size(S, mode=[2])
    m_idx = bidx // tiles_n
    n_idx = bidx % tiles_n
    block_coordinate = ((None, None), m_idx, n_idx)
    tile_S = S[block_coordinate]
    tile_D = D[block_coordinate]

    # print("Block Tile:")
    # print(f"tile_S = {tile_S}")
    # print(f"tile_D = {tile_D}")

    thr_tile_S = cute.local_partition(tile_S, ThreadLayout, tidx)
    thr_tile_D = cute.local_partition(tile_D, ThreadLayout, tidx)

    # print("Thread Tile:")
    # print(f"thr_tile_S = {thr_tile_S}")
    # print(f"thr_tile_D = {thr_tile_D}")

    fragment = cute.make_fragment_like(thr_tile_S)

    # print("Fragment:")
    # print(f"fragment = {fragment}")

    fragment.store(thr_tile_S.load())
    thr_tile_D.store(fragment.load())
    return

@cute.jit
def launch_copy2(
    tensor_S: cute.Tensor,  # Pointer to Source
    tensor_D: cute.Tensor,  # Pointer to Destination
    block_shape: cute.Shape,  # (M, N)
):
    print("Tensors:")
    print(f"tensor_S = {tensor_S}")
    print(f"tensor_D = {tensor_D}")

    # Divide into block tiles: ((M, N), m', n')
    tiled_tensor_S = cute.tiled_divide(tensor_S, block_shape)
    tiled_tensor_D = cute.tiled_divide(tensor_D, block_shape)

    print("Block Tile Tensor:")
    print(f"tiled_tensor_S = {tiled_tensor_S}")
    print(f"tiled_tensor_D = {tiled_tensor_D}")

    # Use one thread per element in the block tile
    thr_layout = cute.make_layout(block_shape, stride=(block_shape[1], 1))

    print("Thread Layout:")
    print(f"thr_layout = {thr_layout}")

    tiles_m = cute.size(tiled_tensor_D, mode=[1])
    tiles_n = cute.size(tiled_tensor_D, mode=[2])
    grid_dim = (tiles_m * tiles_n, 1, 1)
    block_dim = (cute.size(thr_layout), 1, 1)

    print("Grid and Block Configuration:")
    print(f"grid_dim = {grid_dim}")
    print(f"block_dim = {block_dim}")

    copy_kernel2(tiled_tensor_S, tiled_tensor_D, thr_layout).launch(
        grid=grid_dim, block=block_dim
    )
    return

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    print(f"Starting...")

    tensor_shape = (8192, 8192)
    block_shape = (1, 16)
    num_threads = 256
    # block_shape = (32, 256)
    thread_shape = (8, 32)

    S = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    D = torch.zeros(8192, 8192, device="cuda", dtype=torch.bfloat16)

    tensor_S = from_dlpack(S, assumed_align=16)
    tensor_D = from_dlpack(D, assumed_align=16)

    # Create a CUDA stream compatible with CuTeDSL (pattern from Mbarrier example)
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    # Measure compile time (host wall time)
    '''t0 = time.perf_counter()
    compiled_launch_copy = cute.compile(
        launch_copy2,
        tensor_S,
        tensor_D,
        block_shape,
        thread_shape,
        # current_stream,
    )
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(f"Compile time: {compile_ms:.3f} ms")
    '''

    # Benchmark using CuTeDSL helper (mirrors GEMM example)
    compiled = cute.compile(
        launch_copy2, tensor_S, tensor_D, block_shape
    )
    kernel_args = cute_testing.JitArguments(
        tensor_S, tensor_D, block_shape
    )
    time_us = cute_testing.benchmark(
        compiled,
        kernel_arguments=kernel_args,
        warmup_iterations=10,
        iterations=100,
    )
    time_ms = time_us / 1000.0
    print(f"Kernel time: {time_ms:.3f} ms")

    # Bandwidth calculation (GB/s and % of theoretical)
    bytes_moved = (S.numel() + D.numel()) * S.element_size()  # read + write
    gbps = (bytes_moved / (time_us * 1e-6)) / 1e9
    print(f"Achieved bandwidth: {gbps:.2f} GB/s")

    # Estimate theoretical peak from device attributes (approx)
    hw = cutlass.utils.HardwareInfo()
    # For B200 (Blackwell), adjust this if you have exact peak BW; placeholder example:
    # Using L2 size doesn’t give BW; if you know peak (e.g., ~5 TB/s HBM), set here.
    # As a placeholder, derive from env or assume 5100 GB/s.
    theoretical_gbps = float(
        int(os.environ.get("B200_PEAK_GBPS", "5100"))
    )
    pct = min(100.0, gbps / theoretical_gbps * 100.0)
    print(f"Bandwidth utilization: {pct:.2f}% of {theoretical_gbps:.0f} GB/s")

    torch.testing.assert_close(S, D)
    print("Sucess!")
    print(S[0:4, 0:4])
    print(D[0:4, 0:4])