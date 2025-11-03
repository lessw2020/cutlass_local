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

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange(cutlass.Int64, ab_stages * 2)
    acc_mbar_ptr: cute.struct.MemRange(cutlass.Int64, acc_stages*2)
    tmem_holding_buff: cutlass.Int32

@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TileMma, 
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor, 
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.CompostedLayout,
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

    