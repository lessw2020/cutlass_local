#!/bin/bash

# 🚀 OPTIMIZED BLOCKSCALED GEMM BENCHMARK SCRIPT 🚀
#
# This script runs a comparison between the baseline and optimized kernels
# for various problem sizes and configurations.

echo "======================================================================"
echo " BASELINE vs OPTIMIZED BLOCKSCALED GEMM BENCHMARK"
echo "======================================================================"
echo ""

# Default: Medium problem size with NVF4
echo "[Test 1/3] Medium problem (8K x 8K x 1K), NVF4, N=128"
python faster_dense_blockscaled_gemm_persistent.py \
  --benchmark_mode \
  --mnkl 8192,8192,1024,1 \
  --mma_tiler_mn 256,128 \
  --cluster_shape_mn 2,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16 \
  --warmup_iterations 10 \
  --iterations 50

echo ""
echo "======================================================================"
echo ""

# Large N dimension (should show bigger gains from epilogue optimization)
echo "[Test 2/3] Medium problem (8K x 8K x 1K), NVF4, N=256"
python faster_dense_blockscaled_gemm_persistent.py \
  --benchmark_mode \
  --mnkl 8192,8192,1024,1 \
  --mma_tiler_mn 256,256 \
  --cluster_shape_mn 2,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16 \
  --warmup_iterations 10 \
  --iterations 50

echo ""
echo "======================================================================"
echo ""

# Small problem (for quick testing)
echo "[Test 3/3] Small problem (2K x 2K x 512), NVF4, N=128"
python faster_dense_blockscaled_gemm_persistent.py \
  --benchmark_mode \
  --mnkl 2048,2048,512,1 \
  --mma_tiler_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16 \
  --warmup_iterations 5 \
  --iterations 30

echo ""
echo "======================================================================"
echo " BENCHMARK COMPLETE!"
echo "======================================================================"

