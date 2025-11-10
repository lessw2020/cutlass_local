# Profiling the BlockScaled GEMM Kernel

## Why Profile?

Our optimization attempts failed because we made assumptions about bottlenecks. Profiling gives us **real data** about:
- Where time is actually spent
- TMA efficiency and latency
- Warp stall reasons (memory, sync, compute)
- SMEM bank conflicts
- Instruction pipeline utilization

## Quick Profiling with Nsight Compute

### 1. Generate the kernel binary

```bash
cd /home/less/cutlass_local/examples/python/CuTeDSL/blackwell

# Run once to compile and cache the kernel
python dense_blockscaled_gemm_persistent.py \
  --mnkl 8192,8192,1024,1 \
  --mma_tiler_mn 256,128 \
  --cluster_shape_mn 2,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16 \
  --skip_ref_check \
  --iterations 1
```

### 2. Profile with NCU (Basic Metrics)

```bash
ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  -o baseline_profile \
  python dense_blockscaled_gemm_persistent.py \
    --mnkl 8192,8192,1024,1 \
    --mma_tiler_mn 256,128 \
    --cluster_shape_mn 2,1 \
    --ab_dtype Float4E2M1FN \
    --sf_dtype Float8E8M0FNU \
    --sf_vec_size 16 \
    --c_dtype Float16 \
    --skip_ref_check \
    --iterations 1
```

This creates `baseline_profile.ncu-rep` that you can open in Nsight Compute UI.

### 3. Profile with Specific Sections (Faster)

```bash
ncu \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section Occupancy \
  --target-processes all \
  --force-overwrite \
  -o baseline_quick \
  python dense_blockscaled_gemm_persistent.py \
    --mnkl 8192,8192,1024,1 \
    --mma_tiler_mn 256,128 \
    --cluster_shape_mn 2,1 \
    --ab_dtype Float4E2M1FN \
    --sf_dtype Float8E8M0FNU \
    --sf_vec_size 16 \
    --c_dtype Float16 \
    --skip_ref_check \
    --iterations 1
```

### 4. Key Metrics to Check

Open the `.ncu-rep` file and look for:

#### **Speed of Light (SOL)**
- **SM Throughput**: How busy are the SMs? (Target: >80%)
- **Memory Throughput**: Are we memory bound? (Target: >70%)
- **Compute Throughput**: Are Tensor Cores busy? (Target: >60%)

#### **Warp Stall Reasons**
- **Memory Throttle**: Waiting on TMA/global loads
- **Barrier**: Waiting on __syncthreads() or TMA barriers
- **MIO Throttle**: Waiting on shared memory accesses
- **Math Pipe Throttle**: Waiting on tensor core operations

#### **Memory Workload**
- **L2 Hit Rate**: Are we reusing data effectively?
- **TMA Efficiency**: Bytes transferred vs bytes requested
- **SMEM Bank Conflicts**: Are layouts causing conflicts?

#### **Occupancy**
- **Theoretical Occupancy**: Based on resource usage
- **Achieved Occupancy**: Actual warps active
- **Limiting Factor**: What's preventing higher occupancy? (SMEM, registers, warps)

## What to Look For

### If Memory Bound (Most Likely)
- **Warp stalls on memory > 30%**: TMA latency is the issue
  - ✅ Baseline already uses greedy AB staging (good!)
  - 🔍 Check if TMA multicast is working correctly
  - 🔍 Check if we're hitting L2 cache effectively

### If Compute Bound (Unlikely for GEMM)
- **Warp stalls on math pipe > 20%**: Tensor cores are bottleneck
  - 🔍 Check if we're using optimal MMA instruction shapes
  - 🔍 Consider different tile sizes

### If Sync Bound
- **Warp stalls on barrier > 25%**: Pipeline bubbles
  - 🔍 ACC staging might help (but we tried this!)
  - 🔍 Check if warp specialization is load-balanced

### If Low Occupancy (<50%)
- **SMEM or register pressure**: Reduce stages or tile sizes
  - 🔍 Try smaller tiles
  - 🔍 Reduce pipeline stages (but hurts latency hiding)

## Comparing Configurations

Profile multiple configurations to see which is best:

```bash
# Config 1: N=128
ncu --set full -o profile_n128 \
  python dense_blockscaled_gemm_persistent.py \
    --mnkl 8192,8192,1024,1 \
    --mma_tiler_mn 256,128 \
    --cluster_shape_mn 2,1 \
    --ab_dtype Float4E2M1FN \
    --sf_dtype Float8E8M0FNU \
    --sf_vec_size 16 \
    --c_dtype Float16 \
    --skip_ref_check \
    --iterations 1

# Config 2: N=256
ncu --set full -o profile_n256 \
  python dense_blockscaled_gemm_persistent.py \
    --mnkl 8192,8192,1024,1 \
    --mma_tiler_mn 256,256 \
    --cluster_shape_mn 2,1 \
    --ab_dtype Float4E2M1FN \
    --sf_dtype Float8E8M0FNU \
    --sf_vec_size 16 \
    --c_dtype Float16 \
    --skip_ref_check \
    --iterations 1
```

Then compare in Nsight Compute UI to see which has better metrics.

## CLI-Only Analysis (No UI)

If you can't use the GUI:

```bash
ncu \
  --csv \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.pct,\
smsp__average_warps_issue_stalled_membar_per_issue_active.pct,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.pct,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sectors_srcunit_tex_op_read.sum \
  python dense_blockscaled_gemm_persistent.py \
    --mnkl 8192,8192,1024,1 \
    --mma_tiler_mn 256,128 \
    --cluster_shape_mn 2,1 \
    --ab_dtype Float4E2M1FN \
    --sf_dtype Float8E8M0FNU \
    --sf_vec_size 16 \
    --c_dtype Float16 \
    --skip_ref_check \
    --iterations 1
```

## Expected Insights

Based on similar kernels, we expect to find:

1. **Memory bound** (60-70% warp stalls on memory)
   - TMA latency hiding is critical
   - Baseline's greedy AB allocation makes sense

2. **Moderate barrier stalls** (15-25%)
   - Pipeline transitions cause some bubbles
   - Warp specialization helps but isn't perfect

3. **Good compute utilization** (>70% Tensor Core usage)
   - MMA instructions are efficient
   - Not the bottleneck

Once we have this data, we can make **informed** optimization decisions!

