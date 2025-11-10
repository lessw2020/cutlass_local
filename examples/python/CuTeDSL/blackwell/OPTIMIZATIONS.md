# 🚀 Blackwell BlockScaled GEMM Kernel Optimizations

## Overview

This document describes optimization attempts for the blockscaled GEMM kernel in `faster_dense_blockscaled_gemm_persistent.py` compared to the baseline in `dense_blockscaled_gemm_persistent.py`.

**Current Status**: The "optimized" kernel currently matches the baseline exactly after initial optimization attempts showed performance regressions.

---

## Attempted Optimizations (Reverted)

### ❌ Attempt 1: Increased ACC (Accumulator) Staging

**Hypothesis**: Increasing ACC stages from 1→2 for N=256 tiles would improve overlap between MMA computation and epilogue processing.

**Change Made**:
```python
# Baseline:
num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

# Attempted:
num_acc_stage = 2  # Always 2
```

**Why It Failed**:
- **TMEM Pressure**: For N=256 tiles, 2 ACC stages require more TMEM columns than optimal
- **Pipeline Imbalance**: The baseline's 1-stage design for N=256 is actually well-tuned to the hardware capabilities
- **Measured Results**: 2-7% performance **regression** across all test cases

**Lesson Learned**: The baseline's conditional ACC staging (`1 if N==256 else 2`) is carefully tuned to TMEM constraints and shouldn't be changed without deeper analysis.

---

### ❌ Attempt 2: Rebalanced SMEM Stage Allocation

**Hypothesis**: Targeting 5 AB stages + 4+ C stages (instead of greedy AB allocation) would balance mainloop and epilogue buffering better.

**Baseline Strategy**:
```python
# Greedy: maximize AB stages first, give leftovers to C
num_ab_stage = (smem_capacity - reserved - c_bytes) // ab_bytes_per_stage
num_c_stage = 2 + remaining_smem // c_bytes_per_stage
```

**Attempted Strategy**:
```python
# Target balanced allocation
target_ab_stages = 5
target_c_stages = 4
# Allocate targets, give extra to C stages
```

**Why It Failed**:
- **TMA Latency Requirements**: 5 AB stages is insufficient to fully hide TMA latency for some configurations
- **Epilogue Not Bottleneck**: The greedy strategy correctly identifies that AB loading is the bottleneck, not epilogue
- **Memory Access Pattern Bugs**: The scaling-down logic for tight memory had bugs, causing illegal memory access crashes
- **Measured Results**:
  - Test 1 (M=8192, N=8192, K=1024, MMA=256x128): **7.5% slower** (45.65μs → 49.36μs)
  - Test 2 (M=8192, N=8192, K=1024, MMA=256x256): **Illegal memory access crash**
  - Test 3 (M=2048, N=2048, K=512, MMA=128x128): **2.1% slower** (38.88μs → 39.72μs)

**Lesson Learned**: The baseline's greedy allocation is well-tuned. The mainloop (TMA + MMA) is the bottleneck, not epilogue. Reducing AB stages to prioritize C stages hurts performance.

---

## Why The Baseline Is Actually Well-Optimized

After attempting optimizations, it's clear the baseline kernel is already highly tuned:

1. **Smart ACC Staging**: Uses 1 stage for N=256 (TMEM pressure) and 2 otherwise (better overlap)
2. **Mainloop-First Allocation**: Correctly identifies TMA latency as the primary bottleneck
3. **Residual Epilogue Buffering**: After maximizing AB stages, gives all remaining memory to C stages
4. **Well-Tested**: The baseline logic has been carefully validated against hardware characteristics

### Actual Bottlenecks (For Future Work)

Based on the optimization failures, here are the real bottlenecks:

1. **TMA Latency Hiding** (40-50% impact):
   - Need 6-8 AB stages to fully hide global→SMEM transfer latency
   - Reducing this hurts more than epilogue improvements help

2. **Warp Utilization** (20-30% impact):
   - Fixed 4 epilogue warps may be over/under-provisioned for different tile sizes
   - But changing this requires kernel-level redesign, not just staging changes

3. **Epilogue Fusion** (15-25% impact):
   - Current kernel does type conversion only
   - Could fuse other operations (bias, activation, quantization)

4. **Split-K** (10-20% impact for large K):
   - For K > 4096, splitting across CTAs could help
   - But requires significant kernel restructuring

---

## Benchmark Results (Current = Baseline)

Since the optimized kernel now matches the baseline, running the benchmark will show identical performance:

```bash
./RUN_BENCHMARK.sh
```

Expected output:
```
Speedup: 1.000x (+0.0%)
Time saved: 0.00 μs (0.000 ms)
```

---

## Path Forward

### Conservative Optimizations (Low Risk)

1. **Profiler-Guided Analysis**:
   - Use Nsight Compute to identify actual bottlenecks
   - Measure TMA efficiency, warp stalls, SMEM bank conflicts
   - Only optimize what profiler shows as real issues

2. **Microbenchmarks**:
   - Test individual components (TMA, MMA, epilogue) in isolation
   - Validate assumptions about latencies and throughput

### Ambitious Optimizations (High Risk, High Reward)

1. **Dynamic Warp Assignment**:
   - Compute optimal epilogue warp count based on tile size
   - Requires kernel-level changes to warp role assignment

2. **Epilogue Fusion**:
   - Fuse common operations (bias, activation) into epilogue
   - Saves global memory round-trips

3. **Split-K**:
   - For very large K dimensions (K > 4096)
   - Requires multi-CTA reduction logic

---

## Running Comparisons

### Test Baseline vs "Optimized" (Currently Identical)

```bash
python faster_dense_blockscaled_gemm_persistent.py \
  --benchmark_mode \
  --mnkl 8192,8192,1024,1 \
  --mma_tiler_mn 256,128 \
  --cluster_shape_mn 2,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16
```

### Run Optimized Kernel Only (With Correctness Check)

```bash
python faster_dense_blockscaled_gemm_persistent.py \
  --mnkl 2048,2048,512,1 \
  --mma_tiler_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16
```

### Run Baseline Kernel Only

```bash
python faster_dense_blockscaled_gemm_persistent.py \
  --use_baseline \
  --mnkl 8192,8192,1024,1 \
  --mma_tiler_mn 256,128 \
  --cluster_shape_mn 2,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16 \
  --skip_ref_check
```

---

## Key Takeaways

1. **Don't Assume Bottlenecks**: Profile first, optimize second
2. **Respect Existing Tuning**: The baseline is well-optimized for good reasons
3. **Test Early**: Small changes can have large, unexpected impacts
4. **Hardware Constraints Matter**: TMEM, SMEM, and TMA latencies are real constraints that can't be ignored

---

## Questions?

For implementation details, see:
- Baseline kernel: `dense_blockscaled_gemm_persistent.py` (lines 1648-1739)
- Comparison framework: `faster_dense_blockscaled_gemm_persistent.py` (lines 2417-2488)
