# Performance Optimization Roadmap for BlockScaled GEMM

## Current Status

The baseline kernel in `dense_blockscaled_gemm_persistent.py` is already well-tuned for:
- ✅ SMEM staging (greedy AB allocation is correct)
- ✅ ACC staging (conditional 1/2 based on N dimension is optimal for TMEM)
- ✅ Warp specialization (TMA/MMA/Epilogue separation)
- ✅ TMA multicast for cluster efficiency

**Naive optimizations (changing staging parameters) caused 2-7% regressions.**

---

## Recommended Optimization Path

### Phase 1: Profile & Understand (REQUIRED FIRST)

**Goal**: Get real data on bottlenecks before making changes.

**Action Items**:
1. ✅ Read `PROFILE_INSTRUCTIONS.md`
2. Run Nsight Compute with "full" metrics set
3. Identify actual bottlenecks from profiler data:
   - Memory bound? → Check TMA efficiency, L2 hit rates
   - Compute bound? → Check Tensor Core utilization
   - Sync bound? → Check pipeline bubble sizes
   - Occupancy limited? → Check resource constraints

**Expected Findings** (hypothesis to verify):
- 60-70% warp stalls on memory (TMA latency)
- 15-25% warp stalls on barriers (pipeline transitions)
- >70% Tensor Core utilization (MMA is efficient)
- Good occupancy (SMEM/register allocation is reasonable)

**Time Investment**: 1-2 hours for profiling + analysis

**Decision Point**: Only proceed to Phase 2 if profiling reveals a clear bottleneck that isn't already optimized.

---

### Phase 2: Low-Risk, High-Impact Optimizations

These are changes that can be made without major kernel restructuring.

#### Option A: Epilogue Subtile Tuning (5-10% expected gain)

**Current State**: Epilogue tile size is computed based on alignment and memory bandwidth, not pipeline efficiency.

**Hypothesis**: Larger epilogue tiles with more subtiles would improve overlap between TMEM→RMEM copy and type conversion.

**Implementation**:
```python
# In _compute_epi_tile_size method
def _compute_epi_tile_size(self, mma_tiler_mn, c_dtype):
    # Current: Simple alignment-based sizing
    epi_m = min(mma_tiler_mn[0], 128)
    epi_n = min(mma_tiler_mn[1], 128)
    
    # Proposed: Compute based on target subtile count
    # Want 4-6 subtiles for good overlap
    target_subtiles = 5
    total_elems = mma_tiler_mn[0] * mma_tiler_mn[1]
    target_tile_size = total_elems / target_subtiles
    
    # Round to valid sizes (powers of 2, aligned)
    epi_m = compute_optimal_m(target_tile_size, mma_tiler_mn[0])
    epi_n = compute_optimal_n(target_tile_size, mma_tiler_mn[1])
    
    return epi_m, epi_n
```

**Risk**: Low (only changes epilogue tile size, not algorithm)

**Validation**: Profile before/after to confirm subtile count changes and overlap improves

**Priority**: ⭐⭐⭐ (Good first attempt after profiling)

---

#### Option B: Dynamic Warp Assignment (5-12% expected gain)

**Current State**: Fixed 4 epilogue warps (warps 0-3), 1 MMA warp (warp 4), 1 TMA warp (warp 5)

**Hypothesis**: For different tile sizes, the epilogue/MMA workload ratio varies. Dynamic assignment could balance better.

**Analysis**:
```python
# Compute work ratio
mma_work = mma_tiler_mnk[0] * mma_tiler_mnk[1] * mma_tiler_mnk[2]  # M*N*K ops
epi_work = mma_tiler_mnk[0] * mma_tiler_mnk[1] * epi_complexity  # M*N*ops

# For N=128: epi_work/mma_work ≈ 0.4 → 4 epi warps reasonable
# For N=256: epi_work/mma_work ≈ 0.6 → 5 epi warps better?
```

**Implementation**:
```python
def _compute_warp_assignment(self, mma_tiler_mnk, epi_tile):
    # Compute workload ratio
    mma_cycles = estimate_mma_cycles(mma_tiler_mnk)
    epi_cycles = estimate_epi_cycles(mma_tiler_mnk, epi_tile)
    
    # Assign warps proportionally (min 3 epi, max 5 epi)
    total_warps = 6
    epi_warps = max(3, min(5, int(total_warps * epi_cycles / (mma_cycles + epi_cycles))))
    
    # Assignment
    epilog_warp_id = list(range(0, epi_warps))
    mma_warp_id = epi_warps
    tma_warp_id = epi_warps + 1
    
    return epilog_warp_id, mma_warp_id, tma_warp_id
```

**Risk**: Medium (changes warp assignment, requires careful testing)

**Validation**: 
- Ensure all configurations still work
- Profile warp utilization before/after
- Check for load imbalance

**Priority**: ⭐⭐ (Try after Option A if profiling shows warp imbalance)

---

#### Option C: TMA Prefetching Improvements (3-8% expected gain)

**Current State**: TMA descriptors are prefetched once at kernel start.

**Hypothesis**: Prefetching TMA descriptors for the next tile during current tile computation could hide more latency.

**Implementation**:
```python
# In TMA warp loop
for k_block_idx in cutlass.range_constexpr(num_k_blocks):
    # Current: Load current tile
    cute.copy(tma_atom_a, ...)
    
    # Proposed: Prefetch next tile descriptor
    if k_block_idx + 1 < num_k_blocks:
        cpasync.prefetch_descriptor(tma_atom_a, next_tile_coords)
    
    cute.copy(tma_atom_a, ...)
```

**Risk**: Low (additive change, doesn't break existing logic)

**Validation**: Check if TMA load latency improves in profiler

**Priority**: ⭐ (Small gain, try if profiling shows TMA descriptor fetch latency)

---

### Phase 3: Moderate-Risk Optimizations

These require more significant changes but stay within the current kernel structure.

#### Option D: Epilogue Fusion (15-25% expected gain for fused ops)

**Current State**: Epilogue only does type conversion (acc_dtype → c_dtype).

**Opportunity**: Many real workloads need additional operations:
- Bias addition
- Activation functions (GELU, ReLU)
- Output quantization
- Residual connections

**Implementation**:
Add optional fusion operations to epilogue:

```python
# Extend epilogue_op to support multiple operations
def epilogue_with_bias_gelu(acc, bias, scale):
    # Type conversion
    output = convert(acc, target_dtype)
    
    # Bias
    output = output + bias
    
    # GELU
    output = 0.5 * output * (1 + tanh(...))
    
    # Scale
    output = output * scale
    
    return output

# Usage
gemm_kernel(..., epilogue_op=epilogue_with_bias_gelu)
```

**Benefits**:
- Saves global memory round-trips (don't need to load/store intermediate results)
- Better data locality (process data while in registers/TMEM)
- Kernel fusion reduces launch overhead

**Risk**: Medium-High (significant code changes, need to support multiple fusion patterns)

**Validation**: 
- Correctness checks with reference implementations
- Benchmark end-to-end latency (not just GEMM time)

**Priority**: ⭐⭐⭐ (High impact if you need fused operations)

**Note**: This is a **feature** more than a pure optimization. Baseline is already optimal for vanilla GEMM.

---

#### Option E: Cluster Size Tuning (5-15% expected gain for specific configs)

**Current State**: Cluster shape is a fixed input parameter.

**Hypothesis**: Different problem sizes have optimal cluster shapes. Could auto-select based on M/N dimensions.

**Analysis**:
```python
# For large M, large N: Cluster (2, 2) or (2, 1) good
# For small M, large N: Cluster (1, 2) better (less M fragmentation)
# For large M, small N: Cluster (2, 1) better
```

**Implementation**:
```python
def _select_optimal_cluster(self, m, n, cluster_constraint):
    if m < 4096 and n >= 8192:
        return (1, 2)  # Favor N dimension
    elif m >= 8192 and n < 4096:
        return (2, 1)  # Favor M dimension
    else:
        return (2, 2)  # Balanced
```

**Risk**: Low (just changes launch parameters)

**Validation**: Benchmark across multiple problem sizes with different clusters

**Priority**: ⭐ (Nice-to-have for auto-tuning, but users can tune manually)

---

### Phase 4: High-Risk, High-Reward Optimizations

These require major kernel restructuring or new algorithms.

#### Option F: Split-K (15-30% expected gain for K > 4096)

**Current State**: Single CTA processes entire K dimension for each output tile.

**Opportunity**: For very large K dimensions, split K across multiple CTAs and reduce.

**Algorithm**:
```
1. Divide K into slices (e.g., K=8192 → 4 slices of 2048)
2. Each CTA computes partial GEMM for its K slice
3. Store partial results to global memory
4. Reduction kernel sums partials → final output
```

**Benefits**:
- Better SM utilization (more CTAs can run in parallel)
- Reduced per-CTA resource usage (smaller accumulators)
- Better cache locality (each CTA works on smaller K range)

**Drawbacks**:
- Extra global memory writes/reads for partials
- Extra reduction kernel launch
- Only helps for large K

**Implementation Complexity**: High (requires multi-kernel launch, reduction logic)

**Risk**: High (major algorithmic change)

**Validation**:
- Correctness checks with reduction
- Benchmark K=512, 1024, 2048, 4096, 8192, 16384
- Profitability threshold analysis

**Priority**: ⭐⭐ (High impact for large K, but high complexity)

---

#### Option G: Asynchronous Pipeline Rebalancing (10-20% expected gain)

**Current State**: 3 separate pipelines (AB, ACC, C) with fixed synchronization points.

**Opportunity**: Make pipeline transitions more flexible to reduce bubbles.

**Concept**:
- Allow MMA to start on next tile before epilogue finishes current tile
- Use more aggressive double-buffering in TMEM
- Overlap TMA loads for tile N+2 with MMA for tile N+1 and epilogue for tile N

**Implementation Complexity**: Very High (requires redesigning pipeline logic)

**Risk**: Very High (easy to introduce deadlocks or correctness issues)

**Priority**: ⭐ (Only attempt if profiling shows large pipeline bubbles AND you have time for deep debugging)

---

#### Option H: Custom Swizzle Patterns (5-12% expected gain)

**Current State**: Uses standard swizzle patterns for SMEM layouts.

**Opportunity**: Custom swizzle patterns tailored to specific access patterns could reduce bank conflicts.

**Analysis**:
- Profile SMEM bank conflicts with Nsight Compute
- If conflicts > 10%, try alternate swizzle patterns
- Test: `128B`, `64B`, `32B` swizzles for AB and C tensors

**Implementation**:
```python
# Try different swizzle patterns
for swizzle_bits in [7, 6, 5]:  # 128B, 64B, 32B
    a_smem_layout.inner = cute.make_swizzle(swizzle_bits, 0, 3)
    # Benchmark and compare
```

**Risk**: Medium (wrong swizzle can hurt performance badly)

**Validation**: Profile SMEM bank conflicts before/after

**Priority**: ⭐ (Only if profiling shows bank conflict issues)

---

## Decision Tree

```
Start
  │
  ├─► Profile with Nsight Compute
  │   │
  │   ├─► Memory bound (TMA latency)?
  │   │   ├─► Try Option A (Epilogue Tuning)
  │   │   └─► Try Option C (TMA Prefetch)
  │   │
  │   ├─► Sync bound (Pipeline bubbles)?
  │   │   ├─► Try Option B (Warp Assignment)
  │   │   └─► Try Option G (Pipeline Rebalance) - High Risk
  │   │
  │   ├─► SMEM bound (Bank conflicts)?
  │   │   └─► Try Option H (Custom Swizzle)
  │   │
  │   └─► Already optimal?
  │       ├─► Try Option D (Epilogue Fusion) - if you need fused ops
  │       ├─► Try Option F (Split-K) - if K > 4096
  │       └─► Try Option E (Cluster Tuning) - for auto-tuning
  │
  └─► Measure & Iterate
      ├─► Benchmark each change
      ├─► Profile to verify improvement
      └─► Document findings in OPTIMIZATIONS.md
```

---

## Benchmarking Protocol

For each optimization attempt:

### 1. Baseline Measurement
```bash
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
```

### 2. Implement Change

Make the optimization in `faster_dense_blockscaled_gemm_persistent.py`.

### 3. Correctness Check
```bash
python faster_dense_blockscaled_gemm_persistent.py \
  --mnkl 2048,2048,512,1 \
  --mma_tiler_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --ab_dtype Float4E2M1FN \
  --sf_dtype Float8E8M0FNU \
  --sf_vec_size 16 \
  --c_dtype Float16
  # (no --skip_ref_check)
```

### 4. Performance Comparison
```bash
./RUN_BENCHMARK.sh  # Runs 3 configurations
```

### 5. Profile the Change
```bash
ncu --set full -o optimized_profile \
  python faster_dense_blockscaled_gemm_persistent.py \
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

### 6. Decision

- **Speedup > 5%**: Keep the change, document in OPTIMIZATIONS.md
- **Speedup 0-5%**: Profile to understand if it helps specific cases, consider keeping
- **Regression**: Revert and document why it failed in OPTIMIZATIONS.md

---

## Expected Outcomes

### Realistic Goals (with profiling data)

| Optimization | Expected Gain | Confidence | Complexity |
|--------------|---------------|------------|------------|
| Epilogue Subtile Tuning | 5-10% | Medium | Low |
| Warp Assignment Tuning | 5-12% | Medium | Medium |
| TMA Prefetch | 3-8% | Medium | Low |
| Epilogue Fusion | 15-25%* | High | Medium-High |
| Split-K (K>4K) | 15-30% | High | High |
| Cluster Tuning | 5-15% | Medium | Low |

*For workloads that need fused operations; minimal gain for vanilla GEMM

### Combined Impact

If successful:
- **Conservative**: 8-15% improvement (Options A + C)
- **Moderate**: 15-25% improvement (Options A + B + D)
- **Aggressive**: 30-40% improvement (Options A + B + D + F, for specific workloads)

**Note**: These gains are relative to an already well-tuned baseline. Getting even 10% improvement is excellent for a highly optimized kernel.

---

## Questions?

- For profiling help: See `PROFILE_INSTRUCTIONS.md`
- For baseline explanation: See `dense_blockscaled_gemm_persistent.py`
- For optimization attempts: See `OPTIMIZATIONS.md`
- For benchmarking: See `QUICKSTART.md`


