#!/usr/bin/env python3
"""
Example usage of the PyTorch wrapper for CUTLASS persistent dense GEMM.

This shows how to use cutlass_matmul() as a drop-in replacement for torch.matmul().
"""

import torch
import sys
sys.path.insert(0, '.')

from dense_gemm_persistent import cutlass_matmul

def main():
    print("=" * 80)
    print("CUTLASS PyTorch Wrapper Example")
    print("=" * 80)
    
    # Example 1: Simple 2D matmul
    print("\n1. Simple 2D Matrix Multiplication")
    print("-" * 80)
    M, N, K = 4096, 4096, 2048
    
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(N, K, dtype=torch.float16, device='cuda')  # Note: B is [N, K] not [K, N]
    
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    
    # Using CUTLASS (computes A @ B^T)
    result_cutlass = cutlass_matmul(A, B)
    print(f"CUTLASS result shape: {result_cutlass.shape}")
    
    # Compare with PyTorch (need to transpose B)
    result_pytorch = torch.matmul(A, B.T)
    print(f"PyTorch result shape: {result_pytorch.shape}")
    
    # Check correctness
    max_diff = (result_cutlass - result_pytorch).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")
    
    # Example 2: Batched matmul
    print("\n2. Batched Matrix Multiplication")
    print("-" * 80)
    L, M, N, K = 8, 2048, 2048, 1024
    
    A_batched = torch.randn(L, M, K, dtype=torch.float16, device='cuda')
    B_batched = torch.randn(L, N, K, dtype=torch.float16, device='cuda')
    
    print(f"A shape: {A_batched.shape}")
    print(f"B shape: {B_batched.shape}")
    
    result_cutlass = cutlass_matmul(A_batched, B_batched)
    print(f"CUTLASS result shape: {result_cutlass.shape}")
    
    # Compare with PyTorch
    result_pytorch = torch.bmm(A_batched, B_batched.transpose(1, 2))
    max_diff = (result_cutlass - result_pytorch).abs().max().item()
    print(f"Max difference: {max_diff:.6f}")
    
    # Example 3: Best performing configuration from sweep
    print("\n3. Best Performance Configuration")
    print("-" * 80)
    M, N, K = 8192, 8192, 8192
    
    A = torch.randn(M, K, dtype=torch.float16, device='cuda')
    B = torch.randn(N, K, dtype=torch.float16, device='cuda')
    
    # Use best config from sweep: MMA=(256,128), Cluster=(2,1)
    result = cutlass_matmul(
        A, B,
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        use_tma_store=True
    )
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    
    # Benchmark comparison - use baseline's run() function for accurate benchmarking
    print("\n4. Performance Comparison (Best Config: MMA=256x128, Cluster=2x1)")
    print("-" * 80)
    
    from dense_gemm_persistent import run
    import cutlass
    
    # Benchmark CUTLASS using baseline's built-in benchmarking
    cutlass_time_us = run(
        mnkl=(M, N, K, 1),
        ab_dtype=cutlass.Float16,
        c_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k", 
        c_major="n",
        mma_tiler_mn=(256, 128),
        cluster_shape_mn=(2, 1),
        use_2cta_instrs=True,
        use_tma_store=True,
        tolerance=1e-01,
        warmup_iterations=5,
        iterations=50,
        skip_ref_check=True,
        use_cold_l2=False,
    )
    cutlass_time = cutlass_time_us / 1000  # Convert to ms
    
    # Benchmark PyTorch
    import time
    # Warmup
    for _ in range(5):
        _ = torch.matmul(A, B.T)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(50):
        result = torch.matmul(A, B.T)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / 50 * 1000  # ms
    
    flops = 2 * M * N * K
    cutlass_tflops = flops / (cutlass_time / 1000) / 1e12
    pytorch_tflops = flops / (pytorch_time / 1000) / 1e12
    
    print(f"Problem size: M={M}, N={N}, K={K}")
    print(f"CUTLASS:  {cutlass_time:.3f} ms ({cutlass_tflops:.2f} TFLOPS)")
    print(f"PyTorch:  {pytorch_time:.3f} ms ({pytorch_tflops:.2f} TFLOPS)")
    print(f"Speedup:  {pytorch_time/cutlass_time:.2f}x")
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

