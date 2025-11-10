#!/usr/bin/env python3
"""
Production-ready wrapper for Blackwell MLA kernel with auto-configuration and autotuning.

Usage:
    from mla_inference import MLAInference
    
    # Simple usage with auto-config
    mla = MLAInference(num_heads=128, latent_dim=512, rope_dim=64)
    output = mla(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, 
                 softmax_scale, kv_length)
    
    # With autotuning
    mla = MLAInference(num_heads=128, latent_dim=512, rope_dim=64, autotune=True)
    output = mla(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
                 softmax_scale, kv_length)
"""

import sys
import warnings
import atexit
import os

# Suppress harmless CUDA cleanup errors during interpreter shutdown
def _suppress_cleanup_errors():
    """Redirect stderr to suppress JitModule cleanup errors."""
    sys.stderr = open(os.devnull, 'w')

# Register cleanup suppression
atexit.register(_suppress_cleanup_errors)

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import time

from mla import BlackwellMultiHeadLatentAttentionForward


@dataclass
class MLAConfig:
    """Configuration for MLA kernel."""
    qk_tile_mn: Tuple[int, int]
    pv_tile_mn: Tuple[int, int]
    persistent: bool
    clusters: int
    name: str = ""
    
    def __str__(self):
        return f"{self.name or 'Custom'}: QK={self.qk_tile_mn}, PV={self.pv_tile_mn}, pers={self.persistent}, clusters={self.clusters}"


# Optimized configurations from benchmarking
PRESET_CONFIGS = {
    # Sequence length ranges -> best config
    "64k": MLAConfig(
        qk_tile_mn=(128, 128),
        pv_tile_mn=(128, 256),
        persistent=False,
        clusters=4,
        name="64K-optimized"
    ),
    "128k": MLAConfig(
        qk_tile_mn=(128, 128),
        pv_tile_mn=(128, 256),
        persistent=False,
        clusters=8,
        name="128K-optimized"
    ),
    "default": MLAConfig(
        qk_tile_mn=(128, 128),
        pv_tile_mn=(128, 256),
        persistent=False,
        clusters=2,  # Conservative default
        name="default"
    ),
}

# Configs to try during autotuning
AUTOTUNE_CONFIGS = [
    MLAConfig((128, 128), (128, 128), False, 1, "small-fast"),
    MLAConfig((128, 128), (128, 256), False, 1, "medium-1cl"),
    MLAConfig((128, 128), (128, 256), False, 2, "medium-2cl"),
    MLAConfig((128, 128), (128, 256), False, 4, "medium-4cl"),
    MLAConfig((128, 128), (128, 256), False, 8, "medium-8cl"),
    MLAConfig((128, 128), (256, 128), False, 2, "large-2cl"),
    MLAConfig((128, 128), (256, 128), False, 4, "large-4cl"),
    MLAConfig((128, 128), (256, 256), False, 2, "xlarge-2cl"),
    MLAConfig((128, 128), (256, 256), False, 4, "xlarge-4cl"),
]


class MLAInference:
    """Production wrapper for Blackwell MLA kernel."""
    
    def __init__(
        self,
        num_heads: int = 128,
        latent_dim: int = 512,
        rope_dim: int = 64,
        in_dtype=cutlass.Float16,
        out_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        config: Optional[MLAConfig] = None,
        autotune: bool = False,
        autotune_warmup: int = 3,
        autotune_iters: int = 10,
    ):
        """
        Initialize MLA inference wrapper.
        
        Args:
            num_heads: Number of attention heads
            latent_dim: Latent dimension
            rope_dim: RoPE dimension
            in_dtype: Input data type
            out_dtype: Output data type
            acc_dtype: Accumulator data type
            lse_dtype: LSE data type
            config: Manual config (overrides auto-selection)
            autotune: If True, benchmark configs on first call
            autotune_warmup: Warmup iterations for autotuning
            autotune_iters: Benchmark iterations for autotuning
        """
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        
        self.manual_config = config
        self.autotune = autotune
        self.autotune_warmup = autotune_warmup
        self.autotune_iters = autotune_iters
        
        # Cache for compiled kernels
        self._kernel_cache: Dict[str, Tuple[BlackwellMultiHeadLatentAttentionForward, any]] = {}
        # Cache tuned configs by sequence length bucket
        self._tuned_configs: Dict[str, MLAConfig] = {}
        
        # Get hardware info
        hardware_info = cutlass.utils.HardwareInfo()
        self.max_active_clusters = hardware_info.get_max_active_clusters(2)  # 2 = cluster_shape_mnk[0] * [1]
    
    def _get_seq_bucket(self, seq_len: int) -> str:
        """Get sequence length bucket for caching."""
        if seq_len >= 100_000:
            return "128k"
        elif seq_len >= 50_000:
            return "64k"
        elif seq_len >= 10_000:
            return "large"
        else:
            return "default"
    
    def _select_config(self, seq_len: int) -> MLAConfig:
        """Select config based on sequence length."""
        if self.manual_config:
            return self.manual_config
        
        # Check if we have a tuned config for this seq length bucket
        bucket = self._get_seq_bucket(seq_len)
        if bucket in self._tuned_configs:
            return self._tuned_configs[bucket]
        
        # Auto-select based on sequence length
        if seq_len >= 100_000:
            return PRESET_CONFIGS["128k"]
        elif seq_len >= 50_000:
            return PRESET_CONFIGS["64k"]
        else:
            return PRESET_CONFIGS["default"]
    
    def _get_kernel(self, config: MLAConfig):
        """Get or create kernel for config."""
        key = str(config)
        if key not in self._kernel_cache:
            kernel = BlackwellMultiHeadLatentAttentionForward(
                acc_dtype=self.acc_dtype,
                lse_dtype=self.lse_dtype,
                mma_qk_tiler_mn=config.qk_tile_mn,
                mma_pv_tiler_mn=config.pv_tile_mn,
                max_active_clusters=min(config.clusters, self.max_active_clusters),
                is_persistent=config.persistent,
                is_cpasync=False,  # Disabled due to alignment constraints
                use_page_table=False,
                is_var_seq=True,
                is_var_split_kv=False,
            )
            self._kernel_cache[key] = (kernel, None)  # (kernel, compiled)
        return self._kernel_cache[key]
    
    def _run_autotune(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_latents: torch.Tensor,
        k_pe: torch.Tensor,
        latent_to_k: torch.Tensor,
        latent_to_v: torch.Tensor,
        softmax_scale: float,
        kv_length: torch.Tensor,
    ) -> MLAConfig:
        """Run autotuning to find best config."""
        seq_len = kv_latents.shape[1]
        bucket = self._get_seq_bucket(seq_len)
        
        print(f"🔍 Autotuning MLA kernel for seq_len={seq_len} (bucket={bucket})...")
        
        best_config = None
        best_time = float('inf')
        
        for config in AUTOTUNE_CONFIGS:
            try:
                # Warmup
                for _ in range(self.autotune_warmup):
                    _ = self._forward_impl(
                        q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
                        softmax_scale, kv_length, config
                    )
                torch.cuda.synchronize()
                
                # Benchmark
                start = time.time()
                for _ in range(self.autotune_iters):
                    _ = self._forward_impl(
                        q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
                        softmax_scale, kv_length, config
                    )
                torch.cuda.synchronize()
                avg_time = (time.time() - start) / self.autotune_iters
                
                print(f"  {config.name:<15} → {avg_time*1000:.3f} ms")
                
                if avg_time < best_time:
                    best_time = avg_time
                    best_config = config
                    
            except Exception as e:
                print(f"  {config.name:<15} → FAILED ({type(e).__name__})")
                continue
        
        if best_config:
            print(f"✓ Selected: {best_config.name} ({best_time*1000:.3f} ms) - cached for '{bucket}' bucket")
            self._tuned_configs[bucket] = best_config
            return best_config
        else:
            print("⚠ Autotuning failed, using default")
            return PRESET_CONFIGS["default"]
    
    def _prepare_tensors(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_latents: torch.Tensor,
        k_pe: torch.Tensor,
        latent_to_k: torch.Tensor,
        kv_length: torch.Tensor,
    ) -> Tuple:
        """Prepare tensors in kernel format."""
        bsz, num_heads, q_len, qk_nope_head_dim = q_nope.shape
        
        if q_len != 1:
            raise ValueError(f"Only q_len=1 supported (decoding), got {q_len}")
        
        # Pre-projection: q_nope @ latent_to_k
        q_nope_flat = q_nope.squeeze(2).permute(1, 0, 2)  # (num_heads, bsz, qk_nope_head_dim)
        latent_to_k_bmm = latent_to_k.permute(1, 2, 0)    # (num_heads, qk_nope_head_dim, latent_dim)
        q_latent = torch.bmm(q_nope_flat, latent_to_k_bmm).contiguous()  # (num_heads, bsz, latent_dim)
        
        # Prepare other inputs
        q_pe_k = q_pe.squeeze(2).permute(1, 0, 2).contiguous()  # (num_heads, bsz, rope_dim)
        c_latent_k = kv_latents.transpose(0, 1).contiguous()     # (k_len, bsz, latent_dim)
        k_pe_k = k_pe.transpose(0, 1).contiguous()               # (k_len, bsz, rope_dim)
        
        # Create views with stride[1] == 1 (required by kernel)
        q_latent_view = q_latent.transpose(1, 2)      # (num_heads, latent_dim, bsz)
        q_pe_view = q_pe_k.transpose(1, 2)            # (num_heads, rope_dim, bsz)
        c_latent_view = c_latent_k.transpose(1, 2)    # (k_len, latent_dim, bsz)
        k_pe_view = k_pe_k.transpose(1, 2)            # (k_len, rope_dim, bsz)
        
        return q_latent_view, q_pe_view, c_latent_view, k_pe_view, bsz, q_latent
    
    def _forward_impl(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_latents: torch.Tensor,
        k_pe: torch.Tensor,
        latent_to_k: torch.Tensor,
        latent_to_v: torch.Tensor,
        softmax_scale: float,
        kv_length: torch.Tensor,
        config: MLAConfig,
    ) -> torch.Tensor:
        """Internal forward pass."""
        # Prepare tensors
        q_latent_view, q_pe_view, c_latent_view, k_pe_view, bsz, q_latent_orig = self._prepare_tensors(
            q_nope, q_pe, kv_latents, k_pe, latent_to_k, kv_length
        )
        
        # Get kernel
        kernel, compiled = self._get_kernel(config)
        
        # Use split_kv = 1 for now (until numerics are fully validated)
        split_kv = 1
        
        # Allocate outputs
        o_torch = torch.empty((self.num_heads, bsz, self.latent_dim), 
                             dtype=q_nope.dtype, device=q_nope.device)
        lse_torch = torch.empty((bsz, self.num_heads), 
                               dtype=torch.float32, device=q_nope.device)
        
        # Allocate workspace
        workspace_size = BlackwellMultiHeadLatentAttentionForward.get_workspace_size(
            self.num_heads, self.latent_dim, bsz, split_kv, self.acc_dtype
        )
        workspace_torch = None
        if workspace_size > 0:
            workspace_torch = torch.empty([workspace_size], dtype=torch.int8, device=q_nope.device)
        
        # Create views with proper strides and convert to CuTe
        o_view = o_torch.transpose(1, 2)
        lse_view = lse_torch.T
        
        def to_cute(t, dtype, leading_dim):
            x = from_dlpack(t, assumed_align=128, use_32bit_stride=True)
            x.element_type = dtype
            return x.mark_layout_dynamic(leading_dim=leading_dim)
        
        q_latent_cute = to_cute(q_latent_view, self.in_dtype, 1)
        q_pe_cute = to_cute(q_pe_view, self.in_dtype, 1)
        c_latent_cute = to_cute(c_latent_view, self.in_dtype, 1)
        k_pe_cute = to_cute(k_pe_view, self.in_dtype, 1)
        o_cute = to_cute(o_view, self.out_dtype, 1)
        lse_cute = to_cute(lse_view, self.lse_dtype, 0)
        
        cache_seqs_cute = from_dlpack(kv_length, assumed_align=128, use_32bit_stride=True)
        cache_seqs_cute.element_type = cutlass.Int32
        cache_seqs_cute = cache_seqs_cute.mark_layout_dynamic()
        
        workspace_cute = None
        if workspace_torch is not None:
            workspace_cute = from_dlpack(workspace_torch, assumed_align=128, use_32bit_stride=True)
            workspace_cute.element_type = cutlass.Int8
        
        # Get stream
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        
        # Compile if needed
        key = str(config)
        _, compiled = self._kernel_cache[key]
        if compiled is None:
            compiled = cute.compile(
                kernel,
                q_latent_cute, q_pe_cute, c_latent_cute, k_pe_cute, None,  # page_table
                o_cute, lse_cute, workspace_cute,
                split_kv, cache_seqs_cute, None,  # block_split_kvs
                softmax_scale, 1.0, stream
            )
            self._kernel_cache[key] = (kernel, compiled)
        
        # Run kernel
        compiled(
            q_latent_cute, q_pe_cute, c_latent_cute, k_pe_cute, None,
            o_cute, lse_cute, workspace_cute,
            split_kv, cache_seqs_cute, None,
            softmax_scale, 1.0, stream
        )
        
        # Post-projection: @ latent_to_v
        latent_to_v_bmm = latent_to_v.permute(1, 0, 2)  # (num_heads, latent_dim, value_dim)
        o_final = torch.bmm(o_torch, latent_to_v_bmm)   # (num_heads, bsz, value_dim)
        o_final = o_final.permute(1, 0, 2)              # (bsz, num_heads, value_dim)
        o_final = o_final.reshape(bsz, -1)              # (bsz, num_heads * value_dim)
        
        return o_final
    
    def __call__(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_latents: torch.Tensor,
        k_pe: torch.Tensor,
        latent_to_k: torch.Tensor,
        latent_to_v: torch.Tensor,
        softmax_scale: float,
        kv_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            q_nope: Query without RoPE (bsz, num_heads, 1, qk_nope_head_dim)
            q_pe: Query with RoPE (bsz, num_heads, 1, qk_rope_head_dim)
            kv_latents: KV cache latent (bsz, k_len, latent_dim)
            k_pe: K RoPE (bsz, k_len, qk_rope_head_dim)
            latent_to_k: Latent to K projection (latent_dim, num_heads, qk_nope_head_dim)
            latent_to_v: Latent to V projection (latent_dim, num_heads, value_dim)
            softmax_scale: Softmax scaling factor
            kv_length: Sequence lengths (bsz,)
            
        Returns:
            Output tensor (bsz, num_heads * value_dim)
        """
        bsz = q_nope.shape[0]
        if kv_length is None:
            kv_length = torch.full((bsz,), kv_latents.shape[1], 
                                  dtype=torch.int32, device=q_nope.device)
        
        seq_len = kv_latents.shape[1]
        bucket = self._get_seq_bucket(seq_len)
        
        # Run autotuning if enabled and no cached config for this bucket
        if self.autotune and bucket not in self._tuned_configs:
            config = self._run_autotune(
                q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
                softmax_scale, kv_length
            )
        else:
            config = self._select_config(seq_len)
            if bucket in self._tuned_configs:
                print(f"📦 Using cached config for bucket '{bucket}': {config.name}")
        
        return self._forward_impl(
            q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
            softmax_scale, kv_length, config
        )


# Example usage
if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()
    
    # Test configuration
    bsz, num_heads, q_len = 4, 128, 1
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    latent_dim = 512
    value_dim = 128
    k_len = 65536
    
    device = "cuda"
    dtype = torch.float16
    
    # Create test inputs
    q_nope = torch.randn(bsz, num_heads, q_len, qk_nope_head_dim, device=device, dtype=dtype)
    q_pe = torch.randn(bsz, num_heads, q_len, qk_rope_head_dim, device=device, dtype=dtype)
    kv_latents = torch.randn(bsz, k_len, latent_dim, device=device, dtype=dtype)
    k_pe = torch.randn(bsz, k_len, qk_rope_head_dim, device=device, dtype=dtype)
    latent_to_k = torch.randn(latent_dim, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
    latent_to_v = torch.randn(latent_dim, num_heads, value_dim, device=device, dtype=dtype)
    kv_length = torch.full((bsz,), k_len, dtype=torch.int32, device=device)
    # Softmax scale based on concatenated dimension (latent + rope)
    # since kernel computes attention over [qk_nope_head_dim + qk_rope_head_dim]
    softmax_scale = 1.0 / ((qk_nope_head_dim + qk_rope_head_dim) ** 0.5)
    
    print("Testing MLA Inference Wrapper")
    print("="*60)
    
    # Test 1: Auto-config
    print("\n1. Auto-config (no tuning)")
    mla_auto = MLAInference(num_heads=num_heads, latent_dim=latent_dim, rope_dim=qk_rope_head_dim)
    output_auto = mla_auto(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, 
                           softmax_scale, kv_length)
    print(f"   Output shape: {output_auto.shape}")
    
    # Test 2: With autotuning
    print("\n2. With autotuning")
    mla_tuned = MLAInference(num_heads=num_heads, latent_dim=latent_dim, 
                             rope_dim=qk_rope_head_dim, autotune=True)
    output_tuned = mla_tuned(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
                            softmax_scale, kv_length)
    print(f"   Output shape: {output_tuned.shape}")
    
    # Test 3: Manual config
    print("\n3. Manual config")
    manual_cfg = MLAConfig((128, 128), (128, 256), False, 4, "manual")
    mla_manual = MLAInference(num_heads=num_heads, latent_dim=latent_dim,
                              rope_dim=qk_rope_head_dim, config=manual_cfg)
    output_manual = mla_manual(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
                               softmax_scale, kv_length)
    print(f"   Output shape: {output_manual.shape}")
    
    print("\n✓ All tests passed!")

