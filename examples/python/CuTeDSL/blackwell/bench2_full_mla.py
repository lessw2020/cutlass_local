import torch
import time
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from typing import Optional

# Import the kernel class
from mla import BlackwellMultiHeadLatentAttentionForward


class CuteDSLMLAWrapper:
    """Drop-in replacement for slow_mla_attn using CuteDSL kernel"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        rope_dim: int = 64,
        num_heads: int = 128,
        in_dtype=cutlass.Float16,
        out_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        use_page_table: bool = False,
        page_size: int = 128,
        is_persistent: bool = False,
        is_cpasync: bool = False,
    ):
        self.latent_dim = latent_dim
        self.rope_dim = rope_dim
        self.num_heads = num_heads
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.use_page_table = use_page_table
        self.page_size = page_size
        self.is_persistent = is_persistent
        self.is_cpasync = is_cpasync
        
        # Get hardware info
        hardware_info = cutlass.utils.HardwareInfo()
        cluster_shape_mnk = (2, 1, 1)
        self.max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        )
        
        # Create kernel instance
        self.kernel = BlackwellMultiHeadLatentAttentionForward(
            acc_dtype=acc_dtype,
            lse_dtype=lse_dtype,
            mma_qk_tiler_mn=mma_qk_tiler_mn,
            mma_pv_tiler_mn=mma_pv_tiler_mn,
            max_active_clusters=self.max_active_clusters,
            is_persistent=is_persistent,
            is_cpasync=is_cpasync,
            use_page_table=use_page_table,
            is_var_seq=True,
            is_var_split_kv=False,
        )
        
        self.compiled_kernel = None
        self.last_config = None
    
    def _convert_to_cute(self, tensor: torch.Tensor, dtype) -> cute.Tensor:
        """Convert PyTorch tensor to CuTe tensor"""
        cute_tensor = from_dlpack(tensor, assumed_align=16, use_32bit_stride=True)
        cute_tensor.element_type = dtype
        return cute_tensor
    
    def _get_split_kv(self, batch_size: int, max_seq_len: int) -> int:
        """Calculate optimal split_kv value"""
        return BlackwellMultiHeadLatentAttentionForward.get_split_kv(
            batch_size,
            max_seq_len,
            self.mma_qk_tiler_mn,
            self.max_active_clusters * 2,
        )
    
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
        Drop-in replacement for slow_mla_attn.
        
        Args:
            q_nope: Query without RoPE (bsz, num_heads, q_len, qk_nope_head_dim)
            q_pe: Query with RoPE (bsz, num_heads, q_len, qk_rope_head_dim)
            kv_latents: KV cache latent (bsz, k_len, latent_dim)
            k_pe: K RoPE (bsz, k_len, qk_rope_head_dim)
            latent_to_k: Latent to K projection (latent_dim, num_heads, qk_nope_head_dim)
            latent_to_v: Latent to V projection (latent_dim, num_heads, value_dim)
            softmax_scale: Softmax scaling factor
            kv_length: Sequence lengths (bsz,)
            
        Returns:
            Output tensor (bsz, num_heads * value_dim)
        """
        bsz, num_heads, q_len, qk_nope_head_dim = q_nope.shape
        
        if q_len != 1:
            raise ValueError(
                f"CuteDSL kernel only supports q_len=1 (decoding), got q_len={q_len}"
            )
        
        # Step 1: Pre-projection - q_nope @ latent_to_k
        from einops import rearrange
        q_nope_flat = rearrange(q_nope, "b h q d -> h (b q) d")
        q_latent = torch.bmm(q_nope_flat, rearrange(latent_to_k, "l h d -> h d l"))
        q_latent = rearrange(q_latent, "h (b q) l -> h l b", b=bsz)
        
        # Prepare q_pe: (bsz, num_heads, 1, rope_dim) -> (num_heads, rope_dim, bsz)
        q_pe_cute = q_pe.squeeze(2).permute(1, 2, 0).contiguous()
        
        # Prepare KV cache: (bsz, k_len, dim) -> (k_len, dim, bsz)
        c_latent_cute = kv_latents.permute(1, 2, 0).contiguous()
        k_pe_cute = k_pe.permute(1, 2, 0).contiguous()
        
        # Prepare cache_seqs
        if kv_length is None:
            kv_length = torch.full((bsz,), kv_latents.shape[1], dtype=torch.int32, device=q_nope.device)
        cache_seqs = kv_length.int()
        
        # Calculate split_kv
        max_seq_len = torch.max(cache_seqs).item()
        split_kv = self._get_split_kv(bsz, max_seq_len)
        
        # Allocate output tensors
        o_torch = torch.empty((num_heads, self.latent_dim, bsz), dtype=torch.float16, device=q_nope.device)
        lse_torch = torch.empty((num_heads, bsz), dtype=torch.float32, device=q_nope.device)
        
        # Allocate workspace
        workspace_size = BlackwellMultiHeadLatentAttentionForward.get_workspace_size(
            num_heads, self.latent_dim, bsz, split_kv, self.acc_dtype
        )
        workspace_torch = None
        if workspace_size > 0:
            workspace_torch = torch.empty([workspace_size], dtype=torch.int8, device=q_nope.device)
        
        # Convert to CuTe tensors
        q_latent_cute = self._convert_to_cute(q_latent.contiguous(), self.in_dtype)
        q_latent_cute = q_latent_cute.mark_layout_dynamic(leading_dim=1)
        
        q_pe_cute_t = self._convert_to_cute(q_pe_cute, self.in_dtype)
        q_pe_cute_t = q_pe_cute_t.mark_layout_dynamic(leading_dim=1)
        
        c_latent_cute_t = self._convert_to_cute(c_latent_cute, self.in_dtype)
        c_latent_cute_t = c_latent_cute_t.mark_layout_dynamic(leading_dim=1)
        
        k_pe_cute_t = self._convert_to_cute(k_pe_cute, self.in_dtype)
        k_pe_cute_t = k_pe_cute_t.mark_layout_dynamic(leading_dim=1)
        
        o_cute = self._convert_to_cute(o_torch, self.out_dtype)
        o_cute = o_cute.mark_layout_dynamic(leading_dim=1)
        
        lse_cute = self._convert_to_cute(lse_torch, self.lse_dtype)
        lse_cute = lse_cute.mark_layout_dynamic(leading_dim=0)
        
        cache_seqs_cute = self._convert_to_cute(cache_seqs, cutlass.Int32)
        cache_seqs_cute = cache_seqs_cute.mark_layout_dynamic()
        
        workspace_cute = None
        if workspace_torch is not None:
            workspace_cute = self._convert_to_cute(workspace_torch, cutlass.Int8)
        
        # Get CUDA stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)
        
        # Compile kernel if needed
        config = (bsz, max_seq_len, split_kv)
        if self.compiled_kernel is None or self.last_config != config:
            self.compiled_kernel = cute.compile(
                self.kernel,
                q_latent_cute,
                q_pe_cute_t,
                c_latent_cute_t,
                k_pe_cute_t,
                None,  # page_table
                o_cute,
                lse_cute,
                workspace_cute,
                split_kv,
                cache_seqs_cute,
                None,  # block_split_kvs
                softmax_scale,
                1.0,   # output_scale
                stream,
            )
            self.last_config = config
        
        # Run kernel
        self.compiled_kernel(
            q_latent_cute,
            q_pe_cute_t,
            c_latent_cute_t,
            k_pe_cute_t,
            None,  # page_table
            o_cute,
            lse_cute,
            workspace_cute,
            split_kv,
            cache_seqs_cute,
            None,  # block_split_kvs
            softmax_scale,
            1.0,   # output_scale
            stream,
        )
        
        # Wait for completion
        torch.cuda.synchronize()
        
        # Step 2: Post-projection - @ latent_to_v
        # o_torch: (num_heads, latent_dim, bsz)
        o_for_proj = rearrange(o_torch, "h l b -> h (b 1) l")
        o_final = torch.bmm(o_for_proj, rearrange(latent_to_v, "l h d -> h l d"))
        o_final = rearrange(o_final, "h (b q) d -> b q (h d)", b=bsz)
        
        return o_final.squeeze(1)  # (bsz, num_heads * value_dim)


def slow_mla_attn(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_latents: torch.Tensor,
    k_pe: torch.Tensor,
    latent_to_k: torch.Tensor,
    latent_to_v: torch.Tensor,
    softmax_scale: float,
    kv_length: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation from your code"""
    import torch.nn.functional as F
    from einops import rearrange
    
    bsz, num_heads, q_len, qk_nope_head_dim = q_nope.shape
    _, _, _, qk_rope_head_dim = q_pe.shape
    _, k_len, latent_dim = kv_latents.shape
    _, _, value_dim = latent_to_v.shape

    q_nope = rearrange(q_nope, "b h q d -> h (b q) d")
    fused_q_nope = torch.bmm(q_nope, rearrange(latent_to_k, "l h d -> h d l"))
    fused_q_nope = rearrange(fused_q_nope, "h (b q) l -> b q h l", b=bsz)
    nope_logits = torch.einsum("b q h l, b k l -> b h q k", fused_q_nope, kv_latents)
    pe_logits = torch.einsum("b h q e, b k e -> b h q k", q_pe, k_pe)
    attn_logits = (nope_logits + pe_logits) * softmax_scale
    
    q_ind = torch.arange(q_len, device=q_nope.device)
    k_ind = torch.arange(k_len, device=q_nope.device)
    mask = q_ind[None, :, None] + kv_length[:, None, None] >= k_ind[None, None, :]
    mask = mask.unsqueeze(1)
    attn_logits = torch.where(mask, attn_logits, float("-inf"))
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_output = torch.einsum("b k l, b h q k -> b h q l", kv_latents, attn_weights)
    attn_output = rearrange(attn_output, "b h q l -> h (b q) l")
    attn_output = torch.bmm(attn_output, rearrange(latent_to_v, "l h d -> h l d"))
    attn_output = rearrange(attn_output, "h (b q) d -> b q (h d)", b=bsz)

    return attn_output


def test_decode_step():
    """Test single decode step and compare with slow_mla_attn"""
    
    print("=" * 80)
    print("Testing CuteDSL MLA Wrapper vs slow_mla_attn")
    print("=" * 80)
    
    # Configuration
    bsz = 8
    num_heads = 128
    q_len = 1  # Decode: single token
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    latent_dim = 512
    value_dim = 128
    k_len = 2048
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {bsz}")
    print(f"  Num heads: {num_heads}")
    print(f"  KV cache length: {k_len}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Value dim: {value_dim}")
    print(f"  Data type: {dtype}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create inputs
    q_nope = torch.randn(bsz, num_heads, q_len, qk_nope_head_dim, device=device, dtype=dtype)
    q_pe = torch.randn(bsz, num_heads, q_len, qk_rope_head_dim, device=device, dtype=dtype)
    kv_latents = torch.randn(bsz, k_len, latent_dim, device=device, dtype=dtype)
    k_pe = torch.randn(bsz, k_len, qk_rope_head_dim, device=device, dtype=dtype)
    latent_to_k = torch.randn(latent_dim, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
    latent_to_v = torch.randn(latent_dim, num_heads, value_dim, device=device, dtype=dtype)
    kv_length = torch.full((bsz,), k_len, dtype=torch.int32, device=device)
    
    softmax_scale = 1.0 / (qk_nope_head_dim ** 0.5)
    
    # Initialize fast kernel
    print("\nInitializing CuteDSL kernel...")
    fast_mla = CuteDSLMLAWrapper(
        latent_dim=latent_dim,
        rope_dim=qk_rope_head_dim,
        num_heads=num_heads,
        in_dtype=cutlass.Float16,
        out_dtype=cutlass.Float16,
    )
    
    # Warm up both implementations
    print("\nWarming up...")
    for _ in range(3):
        _ = slow_mla_attn(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, softmax_scale, kv_length)
        _ = fast_mla(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, softmax_scale, kv_length)
    torch.cuda.synchronize()
    
    # Run slow implementation
    print("\nRunning slow_mla_attn...")
    torch.cuda.synchronize()
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        output_slow = slow_mla_attn(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, softmax_scale, kv_length)
    torch.cuda.synchronize()
    time_slow = (time.time() - start) / num_iters * 1000  # ms
    
    # Run fast implementation
    print("Running CuteDSL kernel...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        output_fast = fast_mla(q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, softmax_scale, kv_length)
    torch.cuda.synchronize()
    time_fast = (time.time() - start) / num_iters * 1000  # ms
    
    # Compare outputs
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Output shape: {output_fast.shape}")
    print(f"slow_mla_attn time: {time_slow:.3f} ms")
    print(f"CuteDSL kernel time: {time_fast:.3f} ms")
    print(f"Speedup: {time_slow / time_fast:.2f}x")
    
    # Check correctness
    max_diff = torch.max(torch.abs(output_slow - output_fast)).item()
    mean_diff = torch.mean(torch.abs(output_slow - output_fast)).item()
    
    print(f"\nNumerical difference:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    
    # Set tolerance based on dtype
    if dtype == torch.bfloat16:
        tolerance = 0.1
    else:
        tolerance = 0.01
    
    if max_diff < tolerance and mean_diff < tolerance:
        print(f"✓ PASSED: Outputs match within tolerance ({tolerance})")
    else:
        print(f"✗ FAILED: Outputs differ (tolerance: {tolerance})")
        print(f"\nSample outputs (first 5 elements):")
        print(f"  Slow: {output_slow[0, :5]}")
        print(f"  Fast: {output_fast[0, :5]}")
    
    print("=" * 80)
    
    return output_slow, output_fast, time_slow, time_fast


if __name__ == "__main__":
    # Check if we have a Blackwell GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        exit(1)
    
    props = torch.cuda.get_device_properties(0)
    if props.major != 10:
        print(f"WARNING: This example requires Blackwell (SM100) GPU")
        print(f"Current GPU: {props.name} (compute capability {props.major}.{props.minor})")
        print("The kernel will likely fail to compile.")
    
    try:
        output_slow, output_fast, time_slow, time_fast = test_decode_step()
        
        print("\n" + "=" * 80)
        print("Summary:")
        print("=" * 80)
        print(f"✓ Test completed successfully")
        print(f"✓ CuteDSL kernel is {time_slow / time_fast:.2f}x faster")
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
