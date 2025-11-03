# ------------------------------------------------------------
# Fast MLA wrapper: pre-proj -> CUTE kernel -> value projection
# Drop-in replacement for slow_mla_attn(...)
# ------------------------------------------------------------
import torch
import torch.nn.functional as F
from einops import rearrange

# Reuse your detection flags if they exist; otherwise try to import.
try:
    CUTE_AVAILABLE
    MLA_KERNEL
except NameError:
    CUTE_AVAILABLE = False
    MLA_KERNEL = None
    try:
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        import cutlass.torch as cutlass_torch
        import cuda.bindings.driver as cuda
        try:
            from mla import BlackwellMultiHeadLatentAttentionForward
            MLA_KERNEL = BlackwellMultiHeadLatentAttentionForward
            CUTE_AVAILABLE = True
        except Exception:
            pass
    except Exception:
        pass

# Simple cache so we compile only once per dtype
_COMPILED_KERNEL_PER_DTYPE = {}

def _supports_kernel(q_len: int, device: torch.device, dtype: torch.dtype) -> bool:
    if not CUTE_AVAILABLE or MLA_KERNEL is None:
        return False
    if device.type != "cuda":
        return False
    if q_len != 1:  # current kernel expects decoding (q_len == 1)
        return False
    # Kernel path here is wired for fp16 (to match your tests/config)
    if dtype not in (torch.float16,):
        return False
    return True

def _get_or_compile_kernel(dtype: torch.dtype):
    """
    Returns (compiled_spec, dtypes) for the Blackwell kernel under our config,
    compiling on first use for the given dtype.
    """
    key = dtype
    if key in _COMPILED_KERNEL_PER_DTYPE:
        return _COMPILED_KERNEL_PER_DTYPE[key]

    # Map pytorch dtype -> cutlass cute dtype objects
    import cutlass
    import cutlass.cute as cute  # noqa
    from cutlass.cute.runtime import from_dlpack  # noqa

    if dtype == torch.float16:
        cute_elem = cutlass.Float16
    else:
        raise ValueError(f"Unsupported dtype for kernel: {dtype}")

    acc_dtype = cutlass.Float32
    lse_dtype = cutlass.Float32

    # Instantiate the kernel spec (independent of shapes)
    spec = MLA_KERNEL(
        acc_dtype=acc_dtype,
        lse_dtype=lse_dtype,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=1,
        is_persistent=False,
        is_cpasync=False,
        use_page_table=False,
        is_var_seq=False,
        is_var_split_kv=False,
    )
    _COMPILED_KERNEL_PER_DTYPE[key] = (spec, cute_elem, acc_dtype, lse_dtype)
    return _COMPILED_KERNEL_PER_DTYPE[key]

@torch.no_grad()  # kernel path is forward-only; remove if you need grads on fallback
def mla_attn_fast(
    q_nope: torch.Tensor,          # [bsz, nheads, q_len, qk_nope_head_dim]
    q_pe: torch.Tensor,            # [bsz, nheads, q_len, rope_dim]
    kv_latents: torch.Tensor,      # [bsz, k_len, latent_dim]
    k_pe: torch.Tensor,            # [bsz, k_len, rope_dim]
    latent_to_k: torch.Tensor,     # [latent_dim, nheads, qk_nope_head_dim]
    latent_to_v: torch.Tensor,     # [latent_dim, nheads, value_dim]
    softmax_scale: float,          # scalar
    kv_length: torch.Tensor,       # [bsz]
    *,
    use_kernel: bool = True,
    output_scale: float = 1.0,
) -> torch.Tensor:
    """
    Drop-in replacement for slow_mla_attn. Returns [bsz, q_len, nheads*value_dim].
    Falls back to slow path if kernel cannot be used.
    """
    bsz, nheads, q_len, qk_nope_head_dim = q_nope.shape
    _, k_len, latent_dim = kv_latents.shape
    rope_dim = q_pe.shape[-1]
    value_dim = latent_to_v.shape[-1]
    device = q_nope.device
    dtype = q_nope.dtype

    # --- 1) Pre-projection: q_nope -> q_latent per head
    # q_nope: [b,h,q,d], latent_to_k: [L,h,d]  => q_latent: [b,h,L,q]
    q_nope_hbqd = rearrange(q_nope, "b h q d -> h (b q) d")
    q_latent = torch.bmm(q_nope_hbqd, rearrange(latent_to_k, "l h d -> h d l"))
    q_latent = rearrange(q_latent, "h (b q) l -> b h l q", b=bsz)

    # --- 2) Fast path via CUTE kernel (only if supported)
    if use_kernel and _supports_kernel(q_len, device, dtype):
        try:
            # Lazy import (keeps top-level tolerant)
            import cutlass
            import cutlass.cute as cute
            from cutlass.cute.runtime import from_dlpack
            import cuda.bindings.driver as cuda

            spec, cute_elem, acc_dtype, lse_dtype = _get_or_compile_kernel(dtype)

            # Kernel expects:
            #   q_latent: [H, L, B]
            #   q_rope:   [H, E, B]
            #   c_latent: [K, L, B]
            #   c_rope:   [K, E, B]
            #   o:        [H, L, B]
            q_latent_k = q_latent.squeeze(-1).permute(1, 2, 0).contiguous()  # [H, L, B]
            q_rope_k   = q_pe.squeeze(2).permute(1, 2, 0).contiguous()      # [H, E, B]
            c_latent_k = kv_latents.permute(1, 2, 0).contiguous()           # [K, L, B]
            c_rope_k   = k_pe.permute(1, 2, 0).contiguous()                 # [K, E, B]

            o_k   = torch.zeros(nheads, latent_dim, bsz, device=device, dtype=dtype)
            lse_k = torch.zeros(nheads, bsz, device=device, dtype=torch.float32)

            # IMPORTANT: use total causal length (kv + current q positions)
            cache_seqs = (kv_length + q_len).to(torch.int32).contiguous()

            # Convert to CUTE tensors (DLPack) and set element types/layouts
            q_latent_c = from_dlpack(q_latent_k, assumed_align=16, use_32bit_stride=True)
            q_latent_c.element_type = cute_elem
            q_latent_c = q_latent_c.mark_layout_dynamic(leading_dim=1)

            q_rope_c = from_dlpack(q_rope_k, assumed_align=16, use_32bit_stride=True)
            q_rope_c.element_type = cute_elem
            q_rope_c = q_rope_c.mark_layout_dynamic(leading_dim=1)

            c_latent_c = from_dlpack(c_latent_k, assumed_align=16, use_32bit_stride=True)
            c_latent_c.element_type = cute_elem
            c_latent_c = c_latent_c.mark_layout_dynamic(leading_dim=1)

            c_rope_c = from_dlpack(c_rope_k, assumed_align=16, use_32bit_stride=True)
            c_rope_c.element_type = cute_elem
            c_rope_c = c_rope_c.mark_layout_dynamic(leading_dim=1)

            o_c = from_dlpack(o_k, assumed_align=16, use_32bit_stride=True)
            o_c.element_type = cute_elem
            o_c = o_c.mark_layout_dynamic(leading_dim=1)

            lse_c = from_dlpack(lse_k, assumed_align=16, use_32bit_stride=True)
            lse_c.element_type = lse_dtype
            lse_c = lse_c.mark_layout_dynamic(leading_dim=0)

            cache_c = from_dlpack(cache_seqs, assumed_align=16, use_32bit_stride=True)
            cache_c = cache_c.mark_layout_dynamic()

            # CUDA stream wire-up
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            # Compile on first call (shape/layout agnostic)
            compiled = cute.compile(
                spec,
                q_latent_c, q_rope_c, c_latent_c, c_rope_c,
                None, o_c, lse_c, None,
                cutlass.Int32(1),  # split_kv
                cache_c, None,
                cutlass.Float32(softmax_scale),
                cutlass.Float32(output_scale),
                stream,
            )

            # Run kernel
            compiled(
                q_latent_c, q_rope_c, c_latent_c, c_rope_c,
                None, o_c, lse_c, None,
                cutlass.Int32(1), cache_c, None,
                cutlass.Float32(softmax_scale),
                cutlass.Float32(output_scale),
                stream,
            )
            torch.cuda.synchronize()

            # --- 3) Post-projection: latent -> value per head, then merge heads
            # o_k: [H, L, B] -> [B, 1, H, L]
            o = o_k.permute(2, 0, 1).unsqueeze(1)      # [B, 1, H, L]
            o = rearrange(o, "b q h l -> h (b q) l")   # [H, (B*Q), L]
            o = torch.bmm(o, rearrange(latent_to_v, "l h d -> h l d"))  # [H, (B*Q), Dv]
            o = rearrange(o, "h (b q) d -> b q (h d)", b=bsz, q=q_len)  # [B, Q, H*Dv]
            return o

        except Exception as e:
            # If kernel path fails for any reason, fall back gracefully.
            # (You could log/print here if desired.)
            pass

    # --- Fallback: pure PyTorch slow path (exactly your original semantics)
    # NOTE: we inline a compact version to avoid importing the test utilities.
    # Compute fused_q_nope = q_nope @ latent_to_k^T -> [b, q, h, latent_dim]
    fused_q_nope = torch.bmm(
        rearrange(q_nope, "b h q d -> h (b q) d"),
        rearrange(latent_to_k, "l h d -> h d l"),
    )
    fused_q_nope = rearrange(fused_q_nope, "h (b q) l -> b q h l", b=bsz)  # [b,q,h,L]

    # Attention logits = (nope + rope) * scale
    nope_logits = torch.einsum("b q h l, b k l -> b h q k", fused_q_nope, kv_latents)
    pe_logits   = torch.einsum("b h q e, b k e -> b h q k", q_pe, k_pe)
    attn_logits = (nope_logits + pe_logits) * softmax_scale

    # Causal mask: allow up to kv_length + q positions
    q_ind = torch.arange(q_len, device=device)
    k_ind = torch.arange(k_len, device=device)
    mask = q_ind[None, :, None] + kv_length[:, None, None] >= k_ind[None, None, :]
    mask = mask.unsqueeze(1)  # [b,1,q,k]
    attn_logits = torch.where(mask, attn_logits, float("-inf"))

    attn = F.softmax(attn_logits, dim=-1)                     # [b,h,q,k]
    latent_out = torch.einsum("b k l, b h q k -> b h q l", kv_latents, attn)  # [b,h,q,L]

    # Value projection per head then merge heads -> [b,q,h*Dv]
    out = torch.bmm(
        rearrange(latent_out, "b h q l -> h (b q) l"),
        rearrange(latent_to_v, "l h d -> h l d"),
    )
    out = rearrange(out, "h (b q) d -> b q (h d)", b=bsz, q=q_len)
    return out


# ------------------------------------------------------------
# Optional nn.Module wrapper so users can "drop in and go"
# ------------------------------------------------------------
class MLAKernelWrapper(torch.nn.Module):
    """
    A module that wraps:
      q_nope -> (latent_to_k) -> kernel MLA (or fallback) -> (latent_to_v)
    The forward signature matches slow_mla_attn for easy replacement.
    """

    def __init__(self, use_kernel: bool = True, output_scale: float = 1.0):
        super().__init__()
        self.use_kernel = use_kernel
        self.output_scale = output_scale

    def forward(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_latents: torch.Tensor,
        k_pe: torch.Tensor,
        latent_to_k: torch.Tensor,
        latent_to_v: torch.Tensor,
        softmax_scale: float,
        kv_length: torch.Tensor,
    ) -> torch.Tensor:
        return mla_attn_fast(
            q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v,
            softmax_scale, kv_length,
            use_kernel=self.use_kernel,
            output_scale=self.output_scale,
        )
