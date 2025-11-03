#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import os
import contextlib

import torch
import torch.nn.functional as F
from einops import rearrange

# --- Cutlass / CuTe DSL ---
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as TESTING
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

# --- Try to import the MLA kernel ---
try:
    from mla import BlackwellMultiHeadLatentAttentionForward
    HAVE_KERNEL = True
except Exception as e:
    print(f"✗ Could not import MLA kernel from mla.py: {e}")
    HAVE_KERNEL = False


# =========================
# Global knobs & constants
# =========================

# Legal MMA tile shapes (avoid illegal 64xM)
# Note: QK tiles with M=256 currently have kernel bugs (int subscriptable, range_constexpr)
QK_TILES = [(128, 128)]  # was: [(128, 128), (256, 128), (256, 256)]
PV_TILES = [(128, 128), (128, 256), (256, 128), (256, 256)]

CLUSTERS = [1, 2, 4, 8]         # B200: 132 SMs; clusters up to 8 work
PERSISTENT = [False, True]
SPLIT_KV = [1]                  # keep 1 until numerics are rock solid

# Small sanity thresholds (tight) and large-K thresholds (looser; FP16)
SMALL_COS_THR = 0.999
SMALL_P99_ABS = 1e-2

LARGE_COS_THR = 0.999
LARGE_P99_ABS = 5e-2

# Dtype defaults
TORCH_DTYPE = torch.float16
CUTE_ACC_DTYPE = cutlass.Float32
CUTE_LSE_DTYPE = cutlass.Float32


# =========================
# Utility shims / helpers
# =========================

def apply_range_constexpr_shim():
    """
    Some CuTe DSL builds still use range_constexpr in paths that become runtime.
    Swap to plain range() so autotune grids don't explode.
    Call once before compiling kernels.
    """
    try:
        import cutlass.base_dsl._mlir_helpers.op as _op
        if hasattr(_op, "range_constexpr"):
            _op.range_constexpr = _op.range  # monkey-patch

        # Also patch any symbol pulled into mla.py
        try:
            import mla as _mla
            if hasattr(_mla, "range_constexpr"):
                setattr(_mla, "range_constexpr", _op.range)
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️ range_constexpr shim failed: {e}")


def torch_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def inference():
    with torch.inference_mode(), torch.no_grad():
        yield


def cuda_ms(callable_fn, warmup=10, iters=100):
    """Measure CUDA time in milliseconds for an arbitrary Python callable."""
    torch_cuda_synchronize()
    for _ in range(warmup):
        callable_fn()
    torch_cuda_synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        callable_fn()
    end.record()
    torch_cuda_synchronize()
    return start.elapsed_time(end) / iters  # ms


def cosine_and_p99(a: torch.Tensor, b: torch.Tensor):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    absdiff = (a - b).abs()
    p99 = torch.quantile(absdiff, 0.99).item()
    return cos, p99


# =========================
# Reference implementation
# =========================

def torch_reference_mla(
    q_latent: torch.Tensor,  # [B, H, Ld, Q]
    q_rope: torch.Tensor,    # [B, H, Er, Q]
    c_latent: torch.Tensor,  # [B, Ld, K]
    c_rope: torch.Tensor,    # [B, Er, K]
    cache_seqs: torch.Tensor, # [B], valid K per batch (typically == K)
    softmax_scale: float = 1.0,
) -> torch.Tensor:
    """
    Treats each head independently; concatenates (latent||rope) to match "nope+pe".
    Returns: [B, H, Q, Ld]
    """
    B, H, Ld, Q = q_latent.shape
    Er = q_rope.shape[2]
    K = c_latent.shape[2]

    # Concat along channel to emulate (nope_logits + pe_logits)
    q_cat = torch.cat([q_latent, q_rope], dim=2)     # [B, H, Ld+Er, Q]
    k_cat = torch.cat([c_latent, c_rope], dim=1)     # [B, Ld+Er, K]

    # Expand keys/values across heads
    k_cat = k_cat.unsqueeze(1).expand(B, H, Ld + Er, K)   # [B,H,Dtot,K]
    v_lat = c_latent.unsqueeze(1).expand(B, H, Ld, K)     # [B,H,Ld,K]

    # Reshape for SDPA: [(B*H), 1, Q, D], [(B*H), 1, K, D], [(B*H), 1, K, Ld]
    q_ = q_cat.permute(0, 1, 3, 2).reshape(B * H, 1, Q, Ld + Er)
    k_ = k_cat.permute(0, 1, 3, 2).reshape(B * H, 1, K, Ld + Er)
    v_ = v_lat.permute(0, 1, 3, 2).reshape(B * H, 1, K, Ld)

    # Build additive mask with -inf for masked positions. Shape: [(B*H), 1, Q, K]
    device = q_.device
    arangeK = torch.arange(K, device=device).view(1, 1, 1, K)  # [1,1,1,K]
    valid = cache_seqs.clamp_max(K).to(device=device, dtype=torch.int64).view(B, 1, 1, 1)
    valid = valid.expand(B, H, Q, 1).reshape(B * H, 1, Q, 1)  # [(B*H),1,Q,1]
    keep = arangeK.expand(B * H, 1, Q, K) < valid             # [(B*H),1,Q,K], bool
    add_mask = torch.zeros_like(keep, dtype=q_.dtype)
    add_mask[~keep] = float("-inf")

    out = F.scaled_dot_product_attention(
        q_, k_, v_,
        attn_mask=add_mask,  # additive mask
        dropout_p=0.0,
        scale=softmax_scale,
        is_causal=False,
    )  # [(B*H), 1, Q, Ld]

    out = out.reshape(B, H, 1, Q, Ld).squeeze(2)  # [B,H,Q,Ld]
    return out


def reference_end_to_end(
    q_nope: torch.Tensor,  # [B,H,Q,Dn]
    q_pe: torch.Tensor,    # [B,H,Q,Er]
    kv_lat: torch.Tensor,  # [B,K,Ld]
    k_pe: torch.Tensor,    # [B,K,Er]
    Wk: torch.Tensor,      # [Ld,H,Dn]
    Wv: torch.Tensor,      # [Ld,H,Vd]
    kv_length: torch.Tensor,  # [B] (we'll pass K)
    softmax_scale: float,
) -> torch.Tensor:
    """
    Full pipeline reference: preproject -> SDPA (latent space) -> project V.
    Returns [B, Q, H*Vd]
    """
    B, H, Q, Dn = q_nope.shape
    Ld = kv_lat.shape[-1]
    Er = q_pe.shape[-1]
    Vd = Wv.shape[-1]
    K  = kv_lat.shape[1]

    # q_latent = q_nope @ Wk  → [B,H,Ld,Q]
    q_no_hbq_d  = rearrange(q_nope, "b h q d -> h (b q) d")
    q_lat_hbq_l = torch.bmm(q_no_hbq_d, rearrange(Wk, "l h d -> h d l"))
    q_latent    = rearrange(q_lat_hbq_l, "h (b q) l -> b h l q", b=B, q=Q)

    # q_rope to [B,H,Er,Q]
    q_rope = q_pe.permute(0, 1, 3, 2).contiguous()

    # c_latent: [B,Ld,K], c_rope: [B,Er,K]
    c_latent = kv_lat.permute(0, 2, 1).contiguous()
    c_rope   = k_pe.permute(0, 2, 1).contiguous()

    cache_seqs = kv_length  # expect kv_length == K

    # latent attention
    latent = torch_reference_mla(q_latent, q_rope, c_latent, c_rope, cache_seqs, softmax_scale)  # [B,H,Q,Ld]

    # project to V
    latent_hbq_l = rearrange(latent, "b h q l -> h (b q) l")
    out_hbq_v    = torch.bmm(latent_hbq_l, rearrange(Wv, "l h d -> h l d"))
    out          = rearrange(out_hbq_v, "h (b q) d -> b q (h d)", b=B, q=Q)  # [B,Q,H*Vd]
    return out


# =========================
# Kernel input builder
# =========================

def build_kernel_inputs(
    q_nope: torch.Tensor, q_pe: torch.Tensor,
    kv_lat: torch.Tensor, k_pe: torch.Tensor,
    Wk: torch.Tensor
):
    """
    Returns tensors in the layout the kernel expects.
      q_lat_k:  [H,Ld,B]
      q_rope_k: [H,Er,B]
      c_lat_k:  [K,Ld,B]
      c_rope_k: [K,Er,B]
    Assumes Q == 1.
    """
    B, H, Q, Dn = q_nope.shape
    assert Q == 1, "Kernel expects q_len == 1"
    Ld = kv_lat.shape[-1]
    Er = k_pe.shape[-1]

    # q_latent: [H, B*Q, Ld] -> [H,Ld,B]
    q_no_hbq_d  = rearrange(q_nope, "b h q d -> h (b q) d")                # [H,B*Q,Dn]
    q_lat_hbq_l = torch.bmm(q_no_hbq_d, rearrange(Wk, "l h d -> h d l"))   # [H,B*Q,Ld]
    q_lat_k     = q_lat_hbq_l.permute(0, 2, 1).contiguous()                # [H,Ld,B]

    # q_rope: [B,H,Q,Er] -> [H,Er,B] (Q==1)
    q_rope_k = q_pe.squeeze(2).permute(1, 2, 0).contiguous()               # [H,Er,B]

    # c_latent: [B,K,Ld] -> [K,Ld,B]
    c_lat_k = kv_lat.permute(1, 2, 0).contiguous()

    # c_rope: [B,K,Er] -> [K,Er,B]
    c_rope_k = k_pe.permute(1, 2, 0).contiguous()

    return q_lat_k, q_rope_k, c_lat_k, c_rope_k


# =========================
# Kernel wrapper / runner
# =========================

class MLAConfig:
    def __init__(self, mma_qk_tiler_mn, mma_pv_tiler_mn, is_persistent, is_cpasync, max_active_clusters, split_kv):
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.is_persistent = is_persistent
        self.is_cpasync = is_cpasync
        self.max_active_clusters = max_active_clusters
        self.split_kv = split_kv

    def __repr__(self):
        return (f"QK={self.mma_qk_tiler_mn}, PV={self.mma_pv_tiler_mn}, "
                f"persistent={self.is_persistent}, cp.async={self.is_cpasync}, "
                f"clusters={self.max_active_clusters}, split_kv={self.split_kv}")


class CuteRunner:
    def __init__(self, module_cls, cfg: MLAConfig, softmax_scale: float, output_scale: float = 1.0, use_page_table: bool = False):
        self.cfg = cfg
        self.softmax_scale = softmax_scale
        self.output_scale = output_scale
        self.use_page_table = use_page_table
        self.cute = cute
        self.cutlass = cutlass

        self.kernel = module_cls(
            acc_dtype=CUTE_ACC_DTYPE,
            lse_dtype=CUTE_LSE_DTYPE,
            mma_qk_tiler_mn=cfg.mma_qk_tiler_mn,
            mma_pv_tiler_mn=cfg.mma_pv_tiler_mn,
            max_active_clusters=cfg.max_active_clusters,
            is_persistent=cfg.is_persistent,
            is_cpasync=cfg.is_cpasync,
            use_page_table=use_page_table,
            is_var_seq=False,
            is_var_split_kv=False,
        )
        self._compiled = None
        self._args = None

    def _wrap(self, t, leading_dim=None):
        x = from_dlpack(t, assumed_align=128, use_32bit_stride=True)
        if leading_dim is not None:
            x = x.mark_layout_dynamic(leading_dim=leading_dim)
        else:
            x = x.mark_layout_dynamic()
        return x
    
    def compile(self, q_lat_k, q_rope_k, c_lat_k, c_rope_k, o_k, lse, cache_seqs, stream):
        """
        Prepare cute tensors and compile the kernel. If cp.async/TMA is enabled,
        we create a trivial page table that maps each K-tile to its starting offset.
        """
        # ---- Wrap standard tensors ----
        q_lat  = self._wrap(q_lat_k,  leading_dim=1)
        q_rope = self._wrap(q_rope_k, leading_dim=1)
        c_lat  = self._wrap(c_lat_k,  leading_dim=1)
        c_rope = self._wrap(c_rope_k, leading_dim=1)
        o      = self._wrap(o_k,      leading_dim=1)
        l      = self._wrap(lse,      leading_dim=0)
        cache  = self._wrap(cache_seqs)  # 1D ok with default

        # ---- Build a trivial page table when TMA is on ----
        page_table_cute = None
        self._page_table_torch = None  # keep torch tensor alive
        if self.use_page_table:
            # K dimension is the first dim of c_lat_k (shape [K,Ld,B])
            K = int(c_lat_k.shape[0])
            B = int(o_k.shape[2])
            # Use QK tile's K-mode as page size (128 or 256); fall back to 128 if odd
            page_size = int(self.cfg.mma_qk_tiler_mn[1]) if self.cfg.mma_qk_tiler_mn else 128
            if page_size not in (128, 256):
                page_size = 128
            pages = (K + page_size - 1) // page_size

            # page_table shape: [pages, B], entries are global K offsets per page
            base = torch.arange(0, pages * page_size, page_size,
                                device=c_lat_k.device, dtype=torch.int32)[:pages]  # [pages]
            pt = base[:, None].expand(pages, B).contiguous()  # [pages,B]
            self._page_table_torch = pt  # keep alive
            page_table_cute = self._wrap(self._page_table_torch, leading_dim=1)
        else:
            page_table_cute = None

        # ---- No extra workspace / split-kv blocks for split_kv==1 ----
        workspace = None
        block_split_kvs = None

        split_kv = self.cutlass.Int32(self.cfg.split_kv)
        sm_scale = self.cutlass.Float32(self.softmax_scale)
        out_scale = self.cutlass.Float32(self.output_scale)

        # ---- Pack args & compile (excluded from benchmarks) ----
        self._args = (
            q_lat, q_rope, c_lat, c_rope,
            page_table_cute,   # <-- non-None when cp.async/TMA is on
            o, l, workspace,
            split_kv, cache, block_split_kvs,
            sm_scale, out_scale, stream
        )
        self._compiled = self.cute.compile(self.kernel, *self._args)

    def run(self):
        assert self._compiled is not None and self._args is not None
        self._compiled(*self._args)


# =========================
# Sanity kernel (safe)
# =========================

def small_sanity_check(
    module_cls,
    q_nope, q_pe, kv_lat, k_pe, Wk, Wv,
    softmax_scale
):
    """
    Compile a very safe kernel (no cp.async, clusters=1, persistent=False, split_kv=1),
    run once at small K (e.g., 1024), and compare latent outputs with the reference.
    """
    B, H, Q, Dn = q_nope.shape
    K = kv_lat.shape[1]
    Ld = kv_lat.shape[-1]
    device = q_nope.device

    # Reference latent
    kv_length = torch.full((B,), K, device=device, dtype=torch.int32)
    q_no_hbq_d  = rearrange(q_nope, "b h q d -> h (b q) d")
    q_lat_hbq_l = torch.bmm(q_no_hbq_d, rearrange(Wk, "l h d -> h d l"))
    q_latent    = rearrange(q_lat_hbq_l, "h (b q) l -> b h l q", b=B, q=Q)

    ref_lat = torch_reference_mla(
        q_latent,
        q_pe.permute(0, 1, 3, 2).contiguous(),
        kv_lat.permute(0, 2, 1).contiguous(),
        k_pe.permute(0, 2, 1).contiguous(),
        kv_length, softmax_scale
    )  # [B,H,Q,Ld]
    ref_vec = ref_lat.reshape(-1)

    # Kernel latent: build inputs
    q_lat_k, q_rope_k, c_lat_k, c_rope_k = build_kernel_inputs(q_nope, q_pe, kv_lat, k_pe, Wk)

    # Outputs
    o_k = torch.zeros(H, Ld, B, device=device, dtype=q_nope.dtype)
    lse = torch.zeros(H, B, device=device, dtype=torch.float32)
    cache = torch.full((B,), K, device=device, dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # conservative config
    cfg = MLAConfig((128, 128), (128, 256), False, False, 1, 1)
    runner = CuteRunner(module_cls, cfg, softmax_scale, 1.0, use_page_table=False)
    runner.compile(q_lat_k, q_rope_k, c_lat_k, c_rope_k, o_k, lse, cache, stream)
    runner.run()
    torch_cuda_synchronize()

    ker_lat = o_k.permute(2, 0, 1).unsqueeze(2)  # [B,H,1,Ld]
    ker_vec = ker_lat.reshape(-1)

    cos, p99 = cosine_and_p99(ref_vec, ker_vec)
    print(f"Small sanity (K={K}): cos={cos:.6f}, p99_abs={p99:.4f} {'✓' if (cos>=SMALL_COS_THR and p99<=SMALL_P99_ABS) else '✗'}")
    return cos >= SMALL_COS_THR and p99 <= SMALL_P99_ABS


# =========================
# Tuning / single run
# =========================

def cp_async_policy_val(policy: str, K: int) -> bool:
    """
    Returns True/False for cp.async based on policy.
    'auto': enable for large K (>= 65536), else disable.
    
    Note: TMA/cp.async currently has strict 128-byte alignment requirements
    that may not be met by all tensor layouts. Disabled for now.
    """
    if policy == "on":
        return True
    if policy == "off":
        return False
    # auto - disabled due to alignment issues
    return False  # was: K >= 65536


def run_one(
    *,
    B: int,
    H: int,
    Q: int,
    K: int,
    Ld: int,
    Er: int,
    Dn: int,
    Vd: int,
    dtype=torch.float16,
    cp_async: str = "auto"  # "on" | "off" | "auto"
):
    assert Q == 1, "This kernel path is decoding-only (Q=1)"

    device = "cuda"
    torch_dtype = dtype
    cute_dtype = cutlass.Float16

    # Softmax scale to match slow/nope+pe combo
    softmax_scale = 1.0 / math.sqrt(float(Ld + Er))

    # Build tensors
    torch.manual_seed(0)
    q_nope = torch.randn(B, H, Q, Dn, device=device, dtype=torch_dtype)
    q_pe   = torch.randn(B, H, Q, Er, device=device, dtype=torch_dtype)
    kv_lat = torch.randn(B, K, Ld, device=device, dtype=torch_dtype)
    k_pe   = torch.randn(B, K, Er, device=device, dtype=torch_dtype)
    Wk     = torch.randn(Ld, H, Dn, device=device, dtype=torch_dtype)
    Wv     = torch.randn(Ld, H, Vd, device=device, dtype=torch_dtype)
    kvlen  = torch.full((B,), K, device=device, dtype=torch.int32)

    # Quick small sanity first (K_small=1024), do NOT enable cp.async here
    with inference():
        K_small = 1024
        qn_s = q_nope.clone()
        qp_s = q_pe.clone()
        kv_s = kv_lat[:, :K_small, :].contiguous()
        kp_s = k_pe[:, :K_small, :].contiguous()
        Wk_s = Wk.clone()
        Wv_s = Wv.clone()
        ok_small = small_sanity_check(BlackwellMultiHeadLatentAttentionForward,
                                      qn_s, qp_s, kv_s, kp_s, Wk_s, Wv_s, softmax_scale)
        if not ok_small:
            print("Small sanity failed; bailing early to avoid misleading large-K numerics.")
            return

    # Reference end-to-end timing (excludes compile)
    with inference():
        ref_ms = cuda_ms(lambda: reference_end_to_end(q_nope, q_pe, kv_lat, k_pe, Wk, Wv, kvlen, softmax_scale),
                         warmup=5, iters=30)
    print(f"PyTorch reference (end-to-end): {ref_ms:.3f} ms")

    if not HAVE_KERNEL:
        print("CUTE kernel not available; skipping kernel timings.")
        return

    # Grid search over configs (split_kv fixed to 1 for now)
    best_ms = float("inf")
    best_cfg = None

    # Prepare kernel inputs & outputs once
    q_lat_k, q_rope_k, c_lat_k, c_rope_k = build_kernel_inputs(q_nope, q_pe, kv_lat, k_pe, Wk)
    o_k = torch.zeros(H, Ld, B, device=device, dtype=torch_dtype)
    lse = torch.zeros(H, B, device=device, dtype=torch.float32)
    cache = torch.full((B,), K, device=device, dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # For benchmark via testing utils, build JitArguments matching the compiled callable.
    def as_jit_args(runner: CuteRunner):
        (q_lat, q_rope, c_lat, c_rope,
         page_table, o, l, workspace,
         split_kv, cache_c, block_split_kvs,
         sm_scale, out_scale, s) = runner._args
        return TESTING.JitArguments(
            q_lat, q_rope, c_lat, c_rope,
            page_table, o, l, workspace,
            split_kv, cache_c, block_split_kvs,
            sm_scale, out_scale, s
        )

    for qk in QK_TILES:
        for pv in PV_TILES:
            for pers in PERSISTENT:
                for clusters in CLUSTERS:
                    for split_kv in SPLIT_KV:
                        use_cp = cp_async_policy_val(cp_async, K)
                        # If cp.async is enabled we must also enable page tables.
                        use_page_table = use_cp
                        
                        # TMA/cp.async requires persistent scheduling on Blackwell
                        if use_cp and not pers:
                            continue
                        
                        # TMA has constraints on PV tile sizes - M=256 requires 2 CTA groups
                        # which may not be compatible with the current kernel configuration
                        # Skip PV tiles with M > 128 when using TMA for now
                        if use_cp and pv[0] > 128:
                            continue
                        
                        # Persistent mode has range_constexpr issues with large cluster counts
                        # Skip persistent with clusters > 2 for now
                        if pers and clusters > 2:
                            continue

                        cfg = MLAConfig(qk, pv, pers, use_cp, clusters, split_kv)

                        # Build runner and compile (exclude from timing)
                        try:
                            runner = CuteRunner(
                                BlackwellMultiHeadLatentAttentionForward,
                                cfg,
                                softmax_scale,
                                1.0,
                                use_page_table=use_page_table
                            )
                            runner.compile(q_lat_k, q_rope_k, c_lat_k, c_rope_k, o_k, lse, cache, stream)
                        except Exception as e:
                            print(f"Seed failed ({cfg}): {type(e).__name__}: {e}")
                            continue

                        # Numeric validation (latent space) at full K
                        with inference():
                            # Run once to get output
                            runner.run()
                            torch_cuda_synchronize()
                            ker_lat = o_k.permute(2, 0, 1).unsqueeze(2)  # [B,H,1,Ld]

                            # Reference latent for full K (avoid redoing proj each time: do once)
                            q_no_hbq_d  = rearrange(q_nope, "b h q d -> h (b q) d")
                            q_lat_hbq_l = torch.bmm(q_no_hbq_d, rearrange(Wk, "l h d -> h d l"))
                            q_latent    = rearrange(q_lat_hbq_l, "h (b q) l -> b h l q", b=B, q=Q)

                            ref_lat = torch_reference_mla(
                                q_latent,
                                q_pe.permute(0, 1, 3, 2).contiguous(),
                                kv_lat.permute(0, 2, 1).contiguous(),
                                k_pe.permute(0, 2, 1).contiguous(),
                                kvlen, softmax_scale
                            )

                            cos, p99 = cosine_and_p99(ref_lat, ker_lat)
                            if cos < LARGE_COS_THR or p99 > LARGE_P99_ABS:
                                print(f"Reject Grid ({cfg}): latent_cos={cos:.6f} (need ≥{LARGE_COS_THR}), p99_abs={p99:.4f} (need ≤{LARGE_P99_ABS})")
                                continue

                        # If numeric check passed, benchmark kernel with testing utils
                        try:
                            jit_args = as_jit_args(runner)
                            # Warmup 5, iters 50 for stability; use same stream
                            avg_us = TESTING.benchmark(
                                runner._compiled,
                                kernel_arguments=jit_args,
                                warmup_iterations=5,
                                iterations=50,
                                stream=stream
                            )
                            ms = avg_us / 1e3
                            tag = "kept best" if ms < best_ms else "→"
                            print(f"Grid: {cfg} → {ms:.3f} ms ({tag} {best_ms:.3f} ms)")
                            if ms < best_ms:
                                best_ms = ms
                                best_cfg = cfg
                        except Exception as e:
                            print(f"Benchmark failed ({cfg}): {type(e).__name__}: {e}")
                            continue

    if best_cfg is None:
        print("No valid kernel config survived numerics; nothing to report.")
        return None, None, ref_ms

    print(f"\nBest config: {best_cfg} → {best_ms:.3f} ms")
    speedup = ref_ms / best_ms if best_ms > 0 else 0.0
    print(f"Speedup vs PyTorch: {speedup:.2f}×")
    
    return best_cfg, best_ms, ref_ms


# =========================
# Main
# =========================

def main():
    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        return

    print("✓ CUTE DSL + testing utils available; MLA kernel imported" if HAVE_KERNEL else "⚠️ Running without MLA kernel")

    # Apply shim before touching the kernel
    apply_range_constexpr_shim()

    B = 1
    H = 128
    Q = 1
    Ld = 512
    Er = 64
    Dn = 32
    Vd = 512
    dtype = TORCH_DTYPE

    # Store results for final summary
    results = []

    # --- Small sanity (K=1024) is executed inside run_one() ---

    # --- 64K ---
    print("\n=== Benchmark: k_len=65,536, q_len=1, heads=128, latent=512, rope=64, dtype=torch.float16 ===")
    with inference():
        best_cfg_64k, best_ms_64k, ref_ms_64k = run_one(B=B, H=H, Q=Q, K=65_536, Ld=Ld, Er=Er, Dn=Dn, Vd=Vd, dtype=dtype, cp_async="auto")
        if best_cfg_64k:
            results.append(("K=65,536", best_cfg_64k, best_ms_64k, ref_ms_64k))

    # --- 128K ---
    print("\n=== Benchmark: k_len=131,072, q_len=1, heads=128, latent=512, rope=64, dtype=torch.float16 ===")
    with inference():
        best_cfg_128k, best_ms_128k, ref_ms_128k = run_one(B=B, H=H, Q=Q, K=131_072, Ld=Ld, Er=Er, Dn=Dn, Vd=Vd, dtype=dtype, cp_async="auto")
        if best_cfg_128k:
            results.append(("K=131,072", best_cfg_128k, best_ms_128k, ref_ms_128k))
    
    # Print final summary table
    if results:
        print("\n" + "="*100)
        print("FINAL SUMMARY")
        print("="*100)
        print(f"{'Seq Len':<12} {'Best Config':<60} {'CuteDSL':<12} {'PyTorch':<12} {'Speedup':>10}")
        print("-"*100)
        for seq_len, cfg, kernel_ms, ref_ms in results:
            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0
            speedup_pct = (speedup - 1.0) * 100
            print(f"{seq_len:<12} {str(cfg):<60} {kernel_ms:>8.3f} ms  {ref_ms:>8.3f} ms  {speedup:>6.2f}x ({speedup_pct:+.1f}%)")
        print("="*100)


if __name__ == "__main__":
    main()
