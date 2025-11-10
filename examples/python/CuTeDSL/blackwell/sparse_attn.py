import torch
import math

def sparse_attention_forward(Q, kv, indices, sm_scale):
    """
    Q: [s_q, h_q, d_qk], bfloat16
    kv: [s_kv, h_kv, d_qk], bfloat16 (h_kv must be 1)
    indices: [s_q, h_kv, topk], int32 (h_kv must be 1)
    """
    kv = kv.squeeze(1)  # [s_kv, d_qk]
    indices = indices.squeeze(1)  # [s_q, topk]
    focused_kv = kv[indices]  # [s_q, topk, d_qk]
    
    log2e = math.log2(math.e)
    P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * log2e  # [s_q, h_q, topk]
    
    lse = torch.logsumexp(P * math.log(2), dim=-1) / math.log(2)  # [s_q, h_q]
    S = torch.exp2(P - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    
    out = S @ focused_kv  # [s_q, h_q, d_qk]
    
    return out, S, P, lse, focused_kv, indices, sm_scale


def sparse_attention_backward(grad_out, S, P, focused_kv, Q, indices, kv_shape, sm_scale):
    """
    Backward pass for sparse attention.
    """
    s_q, h_q, d_qk = grad_out.shape
    topk = S.shape[-1]
    s_kv, _ = kv_shape
    
    log2e = math.log2(math.e)
    ln2 = math.log(2)
    
    # Backward through: out = S @ focused_kv
    grad_S = grad_out @ focused_kv.transpose(-1, -2)  # [s_q, h_q, topk]
    grad_focused_kv = S.transpose(-1, -2) @ grad_out  # [s_q, topk, d_qk]
    
    # Backward through base-2 softmax
    sum_grad_S_times_S = (grad_S * S).sum(dim=-1, keepdim=True)  # [s_q, h_q, 1]
    grad_P = ln2 * S * (grad_S - sum_grad_S_times_S)  # [s_q, h_q, topk]
    
    # Backward through: P = (Q @ focused_kv.T) * sm_scale * log2(e)
    grad_QK = grad_P * sm_scale * log2e  # [s_q, h_q, topk]
    
    grad_Q = grad_QK @ focused_kv  # [s_q, h_q, d_qk]
    grad_focused_kv_from_matmul = Q.transpose(-1, -2) @ grad_QK
    grad_focused_kv_from_matmul = grad_focused_kv_from_matmul.transpose(-1, -2)  # [s_q, topk, d_qk]
    
    grad_focused_kv = grad_focused_kv + grad_focused_kv_from_matmul
    
    # Backward through gather: scatter gradients back
    grad_kv = torch.zeros(s_kv, d_qk, dtype=grad_focused_kv.dtype, device=grad_focused_kv.device)
    
    indices_flat = indices.reshape(-1)
    grad_focused_kv_flat = grad_focused_kv.reshape(-1, d_qk)
    
    grad_kv.index_add_(0, indices_flat, grad_focused_kv_flat)
    grad_kv = grad_kv.unsqueeze(1)  # [s_kv, 1, d_qk]
    
    return grad_Q, grad_kv


class SparseAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, kv, indices, sm_scale):
        out, S, P, lse, focused_kv, indices_squeezed, _ = sparse_attention_forward(Q, kv, indices, sm_scale)
        ctx.save_for_backward(Q, S, P, focused_kv, indices_squeezed)
        ctx.sm_scale = sm_scale
        ctx.kv_shape = (kv.shape[0], kv.shape[2])  # [s_kv, d_qk]
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        Q, S, P, focused_kv, indices = ctx.saved_tensors
        grad_Q, grad_kv = sparse_attention_backward(
            grad_out, S, P, focused_kv, Q, indices, ctx.kv_shape, ctx.sm_scale
        )
        return grad_Q, grad_kv, None, None


def test_backward_correctness():
    """
    Verify our custom backward matches PyTorch autograd on the same operations.
    """
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Small sizes for easier debugging
    s_q, s_kv, h_q, h_kv, d_qk, topk = 16, 64, 4, 1, 32, 8
    sm_scale = 1.0 / math.sqrt(d_qk)
    
    print("=" * 80)
    print("Testing Sparse Attention Backward Pass")
    print("=" * 80)
    print(f"Config: s_q={s_q}, s_kv={s_kv}, h_q={h_q}, d_qk={d_qk}, topk={topk}")
    print(f"Device: {device}")
    print()
    
    # Create inputs (using float32 for better numerical accuracy in testing)
    Q = torch.randn(s_q, h_q, d_qk, dtype=torch.float32, device=device, requires_grad=True)
    kv = torch.randn(s_kv, h_kv, d_qk, dtype=torch.float32, device=device, requires_grad=True)
    indices = torch.randint(0, s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=device)
    
    # ===== Test with autograd (reference) =====
    print("Running PyTorch autograd (reference)...")
    Q_ref = Q.clone().detach().requires_grad_(True)
    kv_ref = kv.clone().detach().requires_grad_(True)
    
    # Forward pass using autograd
    kv_squeezed = kv_ref.squeeze(1)
    indices_squeezed = indices.squeeze(1)
    focused_kv = kv_squeezed[indices_squeezed]
    
    log2e = math.log2(math.e)
    P = (Q_ref @ focused_kv.transpose(-1, -2)) * sm_scale * log2e
    lse = torch.logsumexp(P * math.log(2), dim=-1) / math.log(2)
    S = torch.exp2(P - lse.unsqueeze(-1))
    out_ref = S @ focused_kv
    
    # Backward with random gradient
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    
    grad_Q_ref = Q_ref.grad.clone()
    grad_kv_ref = kv_ref.grad.clone()
    
    print(f"✓ Autograd backward complete")
    print()
    
    # ===== Test with custom backward =====
    print("Running custom backward implementation...")
    Q_custom = Q.clone().detach().requires_grad_(True)
    kv_custom = kv.clone().detach().requires_grad_(True)
    
    # Use our custom autograd function
    out_custom = SparseAttentionFunc.apply(Q_custom, kv_custom, indices, sm_scale)
    out_custom.backward(grad_out)
    
    grad_Q_custom = Q_custom.grad
    grad_kv_custom = kv_custom.grad
    
    print(f"✓ Custom backward complete")
    print()
    
    # ===== Compare gradients =====
    print("=" * 80)
    print("Gradient Comparison")
    print("=" * 80)
    
    # Check Q gradient
    q_diff = (grad_Q_ref - grad_Q_custom).abs()
    q_rel_error = (q_diff / (grad_Q_ref.abs() + 1e-8)).mean()
    print(f"grad_Q:")
    print(f"  Max absolute diff: {q_diff.max().item():.2e}")
    print(f"  Mean absolute diff: {q_diff.mean().item():.2e}")
    print(f"  Mean relative error: {q_rel_error.item():.2e}")
    print(f"  Allclose (rtol=1e-4, atol=1e-5): {torch.allclose(grad_Q_ref, grad_Q_custom, rtol=1e-4, atol=1e-5)}")
    print()
    
    # Check KV gradient
    kv_diff = (grad_kv_ref - grad_kv_custom).abs()
    kv_rel_error = (kv_diff / (grad_kv_ref.abs() + 1e-8)).mean()
    print(f"grad_kv:")
    print(f"  Max absolute diff: {kv_diff.max().item():.2e}")
    print(f"  Mean absolute diff: {kv_diff.mean().item():.2e}")
    print(f"  Mean relative error: {kv_rel_error.item():.2e}")
    print(f"  Allclose (rtol=1e-4, atol=1e-5): {torch.allclose(grad_kv_ref, grad_kv_custom, rtol=1e-4, atol=1e-5)}")
    print()
    
    # Overall pass/fail
    q_pass = torch.allclose(grad_Q_ref, grad_Q_custom, rtol=1e-4, atol=1e-5)
    kv_pass = torch.allclose(grad_kv_ref, grad_kv_custom, rtol=1e-4, atol=1e-5)
    
    print("=" * 80)
    if q_pass and kv_pass:
        print("✅ ALL TESTS PASSED - Backward implementation is correct!")
    else:
        print("❌ TESTS FAILED - Check implementation")
        if not q_pass:
            print("  - grad_Q mismatch")
        if not kv_pass:
            print("  - grad_kv mismatch")
    print("=" * 80)
    
    return q_pass and kv_pass


if __name__ == "__main__":
    success = test_backward_correctness()
    exit(0 if success else 1)