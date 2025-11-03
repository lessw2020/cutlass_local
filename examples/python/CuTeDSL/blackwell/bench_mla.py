import torch
import time

def benchmark_decode_breakdown():
    """Profile where time is actually spent"""
    
    bsz, num_heads, q_len = 8, 128, 1
    qk_nope_head_dim, latent_dim, value_dim = 128, 512, 128
    k_len = 2048
    device = "cuda"
    dtype = torch.bfloat16
    
    # Create inputs
    q_nope = torch.randn(bsz, num_heads, q_len, qk_nope_head_dim, device=device, dtype=dtype)
    q_pe = torch.randn(bsz, num_heads, q_len, 64, device=device, dtype=dtype)
    kv_latents = torch.randn(bsz, k_len, latent_dim, device=device, dtype=dtype)
    k_pe = torch.randn(bsz, k_len, 64, device=device, dtype=dtype)
    latent_to_k = torch.randn(latent_dim, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
    latent_to_v = torch.randn(latent_dim, num_heads, value_dim, device=device, dtype=dtype)
    kv_length = torch.full((bsz,), k_len, dtype=torch.int32, device=device)
    softmax_scale = 1.0
    
    from einops import rearrange
    
    # Warm up
    for _ in range(10):
        q_nope_flat = rearrange(q_nope, "b h q d -> h (b q) d")
        q_latent = torch.bmm(q_nope_flat, rearrange(latent_to_k, "l h d -> h d l"))
    torch.cuda.synchronize()
    
    num_iters = 1000
    
    # Time pre-projection
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        q_nope_flat = rearrange(q_nope, "b h q d -> h (b q) d")
        q_latent = torch.bmm(q_nope_flat, rearrange(latent_to_k, "l h d -> h d l"))
    torch.cuda.synchronize()
    time_pre = (time.time() - start) / num_iters * 1000
    
    # Time post-projection  
    o_latent = torch.randn(num_heads, bsz, latent_dim, device=device, dtype=dtype)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        o_final = torch.bmm(o_latent, rearrange(latent_to_v, "l h d -> h l d"))
    torch.cuda.synchronize()
    time_post = (time.time() - start) / num_iters * 1000
    
    # Time attention (simulate with similar ops)
    import torch.nn.functional as F
    q_ref = torch.randn(bsz, 1, num_heads, latent_dim + 64, device=device, dtype=torch.float32)
    k_ref = torch.randn(bsz, 1, k_len, latent_dim + 64, device=device, dtype=torch.float32)
    v_ref = torch.randn(bsz, 1, k_len, latent_dim, device=device, dtype=torch.float32)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        _ = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=softmax_scale)
    torch.cuda.synchronize()
    time_attn = (time.time() - start) / num_iters * 1000
    
    print(f"\nDecode Time Breakdown (per token):")
    print(f"  Pre-projection (q_nope @ latent_to_k):  {time_pre:.4f} ms ({time_pre/(time_pre+time_attn+time_post)*100:.1f}%)")
    print(f"  Attention kernel:                       {time_attn:.4f} ms ({time_attn/(time_pre+time_attn+time_post)*100:.1f}%)")
    print(f"  Post-projection (o @ latent_to_v):      {time_post:.4f} ms ({time_post/(time_pre+time_attn+time_post)*100:.1f}%)")
    print(f"  Total:                                  {time_pre+time_attn+time_post:.4f} ms")
    print(f"\nConclusion: Projections are ~{(time_pre+time_post)/(time_pre+time_attn+time_post)*100:.1f}% of total time")

if __name__ == "__main__":
    benchmark_decode_breakdown()
