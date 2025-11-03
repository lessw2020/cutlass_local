def test_all_implementations_at_scale():
    """
    Test all three implementations (slow_mla, torch_reference, CUTE kernel) at 128 heads
    This helps identify if differences are scale-related or kernel-specific
    """
    print("=" * 80)
    print("Testing ALL implementations at scale (128 heads, FP16)")
    print("=" * 80)
    
    # Match CUTE kernel test parameters exactly
    bsz = 1
    num_heads = 128  # Match kernel test
    q_len = 1
    k_len = 128
    qk_nope_head_dim = 32
    qk_rope_head_dim = 64
    latent_dim = 512
    value_dim = 512
    softmax_scale = 1.0 / math.sqrt(latent_dim + qk_rope_head_dim)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16  # Match kernel test
    
    print(f"\nTest Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {bsz}")
    print(f"  Num heads: {num_heads}")
    print(f"  Query length: {q_len}")
    print(f"  KV length: {k_len}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  RoPE dim: {qk_rope_head_dim}")
    print(f"  Value dim: {value_dim}")
    print(f"  Data type: {dtype}")
    
    torch.manual_seed(42)
    
    # Generate test data
    print("\nGenerating test data...")
    q_nope = torch.randn(bsz, num_heads, q_len, qk_nope_head_dim, device=device, dtype=dtype)
    q_pe = torch.randn(bsz, num_heads, q_len, qk_rope_head_dim, device=device, dtype=dtype)
    kv_latents = torch.randn(bsz, k_len, latent_dim, device=device, dtype=dtype)
    k_pe = torch.randn(bsz, k_len, qk_rope_head_dim, device=device, dtype=dtype)
    latent_to_k = torch.randn(latent_dim, num_heads, qk_nope_head_dim, device=device, dtype=dtype)
    latent_to_v = torch.randn(latent_dim, num_heads, value_dim, device=device, dtype=dtype)
    kv_length = torch.tensor([k_len - q_len] * bsz, device=device, dtype=torch.int32)
    
    # 1. Run slow_mla_attn (FULL with latent_to_v projection)
    print("\n" + "=" * 80)
    print("1. Running slow_mla_attn (reference)...")
    print("=" * 80)
    output_slow = slow_mla_attn(
        q_nope, q_pe, kv_latents, k_pe, latent_to_k, latent_to_v, 
        softmax_scale, kv_length
    )
    print(f"  Output shape: {output_slow.shape}")
    
    # 2. Run torch_reference_mla at latent level, then project
    print("\n" + "=" * 80)
    print("2. Running torch_reference_mla...")
    print("=" * 80)
    
    # Pre-compute q_latent
    q_nope_reshaped = rearrange(q_nope, "b h q d -> h (b q) d")
    q_latent = torch.bmm(q_nope_reshaped, rearrange(latent_to_k, "l h d -> h d l"))
    q_latent = rearrange(q_latent, "h (b q) l -> b h l q", b=bsz)
    
    q_rope = q_pe.permute(0, 1, 3, 2)
    c_latent = kv_latents.permute(0, 2, 1)
    c_rope = k_pe.permute(0, 2, 1)
    cache_seqs = kv_length + q_len
    
    output_ref_latent = torch_reference_mla(
        q_latent, q_rope, c_latent, c_rope, cache_seqs, softmax_scale
    )
    print(f"  Latent output shape: {output_ref_latent.shape}")
    
    output_ref = apply_latent_v_projection(output_ref_latent, latent_to_v)
    print(f"  Final output shape: {output_ref.shape}")
    
    # 3. Run CUTE kernel (if available)
    output_kernel = None
    if CUTE_AVAILABLE and MLA_KERNEL is not None:
        print("\n" + "=" * 80)
        print("3. Running CUTE DSL kernel...")
        print("=" * 80)
        
        try:
            # Prepare kernel inputs
            q_latent_proj = q_latent.squeeze(2).permute(1, 2, 0).contiguous()
            q_rope_kernel = q_pe.squeeze(2).permute(1, 2, 0).contiguous()
            c_latent_kernel = kv_latents.permute(1, 2, 0).contiguous()
            c_rope_kernel = k_pe.permute(1, 2, 0).contiguous()
            o_kernel = torch.zeros(num_heads, latent_dim, bsz, device=device, dtype=dtype)
            lse_kernel = torch.zeros(num_heads, bsz, device=device, dtype=torch.float32)
            cache_seqs_kernel = torch.full((bsz,), k_len, device=device, dtype=torch.int32)
            
            # Convert to CUTE tensors
            cute_dtype = cutlass.Float16
            acc_dtype = cutlass.Float32
            lse_dtype = cutlass.Float32
            
            q_latent_cute = from_dlpack(q_latent_proj, assumed_align=16, use_32bit_stride=True)
            q_latent_cute.element_type = cute_dtype
            q_latent_cute = q_latent_cute.mark_layout_dynamic(leading_dim=1)
            
            q_rope_cute = from_dlpack(q_rope_kernel, assumed_align=16, use_32bit_stride=True)
            q_rope_cute.element_type = cute_dtype
            q_rope_cute = q_rope_cute.mark_layout_dynamic(leading_dim=1)
            
            c_latent_cute = from_dlpack(c_latent_kernel, assumed_align=16, use_32bit_stride=True)
            c_latent_cute.element_type = cute_dtype
            c_latent_cute = c_latent_cute.mark_layout_dynamic(leading_dim=1)
            
            c_rope_cute = from_dlpack(c_rope_kernel, assumed_align=16, use_32bit_stride=True)
            c_rope_cute.element_type = cute_dtype
            c_rope_cute = c_rope_cute.mark_layout_dynamic(leading_dim=1)
            
            o_cute = from_dlpack(o_kernel, assumed_align=16, use_32bit_stride=True)
            o_cute.element_type = cute_dtype
            o_cute = o_cute.mark_layout_dynamic(leading_dim=1)
            
            lse_cute = from_dlpack(lse_kernel, assumed_align=16, use_32bit_stride=True)
            lse_cute.element_type = lse_dtype
            lse_cute = lse_cute.mark_layout_dynamic(leading_dim=0)
            
            cache_seqs_cute = from_dlpack(cache_seqs_kernel, assumed_align=16, use_32bit_stride=True)
            cache_seqs_cute = cache_seqs_cute.mark_layout_dynamic()
            
            # Create and run kernel
            mla_kernel = MLA_KERNEL(
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
            
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)
            
            compiled_kernel = cute.compile(
                mla_kernel,
                q_latent_cute, q_rope_cute, c_latent_cute, c_rope_cute,
                None, o_cute, lse_cute, None,
                cutlass.Int32(1), cache_seqs_cute, None,
                cutlass.Float32(softmax_scale), cutlass.Float32(1.0),
                stream,
            )
            
            compiled_kernel(
                q_latent_cute, q_rope_cute, c_latent_cute, c_rope_cute,
                None, o_cute, lse_cute, None,
                cutlass.Int32(1), cache_seqs_cute, None,
                cutlass.Float32(softmax_scale), cutlass.Float32(1.0),
                stream,
            )
            
            torch.cuda.synchronize()
            print("✓ Kernel execution completed")
            
            # Convert kernel output
            o_kernel_out = o_kernel.permute(2, 0, 1).unsqueeze(1)
            o_kernel_out = rearrange(o_kernel_out, "b q h l -> h (b q) l")
            output_kernel = torch.bmm(o_kernel_out, rearrange(latent_to_v, "l h d -> h l d"))
            output_kernel = rearrange(output_kernel, "h (b q) d -> b q (h d)", b=bsz)
            print(f"  Kernel output shape: {output_kernel.shape}")
            
        except Exception as e:
            print(f"✗ Kernel execution failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare all implementations
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Compare slow_mla vs torch_reference
    print("\n--- slow_mla vs torch_reference ---")
    abs_diff_ref = (output_slow - output_ref).abs()
    rel_diff_ref = abs_diff_ref / (output_ref.abs() + 1e-8)
    
    print(f"Absolute difference:")
    print(f"  Mean: {abs_diff_ref.mean().item():.8f}")
    print(f"  Median: {abs_diff_ref.median().item():.8f}")
    print(f"  Max:  {abs_diff_ref.max().item():.8f}")
    
    print(f"Relative difference:")
    print(f"  Mean: {rel_diff_ref.mean().item():.8f}")
    print(f"  Median: {rel_diff_ref.median().item():.8f}")
    print(f"  Max:  {rel_diff_ref.max().item():.8f}")
    
    is_close_ref = torch.allclose(output_slow, output_ref, rtol=1e-4, atol=1e-4)
    print(f"Close (rtol=1e-4, atol=1e-4): {'✓ PASS' if is_close_ref else '✗ FAIL'}")
    
    # Compare slow_mla vs CUTE kernel (if available)
    if output_kernel is not None:
        print("\n--- slow_mla vs CUTE kernel ---")
        abs_diff_kernel = (output_slow - output_kernel).abs()
        rel_diff_kernel = abs_diff_kernel / (output_kernel.abs() + 1e-8)
        
        print(f"Absolute difference:")
        print(f"  Mean: {abs_diff_kernel.mean().item():.8f}")
        print(f"  Median: {abs_diff_kernel.median().item():.8f}")
        print(f"  Max:  {abs_diff_kernel.max().item():.8f}")
        
        print(f"Relative difference:")
        print(f"  Mean: {rel_diff_kernel.mean().item():.8f}")
        print(f"  Median: {rel_diff_kernel.median().item():.8f}")
        print(f"  Max:  {rel_diff_kernel.max().item():.8f}")
        
        # Test at multiple tolerances
        print(f"\nCloseness tests:")
        for rtol, atol in [(1e-5, 1e-5), (1e-4, 1e-4), (1e-3, 1e-3), (1e-2, 1e-2), (5e-2, 5e-2)]:
            is_close = torch.allclose(output_slow, output_kernel, rtol=rtol, atol=atol)
            status = "✓ PASS" if is_close else "✗ FAIL"
            print(f"  rtol={rtol}, atol={atol}: {status}")
        
        # Sample comparison
        print(f"\nSample values (first 5 dims):")
        print(f"  slow_mla:      {output_slow[0, 0, :5].tolist()}")
        print(f"  torch_ref:     {output_ref[0, 0, :5].tolist()}")
        print(f"  kernel output: {output_kernel[0, 0, :5].tolist()}")
        
        # Find max error locations
        max_abs_idx = abs_diff_kernel.flatten().argmax()
        max_rel_idx = rel_diff_kernel.flatten().argmax()
        
        print(f"\nMax absolute error location:")
        print(f"  slow_mla:  {output_slow.flatten()[max_abs_idx].item():.8f}")
        print(f"  kernel:    {output_kernel.flatten()[max_abs_idx].item():.8f}")
        print(f"  diff:      {abs_diff_kernel.flatten()[max_abs_idx].item():.8f}")
        
        print(f"\nMax relative error location:")
        print(f"  slow_mla:  {output_slow.flatten()[max_rel_idx].item():.8f}")
        print(f"  kernel:    {output_kernel.flatten()[max_rel_idx].item():.8f}")
        print(f"  rel diff:  {rel_diff_kernel.flatten()[max_rel_idx].item():.8f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"slow_mla vs torch_reference: {'✓ MATCH' if is_close_ref else '✗ MISMATCH'}")
    if output_kernel is not None:
        is_close_kernel = torch.allclose(output_slow, output_kernel, rtol=5e-2, atol=5e-2)
        print(f"slow_mla vs CUTE kernel: {'✓ MATCH' if is_close_kernel else '✗ MISMATCH'}")
    print("=" * 80)


if __name__ == "__main__":
    # Run the comprehensive scale test
    print("\n" + "🔥" * 40)
    print("COMPREHENSIVE SCALE TEST (128 heads)")
    print("🔥" * 40)
    test_all_implementations_at_scale()
    
    # Keep original tests for regression checking
    print("\n\n" + "🔥" * 40)
    print("ORIGINAL TESTS (4 heads)")
    print("🔥" * 40)
    full_match = test_full_mla_match_decoding()
    latent_decoding_match = test_latent_mla_match_decoding()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"128 heads test completed (see results above)")
    print(f"Full pipeline test (4 heads): {'✓ PASS' if full_match else '✗ FAIL'}")
    print(f"Latent level test (4 heads):  {'✓ PASS' if latent_decoding_match else '✗ FAIL'}")
    print("=" * 80)