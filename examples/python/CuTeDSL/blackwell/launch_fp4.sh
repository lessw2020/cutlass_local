# Float8E4M3FN
python /home/less/cutlass_local/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py \
  --ab_dtype Float4E2M1FN --sf_dtype Float8E4M3FN --sf_vec_size 16 \
  --c_dtype BFloat16 --mma_tiler_mn 256,128 --cluster_shape_mn 2,1 \
  --mnkl 8192,8192,1024,1