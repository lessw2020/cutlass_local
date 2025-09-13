# Ensure local repo python packages are discoverable (examples, local cutlass),
# but rely on installed nvidia-cutlass-dsl for CuTeDSL (provides compiled CuTeDSL._mlir)
export PYTHONPATH="/home/less/cutlass_local:${PYTHONPATH}"
# may have to set this:
# sudo /usr/bin/nvidia-modprobe -u -c=0

# need cutlass 'installed' for this to work
# cd /home/less/cutlass_local
# pip install -e .

# verify: python -c "import cutlass, cutlass_cppgen; print('ok')"


ncu --target-processes all --set full -- python -m examples.python.CuTeDSL.blackwell.dense_blockscaled_gemm_persistent \
      --ab_dtype Float4E2M1FN --sf_dtype Float8E4M3FN --sf_vec_size 16 \
      --c_dtype BFloat16 \
      --mma_tiler_mn 256,128 --cluster_shape_mn 2,1 \
      --mnkl 8192,8192,1024,1 \
      --warmup_iterations 1 --iterations 10 --skip_ref_check