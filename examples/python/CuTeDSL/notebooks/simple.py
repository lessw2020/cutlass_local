import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello, World - I am here in the GPU!!!")

@cute.jit
def hello_world():
    cute.printf("Hello, World - from the CPU side!")

    kernel().launch(
        grid = (1,1,1),  # single thread block
        block = (32, 1, 1),  # one warp (32 threads) per thread block   

    )

cutlass.cuda.initialize_cuda_context()
#print("Running JIT hello world...")
#hello_world()

print("Compiling...")
hello_world_compiled = cute.compile(hello_world)


print("Running compiled version...")
hello_world_compiled()
'''.launch(
    grid = (1,1,1),  # single thread block
    block = (32, 1, 1),  # one warp (32 threads) per thread block   
)
'''


# Dump PTX/CUBIN files while compiling
from cutlass.cute import KeepPTX, KeepCUBIN

print("Compiling with PTX/CUBIN dumped...")
# Alternatively, compile with string based options like
# cute.compile(hello_world, options="--keep-ptx --keep-cubin") would also work.
hello_world_compiled_ptx_on = cute.compile[KeepPTX, KeepCUBIN](hello_world)

# Run the pre-compiled version
print("Running compiled version with PTX/CUBIN dumped...")
# hello_world_compiled()
hello_world_compiled_ptx_on()
print(hello_world_compiled_ptx_on.__ptx__)