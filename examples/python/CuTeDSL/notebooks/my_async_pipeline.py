import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# bank conflict: multiple threads access the same shared memory bank
# where sm is divided into 32 banks

@cute.kernel
def synced_producer_consumer(SharedStorage: cutlass.Constexpr, res: cute.Tensor):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)

    staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(1))
    staging_smem.fill(0)
    cute.arch.sync_threads()

    for i in cutlass.range(cute.size(res)):
        if warp_idx ==0:
            staging_smem[0] = i * 1.0
        # mark enter of critical region
        cute.arch.sync_threads()
        if warp_idx == 1:
            res[i] = staging_smem[0]
        # mark exit of critical region
        cute.arch.sync_threads()

@cute.jit
def run_synced_producer_consumer(res: cute.Tensor):
    @cute.struct
    class SharedStorage:
        staging_buffer: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 1], 1024]   
    synced_producer_consumer(SharedStorage, res).launch(
        grid=(1, 1, 1), block=(64, 1, 1), smem=SharedStorage.size_in_bytes()
    )

res = torch.zeros((8,), device="cuda")
run_synced_producer_consumer(from_dlpack(res))
torch.cuda.synchronize()
print(res)

@cute.kernel
def async_pipeline_kernel(res: cute.Tensor):
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    @cute.struct
    class SharedStorage:
        tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        staging_buffer: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 1], 1024]
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)

    # warp 0
    producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 32)
    # warp 1
    consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 32)

    pipeline = cutlass.pipeline.PipelineAsync.create(
        num_stages=1,
        producer_group=producer_group,
        consumer_group=consumer_group,
        barrier_storage=storage.tma_mbar_ptr.data_ptr(),
    )

    staging_smem = storage.staging_buffer.get_tensor(cute.make_layout(1))
    staging_smem.fill(0)
    cute.arch.sync_threads()

    producer, consumer = pipeline.make_participants()
    # producer warp
    if warp_idx == 0:
        for i in cutlass.range(cute.size(res)):
            handle = producer.acquire_and_advance()
            staging_smem[handle.index] = 1.0 * i
            handle.commit()
        producer.tail()
    # consumer warp
    if warp_idx == 1:
        for i in cutlass.range(cute.size(res)):
            handle = consumer.wait_and_advance()
            res[i] = staging_smem[handle.index]
            handle.release()
@cute.jit
def async_pipeline(res: cute.Tensor):
    # Launch kernel with two warps: producer and consumer
    async_pipeline_kernel(res).launch(grid=(1, 1, 1), block=(64, 1, 1))


res = torch.zeros((8,), device="cuda")
async_pipeline(from_dlpack(res))
print(res)

# ---- ring buffer ---------

@cute.kernel
def my_async_pipeline_staged_kernel(SharedStorage: cutlass.Constexpr, res: cute.Tensor, staging: cute.Tensor):
    stages = cute.size(staging)
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)

    # warp 0
    producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 32)
    # warp 1
    consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, 32)

    pipeline = cutlass.pipeline.PipelineAsync.create(
        num_stages = stages,
        producer_group = producer_group,
        consumer_group = consumer_group,
        barrier_storage = storage.tma_mbar_ptr.data_ptr(),
    )
    staging_smem = storage.staging_buffer.get_tensor(staging.layout)
    staging_smem.fill(0)
    cute.arch.sync_threads()

    producer, consumer = pipeline.make_participants()

    # producer warp
    if warp_idx == 0:
        for i in cutlass.range(cute.size(res)):
            handle = producer.acquire_and_advance()
            staging_smem[handle.index] = 1.0 * i
            handle.commit()
        # prevents CTA0 from retiring until it receives all expected arrives.
        producer.tail()
       
    # consumer warp
    if warp_idx == 1:
        for i in cutlass.range(cute.size(res)):
            handle = consumer.wait_and_advance()
            res[i] = staging_smem[handle.index]
            handle.release()

    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        staging.store(staging_smem.load())
'''
@cute.kernel
def async_pipeline_staged_kernel(
    SharedStorage: cutlass.Constexpr, res: cute.Tensor, staging: cute.Tensor
):
    stages = cute.size(staging)

    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage, 64)

    # Warp 0
    producer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread, 32
    )
    # Warp 1
    consumer_group = cutlass.pipeline.CooperativeGroup(
        cutlass.pipeline.Agent.Thread, 32
    )

    pipeline = cutlass.pipeline.PipelineAsync.create(
        num_stages=stages,
        producer_group=producer_group,
        consumer_group=consumer_group,
        barrier_storage=storage.tma_mbar_ptr.data_ptr(),
    )

    staging_smem = storage.staging_buffer.get_tensor(staging.layout)
    staging_smem.fill(0)
    cute.arch.sync_threads()

    producer, consumer = pipeline.make_participants()

    # Producer warp
    if warp_idx == 0:
        for i in cutlass.range(cute.size(res)):
            handle = producer.acquire_and_advance()
            staging_smem[handle.index] = 1.0 * i
            handle.commit()  # or producer.commit(handle)

        # prevents CTA0 from retiring until it receives all expected arrives.
        producer.tail()

    # Consumer warp
    if warp_idx == 1:
        for i in cutlass.range(cute.size(res)):
            handle = consumer.wait_and_advance()
            res[i] = staging_smem[handle.index]
            handle.release()  # or consumer.release(handle)

    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        staging.store(staging_smem.load())
'''

@cute.jit
def async_pipeline_staged(res: cute.Tensor, staging: cute.Tensor):
    stages = cute.size(staging)

    @cute.struct
    class SharedStorage:
        tma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, stages * 2]
        staging_buffer: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, stages], 1024
        ]

    my_async_pipeline_staged_kernel(SharedStorage, res, staging).launch(
        grid=(1, 1, 1), block=(64, 1, 1), smem=SharedStorage.size_in_bytes()
    )


res = torch.zeros((8,), device="cuda")
staging = torch.zeros((5,), device="cuda")
async_pipeline_staged(from_dlpack(res), from_dlpack(staging))
torch.cuda.synchronize()
print(f"First Ring Buffer Result: {res}")
print(f"First Ring Buffer Staging: {staging}")