# Async Memory and Synchronization

Use this note for tiled shared-memory kernels, async copy, producer-consumer stages, and barrier placement reviews.

## Baseline rule

- Start with a correct synchronous tiled kernel first.
- Introduce async copy only when the kernel is plausibly memory-bound and the dataflow is clear enough to overlap transfer with useful work.

## Async copy pattern

Use this sequence when staging global data into shared memory:

1. Launch `copy_dram_to_sram_async[...]` for the shared tile.
2. Perform useful independent work while the transfer progresses.
3. Call `async_copy_wait_all()` before reading the shared tile.
4. Call `barrier()` when all threads in the block must observe the completed transfer before continuing.

Treat `async_copy_wait_all()` and `barrier()` as different tools:

- `async_copy_wait_all()` waits for the transfer machinery.
- `barrier()` synchronizes threads within the block.

Do not assume one replaces the other.

## Tile and halo rules

- Model halo regions explicitly for stencil and convolution code.
- Verify that the tile plus halo stays within valid bounds, or add the required guards and padding logic.
- Treat `tile()` as a view, not as bounds-checking or zero-padding behavior.

## Barrier rules

- `barrier()` only synchronizes threads within one block.
- Use `barrier()` after shared-memory writes when later reads depend on data from other threads.
- Do not add a barrier only because the code "uses shared memory"; add it because another thread's writes must be visible before later reads.
- For multi-phase algorithms that span multiple blocks, rely on kernel sequencing or host orchestration, not block barriers.

## Producer-consumer pipelines

- Use barriers between stages when different thread groups in the same block execute different algorithms and consume one another's outputs.
- Keep each stage's ownership and output buffer clear.
- Place barriers at stage boundaries, not arbitrarily inside the stages.

## Review checklist

- Verify that async copy has a synchronous read barrier: wait first, then barrier if threads must converge.
- Verify that the thread layout used for async copy matches the desired coalesced access pattern.
- Verify that halo indexing cannot read outside the underlying buffer.
- Verify that each stage in a staged pipeline has a well-defined producer-consumer handoff.
- Verify that any inter-block algorithm uses kernel phases or host ordering instead of impossible global barrier assumptions.

## Common pitfalls

- Calling `barrier()` without `async_copy_wait_all()` and assuming the transfer is complete.
- Calling `async_copy_wait_all()` and then letting only some threads proceed to shared reads that others still have not synchronized with.
- Using a tile view that silently points beyond valid data.
- Applying warp or block collectives where the algorithm actually needs staged shared-memory communication.
