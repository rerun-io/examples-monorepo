# Warp and Block Collectives

Use this note for reductions, scans, broadcasts, and lane-to-lane communication that should rely on Mojo's collective primitives instead of custom shared-memory choreography.

## Default scope rules

- Use `warp.*` primitives when all cooperating values live within one warp.
- Use `block.*` primitives when the operation spans an entire block.
- Keep `WARP_SIZE` symbolic. Do not hardcode `32` unless a hardware-specific path explicitly requires it.
- Reserve manual shared-memory reductions for cases where no existing primitive matches the operation or scope.

## Warp-level patterns

- Use `warp.sum()` for warp-local reductions instead of shared memory plus tree reduction.
- Use `warp.shuffle_down()` and related shuffles for neighbor access, finite differences, and stencil-like lane communication.
- Use `warp.broadcast()` when one lane should publish a value to the rest of the warp.
- Use `warp.shuffle_xor()` and `warp.prefix_sum()` for butterfly-style communication and scan patterns.

## Block-level patterns

- Use `block.sum()` when the reduction spans multiple warps in one block.
- Use `block.prefix_sum()` for block-wide scans and compaction-style write position calculation.
- Use `block.broadcast()` when one thread computes a block-wide value and the rest of the block needs it.

## Single-writer conventions

- After a warp-wide collective, lane 0 is usually the correct single writer for a warp result.
- After a block-wide collective, thread 0 is usually the correct single writer for a block result.
- Do not let every participant store the same reduced value unless the algorithm explicitly needs a broadcasted write.

## What collectives replace

- Shared-memory scratch buffers used only for reductions or scans.
- Repeated `barrier()` calls in tree reductions.
- Manual stride-halving loops for reductions.
- Manual lane-neighbor loads when the neighbor is in the same warp.

## What collectives do not replace

- Shared-memory staging that persists data across multiple algorithm phases.
- Synchronization between different thread groups running different code paths.
- Cross-block coordination.
- Bounds checks for global memory accesses before the collective.

## Review checklist

- Verify the scope: one warp, one block, or larger.
- Verify that every participating thread or lane contributes a valid value, or an explicit neutral element when out of bounds.
- Verify that the store happens from the intended single writer.
- Verify that warp-only primitives are not being used for cross-warp problems without an extra coordination step.
- Verify that any remaining `barrier()` is there for data visibility, not because the collective itself needs it.

## Common pitfalls

- Using `warp.*` when the algorithm actually spans multiple warps.
- Hardcoding a warp width instead of using `WARP_SIZE`.
- Forgetting that Mojo tensor indexing may return SIMD values and sometimes needs a scalar reduction before a collective.
- Replacing a block-wide collective with a warp-wide one during refactoring and silently changing correctness.
