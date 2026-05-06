# Functional Patterns

Use this note when choosing between MAX `foreach`, raw kernels, tiling, `vectorize`, and manual SIMD control.

## Decision ladder

- Start with `foreach` in MAX custom ops or `elementwise` in codebases that already use that abstraction for regular independent map or zip operations.
- Move to tiling when memory locality or amortized per-thread work matters.
- Use `vectorize` inside a tile when SIMD control should stay high-level and bounds-safe.
- Use manual vectorization only when alignment, exact memory layout, or benchmarking proves the extra control is worth the added complexity.
- Drop to a raw kernel only when the workload is irregular enough that the functional patterns stop fitting cleanly.

## `foreach` and `elementwise`

- Use `foreach`/`elementwise` for straight-line data-parallel work.
- Treat the incoming index as the start of a SIMD chunk, not automatically as a single scalar element.
- Keep the nested callback small, `@parameter`, and capture only the tensors and constants it needs.
- Prefer this for clear, portable code before inventing manual block or grid math.
- In custom ops, pass `target` through the abstraction so the same op can compile for CPU or GPU.

## Tiling

- Tile when a thread should process a chunk of work instead of a single element.
- Use tiles to improve locality and to make per-thread work coarse enough to justify shared memory or SIMD loops.
- Keep the tile size tied to the memory pattern and problem shape, not just to a convenient constant.

## `vectorize`

- Use `vectorize` when the loop is regular and you want automatic chunking with cleaner bounds handling.
- Prefer it to manual SIMD loops when the performance difference is unknown or likely small.
- Keep the scalar tail behavior explicit if the vector width does not divide the work perfectly.

## Manual SIMD

- Use `aligned_load` and vector stores only when alignment and layout are known.
- Keep the chunk size explicit: tile size times SIMD width.
- Make the tradeoff clear in reviews: more control, less safety, harder debugging.
- Avoid this style for first-pass implementations unless the task is already framed as low-level SIMD work.

## Benchmarking guidance

- Benchmark only after correctness is stable.
- Treat small benchmark deltas cautiously; the Modular examples show that the simplest abstraction is often already competitive for elementwise work.
- Use benchmarking to justify moving from `foreach`/`elementwise` to a lower-level pattern, not to rationalize complexity in advance.

## Review checklist

- Verify that the chosen abstraction matches the work granularity.
- Verify that SIMD width and tile size assumptions are valid for the dtype and target.
- Verify that loads and stores align with the chunk semantics.
- Verify that the code handles partial tiles or non-divisible tails when the size is not a perfect multiple.
- Verify that the author is not using a manual thread kernel where `foreach` would be clearer and equally correct.

## Common pitfalls

- Treating `indices[0]` from `elementwise` as a scalar index instead of a SIMD chunk start.
- Reimplementing a MAX custom-op elementwise map with hand-written block math even though `foreach` matches the operation.
- Mixing tiled local indexing with global indexing in the same loop.
- Choosing manual SIMD because it feels lower level, not because the workload needs it.
- Forgetting that more vectorization control increases debugging cost and edge-case risk.
