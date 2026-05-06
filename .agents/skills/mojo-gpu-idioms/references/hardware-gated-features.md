# Hardware-Gated Features

Use this note before recommending platform-specific debugging, profiling, Tensor Core, cluster, or advanced synchronization features.

## Portability first

- Default to portable Mojo GPU patterns first.
- Add hardware-specific paths only when the target platform supports them and the benefit justifies the added complexity.
- State the platform gate explicitly in code reviews and explanations.

## Current puzzle-derived support matrix

- Apple-compatible chapters in the current puzzle repo: `p01`-`p08`, `p11`-`p19`, `p23`-`p28`.
- Source-review only on Apple: `p09`-`p10`, `p20`-`p22`, `p29`-`p34`.
- NVIDIA-only chapters in practice: debugger and sanitizer workflows, profiling, occupancy investigation, bank-conflict measurement, Tensor Cores, and cluster programming.

## Debugging and profiling

- Treat `compute-sanitizer`, NSight Systems, and NSight Compute as NVIDIA-specific workflows.
- Use profiler output to justify optimization work; do not optimize by guesswork.
- High cache hit rates are not automatically good. The performance puzzles explicitly frame profiler metrics as evidence that still requires interpretation.
- Occupancy is a tool, not a goal. Sufficient occupancy can be enough for latency hiding; more is not always better.
- Shared-memory bank conflicts are a real performance bottleneck when the kernel is otherwise well-structured.

## Async copy and advanced memory ops

- Async copy is conceptually portable in the puzzle material, but the docs still frame support in terms of modern GPU capabilities.
- If compilation or runtime support is unclear, keep the synchronous baseline path available and describe the feature gate.

## Tensor Cores

- Treat Tensor Core work as an advanced, hardware-shaped optimization path.
- Keep the baseline tiled kernel available and correct before introducing Tensor Core fragments.
- Verify fragment sizes, warp tiling, and dtype assumptions against the actual target.
- Do not present Tensor Cores as the default solution for matrix multiplication.

## Cluster programming

- Treat cluster primitives as the least portable path in this skill.
- Use them only when the target is clearly in the supported NVIDIA SM90+ class and the task truly requires inter-block coordination inside one launch.
- Make the synchronization hierarchy explicit: intra-block barriers are not cluster synchronization.

## Review checklist

- Verify that the requested feature is supported on the current hardware and toolchain.
- Verify that the baseline portable version exists or could exist.
- Verify that the explanation names the gate directly instead of burying it.
- Verify that profiling or hardware-specific complexity is warranted by measured or stated requirements.

## Common pitfalls

- Suggesting profiler tools on unsupported platforms.
- Recommending Tensor Cores or cluster primitives for baseline portable code.
- Treating occupancy, bank conflicts, or cache metrics as universal truths rather than target-specific evidence.
- Assuming the most advanced hardware path is automatically the most maintainable choice.
