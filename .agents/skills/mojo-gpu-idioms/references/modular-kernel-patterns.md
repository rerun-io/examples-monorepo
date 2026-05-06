# Modular Kernel Patterns

Use this note when a task needs examples from Modular's `max/kernels` PMPP ports, `max/examples/custom_ops`, or `max/examples/pytorch_custom_ops`.

## What the examples teach

- Start from a scalar CPU reference or simple GPU kernel, then optimize one pressure point at a time.
- Keep the launch shape tied to the logical output shape. Use 1D launches for vectors and flattened work, 2D launches for images and matrices, and explicit mapping when coalescing requires swapping row/column ownership.
- Guard every global read and write whose thread domain can exceed the problem domain.
- Prefer gather formulations over scatter formulations when both are valid and gather avoids atomics.
- Use atomics intentionally for histograms, sparse scatter, graph frontier updates, and Coulomb-style scatter. Do not use atomics to patch an avoidable ownership problem.
- Treat one incorrect PMPP example as a review rule: never place `barrier()` in divergent control flow unless every thread in the block reaches the same barrier sequence.

## Elementwise and custom-op defaults

- For a MAX custom op that maps each output element independently, use `foreach` over an `OutputTensor`/`InputTensor` closure first.
- Keep the nested closure small, `@parameter`, and `@always_inline` when the local style uses it.
- Use `idx: IndexList[rank]` with `load[width](idx)` and `store[width](idx, value)` at ManagedTensorSlice boundaries.
- Pass `target: StaticString` through `foreach[..., target=target]` so CPU and GPU code stay unified.
- Escalate to `ctx.get_device_context().enqueue_function_experimental[...]` only when direct control over block size, shared memory, or GPU-only primitives is required.

## Thread kernels

- Prefer `global_idx.x`/`.y` for direct one-thread-per-output kernels.
- Use manual `block_dim * block_idx + thread_idx` math when the kernel needs separate block-local coordinates for shared memory, coalesced loads, register tiling, or per-block staging.
- Compute launch sizes with ceiling division so remainder elements are covered, then guard in the kernel.
- Keep constants like block size as `comptime` values when fixed by the algorithm.
- Use `DeviceContext` on standalone Mojo examples and `DeviceContextPtr` inside MAX custom-op `execute` methods.

## Memory locality ladder

Use this optimization order unless profiling or the problem shape says otherwise:

1. Coalesce global loads and stores by mapping adjacent threads to adjacent addresses.
2. Add shared-memory tiling when input values are reused by multiple threads.
3. Add register tiling or thread coarsening when each thread can reuse loaded data across multiple outputs.
4. Add vectorized loads/stores when layout and alignment make chunked access natural.
5. Add async copy or double buffering only after the synchronous tiled version is correct.
6. Add Tensor Cores only behind a target and dtype gate.

## Shared memory and tiling

- Use a tensor view for shared memory when the tile has a stable logical shape: `stack_allocation(...)(row_major[...])` in TileTensor-style code or `LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()` in LayoutTensor-style code.
- Keep global tiles, shared tiles, and register tiles distinct in naming and type.
- Use explicit thread layouts for cooperative tile loads. Match the thread layout to the desired coalesced memory access.
- Use halos explicitly for convolution and stencil kernels. Interior elements may come from shared memory while halo elements fall back to guarded global loads.
- Treat `.tile[...]` as a view into the underlying tensor, not as a copy, padding operation, or bounds check.

## Reductions, scans, and partitioning

- Prefer warp and block collectives when the scope matches; they replace many manual shared-memory reduction trees.
- Keep a single-writer convention after reductions: lane 0 for warp results and thread 0 for block results unless the algorithm needs all lanes to write.
- For stream compaction and radix-like partitioning, compute positions with scan or warp vote patterns before scattering.
- For multi-block scan or reduction, split the work into phases and launch separate kernels or use host orchestration. A block barrier is not a grid barrier.

## Irregular workloads

- Sparse matrix, graph traversal, merge, sort, and dynamic-programming examples use raw pointers and explicit indexing more often because the data is not a dense rectangular tensor.
- Choose data layout before kernel structure: CSR/CSC/COO/ELL and push/pull/frontier BFS variants change both memory coalescing and atomic pressure.
- Prefer frontier or worklist approaches when they reduce inactive thread work enough to justify queue management.
- For wavefront dynamic programming, make the dependency direction explicit and use kernel phases or diagonal progression instead of assuming global synchronization inside one launch.

## Matrix, attention, and fused kernels

- For matmul, the Modular custom-op example progresses from naive, to coalesced, to shared-memory tiled, to register-tiled, to vectorized, then Tensor Core.
- Keep algorithm choice as a compile-time parameter in custom ops when the Python wrapper selects variants.
- For attention and softmax, preserve numerical-stability state such as running max and normalization sums. Review these values as part of correctness, not only performance.
- Avoid materializing large intermediates when a tiled fused algorithm can stream through them, but only after the unfused or simpler reference is tested.

## Review checklist

- Verify the abstraction level: `foreach`, tensor tile, explicit thread kernel, raw pointer, or hardware-specific primitive.
- Verify every global access has a bounds story.
- Verify all block threads reach the same barrier sequence.
- Verify atomics are required by write ownership, not by convenience.
- Verify launch dimensions cover the output domain and use the same coordinate convention as the kernel.
- Verify custom-op signatures and Python wrappers agree on names, parameter dictionaries, output order, dtype, shape, and device.
