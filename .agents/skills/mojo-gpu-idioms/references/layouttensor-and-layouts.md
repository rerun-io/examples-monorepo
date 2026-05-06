# LayoutTensor and Layouts

Use this note for regular tensor-shaped kernels, shared-memory tiles, and code reviews that should prefer layout-aware abstractions over manual pointer arithmetic.

## Default choice

- Prefer a layout-aware tensor when the data has a stable logical shape and the kernel benefits from natural indexing.
- Use the local vocabulary: `LayoutTensor` plus `Layout.row_major(...)` in LayoutTensor code, or `TileTensor` plus `row_major[...]()`/`TensorLayout` in TileTensor code.
- Prefer explicit layout declarations at compile time so shape and access intent stay visible.
- Prefer parameterizing reusable kernels on a layout type or layout trait when the surrounding codebase already follows that style.
- Fall back to manual pointer math only for irregular foreign buffers, niche interop constraints, or a proven hot path where the abstraction no longer matches the layout.

## Preserve local tensor vocabulary

- Some codebases use `LayoutTensor`; current Modular GPU examples often use `TileTensor` plus layout traits and helpers.
- Follow the current codebase instead of "upgrading" names during unrelated edits.
- Carry over the idiom, not the symbol spelling: explicit layout, natural indexing, shared-memory tiles, and logical tensor views.

## Core idioms

- Use `tensor[i]` for 1D work and `tensor[row, col]` for 2D work instead of recomputing flat offsets manually.
- Encode shape in the layout, not in scattered index arithmetic.
- Use simple global indexing helpers when that is enough for the kernel; only expand to manual block and thread arithmetic when the kernel needs separate local and global coordinates.
- Use `LayoutTensor[..., address_space=AddressSpace.SHARED].stack_allocation()` for shared-memory tiles instead of raw `stack_allocation[...]` when a logical tensor view helps.
- In TileTensor-style code, use `stack_allocation[dtype, address_space=AddressSpace.SHARED](row_major[...]())` and keep the tile layout visible next to the allocation.
- Keep global layouts and shared layouts separate. Shared tensors usually model per-block tiles rather than full global shapes.
- For nested or tiled layouts, verify the flattened index rank before assuming `tensor[row, col]` is valid.

## When LayoutTensor is the right abstraction

- Elementwise maps and zips over regular vectors or matrices.
- Broadcast-style kernels where natural indexing clarifies the intent.
- Shared-memory staging where each block works on a rectangular or linear tile.
- Sliding-window kernels where neighboring access should read like the math.
- Matrix kernels where tile shape matters as much as index math.

## When not to overuse it

- Do not assume `LayoutTensor` removes the need for bounds checks on global indices.
- Do not assume it synchronizes shared memory for you. Shared-memory readers and writers still need explicit `barrier()` placement.
- Do not assume `tile()` is a safety mechanism. Treat tiles as views that require valid base offsets and sizes.
- Do not mix `LayoutTensor` and `TileTensor` APIs in one refactor unless the surrounding code already bridges them explicitly.

## Review checklist

- Verify that the declared layout matches the intended logical shape.
- Verify that bounds checks protect every global access, especially at image, matrix, and halo edges.
- Verify that shared tensors use the correct `address_space`.
- Verify that local and global indices are not mixed accidentally.
- Verify that result stores use the same logical indexing scheme as the loads.
- Verify that any `.tile[...]` use cannot step outside valid memory.

## Common pitfalls

- Flattening indices manually after already declaring a 2D layout.
- Using `LayoutTensor` in shared memory but forgetting the matching `barrier()`.
- Assuming a tile view zero-pads or clips out-of-bounds accesses.
- Choosing the wrong shared tile shape and silently transposing the computation.
- Replacing clear layout-aware indexing with opaque pointer math during refactors.

## Puzzle-derived heuristics

- Early puzzles show the main payoff: `LayoutTensor` keeps multidimensional indexing readable as the kernels become more complex.
- Shared-memory puzzles show that `LayoutTensor` improves ergonomics, but not synchronization semantics.
- Sliding-window and pooling variants show that `LayoutTensor` is especially valuable when the kernel needs neighboring values, not just a single element.
