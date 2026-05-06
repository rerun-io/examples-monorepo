# Source Map

This file maps each concept note to the source material that informed it. It is maintenance metadata, not a required reading path.

## Research sources

- Modular repo: `max/kernels/examples/pmpp` on `main`, 95 Mojo example files across chapters 2-21.
- Modular repo: `max/examples/custom_ops` and `max/examples/pytorch_custom_ops`.
- Mojo manual: layouts, LayoutTensor, GPU fundamentals, pointers, and unsafe pointers.
- MAX docs: custom ops overview, build custom ops, matmul custom-op optimization, and PyTorch custom kernels.
- Modular skills repo: `mojo-syntax`, `mojo-gpu-fundamentals`, `mojo-python-interop`, and `new-modular-project`.

## `layouttensor-and-layouts.md`

- `p01` preview of `LayoutTensor`
- `p04` introduction and 2D `LayoutTensor`
- `p05` broadcast with `LayoutTensor`
- `p07` 2D blocks with `LayoutTensor`
- `p08` shared memory with `LayoutTensor` and `AddressSpace.SHARED`
- `p11` pooling with shared `LayoutTensor`
- `p12` dot product with `LayoutTensor`
- `p14` prefix sum with `LayoutTensor`
- `p16` tiled matmul with layout-aware shared tiles

## `functional-patterns.md`

- `p23` overview, `elementwise`, `tile`, `vectorize`, threading vs SIMD, benchmarking

## `warp-and-block-collectives.md`

- `p24` `warp.sum()` and SIMT scope
- `p25` `warp.shuffle_down()` and `warp.broadcast()`
- `p26` `warp.shuffle_xor()` and `warp.prefix_sum()`
- `p27` `block.sum()`, `block.prefix_sum()`, and `block.broadcast()`

## `async-memory-and-synchronization.md`

- `p08` shared-memory synchronization basics
- `p12` barrier-heavy reductions
- `p14` multi-phase scan and host-level sequencing
- `p16` tiled matmul and async-copy baseline
- `p28` `copy_dram_to_sram_async`, `async_copy_wait_all`, halos, and overlap
- `p29` staged producer-consumer synchronization inside a block

## `integration-patterns.md`

- `p17` MAX Graph custom ops and `mojo package`
- `p18` custom op packaging and test shape discipline
- `p19` larger MAX Graph custom op structure
- `p20` reuse of the same kernel through a PyTorch wrapper
- `p21` memory-coalescing lessons for integration-facing kernels
- `p22` fusion and backward-pass design
- MAX docs: `custom-ops.mdx`, `build-custom-ops.mdx`, `custom-ops-matmul.mdx`, and `custom-kernels-pytorch.mdx`
- Modular examples: `max/examples/custom_ops` and `max/examples/pytorch_custom_ops`

## `modular-kernel-patterns.md`

- PMPP chapters 2-5: host/device memory, launch shape, multidimensional grids, barrier hazards, shared-memory matmul
- PMPP chapters 7-9: convolution, stencil, histogram, halo handling, atomics, privatization
- PMPP chapters 10-12: reductions, scans, stream compaction, warp/block communication
- PMPP chapters 13-18: merge/sort, dynamic programming, sparse formats, graph traversal
- PMPP chapters 19-21: convolutional layers, softmax/attention, Direct Coulomb Summation
- Custom-op examples: add-one, vector addition, histogram, top-k, matrix multiplication, fused attention, image pipeline
- PyTorch custom-op examples: addition, grayscale, graph op, Whisper fused attention

## `hardware-gated-features.md`

- `howto.md` support matrix
- `p09` and `p10` NVIDIA debugging and sanitizer workflows
- `p28` modern-GPU async-copy caveats
- `p29` advanced synchronization design
- `p30` profiler-driven optimization
- `p31` occupancy heuristics
- `p32` bank-conflict analysis
- `p33` Tensor Cores
- `p34` cluster coordination
