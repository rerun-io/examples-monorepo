---
name: mojo-gpu-idioms
description: Write, refactor, explain, or review idiomatic Mojo GPU kernels and MAX custom ops with emphasis on TileTensor, LayoutTensor, layouts, ManagedTensorSlice/foreach, explicit thread kernels, warp and block collectives, async memory, PyTorch custom-op integration, and hardware-gated features. Use when Codex needs to choose the right Mojo GPU abstraction, turn low-level kernels into clearer and safer code, diagnose GPU kernel correctness or performance issues, or keep Mojo/MAX validation on a pixi-based toolchain.
---

# Mojo GPU Idioms

## Overview

Use this skill to write clear, correct, idiomatic Mojo GPU code. Favor Mojo's layout-aware, tensor, and collective abstractions over manual pointer arithmetic and ad hoc synchronization unless the lower-level path is clearly necessary.

## Companion Skills

- If installed, use `$mojo-syntax` alongside this skill for current Mojo syntax, ownership conventions, and `comptime` rules.
- If installed, use `$mojo-gpu-fundamentals` alongside this skill for the base host/kernel execution model, GPU imports, and launch structure.
- If installed, use `$mojo-python-interop` when the GPU code crosses the Python boundary through MAX Graph, PyTorch, Python objects, or extension modules.
- If installed, use `$new-modular-project` when the task is setting up a fresh Modular, Mojo, or MAX environment rather than writing kernels.

## Core Defaults

- Do not write CUDA syntax in Mojo. No `__global__`, `__device__`, `<<<>>>`, or `__shared__`.
- Kernels are plain `def` functions. Host-side launches use `ctx.enqueue_function[...]`.
- Preserve the tensor and layout API already used by the codebase. If the project uses `LayoutTensor`, keep it. If it uses `TileTensor`, `TensorLayout`, `row_major[...]()`, and `ManagedTensorSlice`, keep that. Do not rename APIs during unrelated edits.
- Prefer explicit layouts such as `Layout.row_major(...)` or `row_major[...]()` over implicit indexing assumptions.
- Prefer natural indexing like `tensor[i]` and `tensor[row, col]` over hand-written offset math.
- Prefer simple indexing helpers like `global_idx.x` when they express the work clearly; drop to manual `block_idx`, `block_dim`, and `thread_idx` arithmetic only when the kernel actually needs it.
- In MAX custom ops, prefer `foreach` for simple elementwise work and direct thread kernels only when the operation needs explicit block layout, shared memory, atomics, or GPU-specific data movement.
- Prefer tiling, `vectorize`, and tensor fragments for regular data-parallel work before dropping to raw pointer arithmetic.
- Prefer `warp.*` and `block.*` collectives over custom shared-memory reductions, scans, and broadcasts when the primitive matches the scope.
- Prefer compile-time layouts and dimensions when the kernel shape is fixed; use runtime-shaped layouts only when the shape is genuinely dynamic.
- Keep Tensor Cores, cluster primitives, vendor-specific debugging, and profiler-driven tuning behind explicit hardware gates.
- Validate with `pixi` and Modular's MAX and Mojo packages. Do not switch to `uv` or `pip`.

## Write Idiomatic Code

### Choose the highest useful abstraction

- Start from the clearest correct formulation.
- Use `LayoutTensor` when the code manipulates logical tensor coordinates.
- Use `TileTensor` when the surrounding codebase follows the current Modular MAX kernel examples or tile-tensor manual.
- Use `ManagedTensorSlice`, `InputTensor`, and `OutputTensor` at MAX custom-op boundaries, then convert or view data in the kernel body only when that improves the algorithm.
- Use raw pointer-style indexing only when the abstraction no longer matches the memory shape or when a proven hot path needs tighter control.
- Treat readability as part of correctness for GPU code. If the indexing obscures the memory pattern, the code is not idiomatic yet.

### Make memory shape explicit

- Declare layouts at compile time.
- Parameterize reusable kernels on layout type or layout trait when the codebase does so.
- Keep global layouts, shared-memory tile layouts, and output layouts visibly separate.
- Make shared-memory allocations read like the logical structure of the data, not like a bag of scalars.
- Treat `.tile[...]` as a view helper, not as a bounds-checking or padding mechanism.

### Use collectives instead of rebuilding them

- Use `warp.sum()` for warp-local reductions.
- Use `warp.shuffle_*` for lane-to-lane neighbor communication.
- Use `warp.prefix_sum()` for single-warp scans.
- Use `block.sum()`, `block.prefix_sum()`, and `block.broadcast()` when the operation spans a full block.
- Keep manual shared-memory tree reductions only when no existing primitive matches the required scope or behavior.

### Introduce async memory only after the synchronous version is right

- First make the tiled kernel correct.
- Then introduce `copy_dram_to_sram_async` only when the dataflow is regular enough to overlap transfer with useful work.
- Call `async_copy_wait_all()` before reading async-populated shared memory.
- Use `barrier()` when threads in the block must observe each other's writes.
- Do not confuse async-copy completion with thread synchronization.

### Keep interop layers thin

- Reuse the kernel when moving between MAX Graph and PyTorch-facing wrappers.
- Change the wrapper contract, not the math, unless the runtime truly requires a different kernel structure.
- Make output allocation, device placement, and parameter passing explicit in the wrapper layer.
- Keep `@compiler.register(...)`, `execute[target: StaticString](...)`, output-first arguments, and `DeviceContextPtr` signatures aligned with the MAX examples.
- In PyTorch wrappers, load Mojo operations through `CustomOpLibrary`, allocate outputs on the input tensor's device, and keep the Mojo op call inside the `torch.compile`-compatible function when that is the integration pattern.
- Defer Python-object conversion details, module-builder mechanics, and Python API surface work to `$mojo-python-interop`.

## Review Kernels Systematically

- Check every global access for bounds safety.
- Check that the chosen layout matches the intended logical shape.
- Check that shared-memory allocations use the correct `address_space`.
- Check that `tile()` usage cannot step beyond valid memory.
- Check that collective scope matches the algorithm scope.
- Check that single-writer stores come from lane 0, thread 0, or another intentionally chosen writer.
- Check that `barrier()` is present because data dependency requires it, not because "shared memory is involved."
- Check that async-copy code waits for transfers and synchronizes readers correctly.
- Check that hardware-specific features are presented as optional or gated, not as the baseline solution.
- Check that docs-derived snippets are updated to the codebase's current Mojo syntax before landing them.

## Escalation Rules

- Escalate from `foreach` or tensor views to explicit thread kernels only when the abstraction is the wrong fit.
- Escalate from `LayoutTensor`/`TileTensor` to lower-level indexing only when the memory shape or interop boundary requires it.
- Escalate from `foreach`/`elementwise` to tiling or manual SIMD only when locality, chunking, or benchmarking justifies it.
- Escalate from block or warp collectives to custom synchronization only when the primitive cannot express the algorithm.
- Escalate to profiler-driven tuning, occupancy work, bank-conflict analysis, Tensor Cores, or cluster programming only after the portable version is already correct.

## Validation

- Use `pixi` for environment setup and execution.
- Prefer repo-local `pixi` tasks when they exist.
- If no project exists yet, create a scratch environment with Modular's nightly channel and install `modular`.
- On paths containing spaces, `pixi run max ...` may be brittle; if needed, retry from a space-free path or invoke the resolved `.pixi/envs/.../bin/max` entrypoint directly.

## References

- Read [references/layouttensor-and-layouts.md](./references/layouttensor-and-layouts.md) for layout-aware indexing and shared-memory tensor patterns.
- Read [references/functional-patterns.md](./references/functional-patterns.md) for `elementwise`, tiling, `vectorize`, and manual SIMD tradeoffs.
- Read [references/warp-and-block-collectives.md](./references/warp-and-block-collectives.md) for reduction, scan, broadcast, and shuffle patterns.
- Read [references/async-memory-and-synchronization.md](./references/async-memory-and-synchronization.md) for async copy, barriers, and staged kernels.
- Read [references/integration-patterns.md](./references/integration-patterns.md) for MAX Graph and PyTorch wrapper design.
- Read [references/modular-kernel-patterns.md](./references/modular-kernel-patterns.md) for patterns mined from Modular's PMPP examples, custom-op kernels, and PyTorch custom-op examples.
- Read [references/hardware-gated-features.md](./references/hardware-gated-features.md) before suggesting vendor-specific tools or advanced hardware paths.
- Read [references/source-map.md](./references/source-map.md) only for provenance.
