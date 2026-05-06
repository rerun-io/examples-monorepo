# Integration Patterns

Use this note when the task bridges Mojo GPU kernels into Python via MAX Graph or PyTorch-facing custom ops.

## MAX Graph custom ops

- Register the op with `@compiler.register(...)`.
- Use `execute[target: StaticString](...)` as the custom-op entry point.
- Put output tensors first in the runtime argument list, followed by inputs and `ctx: DeviceContextPtr`.
- Keep the `execute` interface explicit about tensor ranks, dtypes, and compile-time parameters.
- Use `foreach` directly for simple elementwise operations over `OutputTensor`/`InputTensor`.
- Use `ctx.get_device_context()` and `enqueue_function_experimental[...]` when the op needs explicit GPU thread layout, shared memory, atomics, or GPU-specific primitives.
- Convert graph tensors to `LayoutTensor` or `TileTensor` views before calling an underlying kernel only when that matches the kernel's algorithm.
- Package the op with `mojo package` when building a reusable custom extension.
- Pair this note with `$mojo-python-interop` for `PythonObject`, module-builder, and Python-surface details that are not specific to GPU kernel design.

## Reuse rule

- Prefer keeping the kernel logic in Mojo and changing only the integration layer.
- If the task moves from MAX Graph to another Python wrapper, preserve the kernel unless the runtime interface truly changes the memory or threading model.

## PyTorch-facing patterns

- Reuse the same kernel when possible; change the wrapper, not the math.
- Allocate output tensors explicitly on the correct device before invoking the compiled custom op.
- Treat the output tensor order and parameter dictionary as part of the integration contract.
- Load Mojo custom ops with `CustomOpLibrary(Path(...))`.
- Put the Mojo op call in a `torch.compile`-compatible wrapper when following Modular's PyTorch custom-op pattern.
- Prefer `pic.new_empty(...)`, `x.new_empty(...)`, or equivalent device-preserving output allocation.
- Keep PyTorch tensors on GPU before the call and copy back to CPU only for inspection or host-side libraries.

## Performance and correctness lessons

- Memory access patterns matter as much as the math for embedding-style lookups and similar memory-bound ops.
- Kernel fusion can remove intermediate allocations and launch overhead, but it raises the bar for correctness and backward-pass design.
- If a fused op needs gradients, review the backward path as carefully as the forward path.

## Validation guidance

- Use `pixi` to build and run integration examples.
- Treat PyTorch custom-op examples as requiring a MAX-compatible GPU unless the local environment proves otherwise.
- Prefer comparing wrapper outputs against a known CPU or NumPy reference and, where applicable, against the MAX Graph path that calls the same kernel.

## Review checklist

- Verify that compile-time parameters in Python match the Mojo execute signature.
- Verify that output tensors are allocated with the correct shape, dtype, and device.
- Verify that custom-op names match exactly across `@compiler.register(...)`, `ops.custom(name=...)`, `CustomOpLibrary`, and any parameterized access.
- Verify that the wrapper preserves the kernel's assumptions about layout and buffer ownership.
- Verify that a fused integration path has an equally rigorous backward story if training is in scope.

## Common pitfalls

- Rewriting a correct kernel when only the wrapper layer changed.
- Assuming wrapper code allocates outputs automatically.
- Forgetting to keep parameter names synchronized between Python and Mojo.
- Treating a fused forward path as complete while ignoring its backward path.
