"""CUDA-graph replay of tensor functions, one graph per input signature.

The batched tracker step (``forward_batched``) launches ~2,000 small kernels per step; on a fast GPU the Python
dispatch costs more than the kernels. :class:`GraphCache` runs a function eagerly the first time it sees an input
signature (shapes, strides, dtypes, autocast state and a caller key for the non-tensor arguments), captures it into
a CUDA graph the second time, and replays the graph afterwards. Inputs are copied into static buffers that keep the
caller's strides (so every kernel sees the layout the eager call sees and the results are bit-identical); outputs
are cloned, because the graphs share one memory pool and the next replay may overwrite them.

A captured function must be free of host syncs and host-to-device copies: build index tensors and prompts outside.
Autocast must run with ``cache_enabled=False`` (a cached weight cast is freed when the autocast region exits, and a
graph that read it would read freed memory).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field

import torch
from torch.utils import _pytree as pytree


@dataclass(slots=True)
class _Captured:
    graph: torch.cuda.CUDAGraph
    inputs: list[torch.Tensor]
    outputs: object


@dataclass
class GraphCache:
    """Per-signature CUDA graphs: eager on the first call of a signature, captured on the second, replayed after.

    Memory stays that of the largest signature, not the sum over signatures: the static inputs and outputs of all
    graphs are views into shared buffers (one per argument position and dtype, grown by doubling), and the graphs share
    one pool for their temporaries. Sharing is safe because the graphs replay one at a time on one stream, inputs are
    copied in right before a replay and outputs cloned right after it.
    """

    capture_after: int = 1
    """Eager calls of a signature before it is captured (a capture costs about two eager calls)."""
    max_graphs: int = 4096
    """Signatures beyond this many captured graphs stay eager."""
    captures: int = 0
    replays: int = 0
    eager: int = 0
    _graphs: dict[Hashable, _Captured] = field(default_factory=dict)
    _seen: dict[Hashable, tuple[int, object]] = field(default_factory=dict)
    """Per signature not captured yet: eager calls so far and the outputs of the last one on the meta device (layouts)."""
    _buffers: dict[tuple[str, int, torch.dtype], torch.Tensor] = field(default_factory=dict)
    _pool: tuple[int, int] | None = None
    _stream: torch.cuda.Stream | None = None

    def __call__(self, key: Hashable, fn: Callable[..., object], inputs: Sequence[torch.Tensor]) -> object:
        """``fn(*inputs)``, from a graph when this signature was seen before.

        Args:
            key: Everything besides the input tensors that ``fn``'s kernels depend on (its non-tensor arguments).
            fn: A function of tensors only, returning a pytree of tensors; it must not sync with the host.
            inputs: The tensors, copied into the graph's static buffers on replay.
        """
        signature = (
            key,
            tuple((tuple(t.shape), t.stride(), t.dtype, t.device) for t in inputs),
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
        )
        captured: _Captured | None = self._graphs.get(signature)
        if captured is None:
            calls, layouts = self._seen.get(signature, (0, None))
            if calls < self.capture_after or len(self._graphs) >= self.max_graphs:
                outputs = fn(*inputs)
                self._seen[signature] = (calls + 1, pytree.tree_map_only(torch.Tensor, lambda t: torch.empty_like(t, device="meta"), outputs))
                self.eager += 1
                return outputs
            captured = self._graphs[signature] = self._capture(fn, inputs, layouts)
            del self._seen[signature]
        for static, tensor in zip(captured.inputs, inputs, strict=True):
            static.copy_(tensor)
        captured.graph.replay()
        self.replays += 1
        return pytree.tree_map_only(torch.Tensor, torch.clone, captured.outputs)

    def _view(self, role: str, position: int, like: torch.Tensor, device: torch.device) -> torch.Tensor:
        """A view shaped like ``like`` (its strides when dense, as ``empty_like``) into the shared buffer of
        (``role``, ``position``, dtype); a buffer too small is replaced, the graphs captured on it keep it alive."""
        stride = like.stride() if like.is_meta else torch.empty_like(like, device="meta").stride()
        span: int = 1 + sum((size - 1) * step for size, step in zip(like.shape, stride, strict=True)) if like.numel() else 0
        slot = (role, position, like.dtype)
        buffer: torch.Tensor | None = self._buffers.get(slot)
        if buffer is None or buffer.numel() < span:
            size: int = max(span, 2 * (0 if buffer is None else buffer.numel()))
            buffer = self._buffers[slot] = torch.empty(size, dtype=like.dtype, device=device)
        return buffer.as_strided(like.shape, stride, 0)

    def _capture(self, fn: Callable[..., object], inputs: Sequence[torch.Tensor], layouts: object) -> _Captured:
        assert not torch.is_autocast_enabled("cuda") or not torch.is_autocast_cache_enabled(), "autocast cache_enabled=True"
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._stream = torch.cuda.Stream()
        assert self._stream is not None
        device: torch.device = torch.device("cuda", torch.cuda.current_device())
        static: list[torch.Tensor] = [self._view("in", i, t, device) for i, t in enumerate(inputs)]
        # The eager calls of this signature were the warm-up (cuDNN/Triton choices, lazy init) and gave the layouts.
        leaves, spec = pytree.tree_flatten(layouts)
        outputs: list = [self._view("out", j, t, device) if isinstance(t, torch.Tensor) else t for j, t in enumerate(leaves)]
        for buffer, tensor in zip(static, inputs, strict=True):
            buffer.copy_(tensor)
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            if self.captures == 0:
                fn(*static)  # once, a warm-up of the capture stream itself (its cuBLAS workspace), per the torch recipe
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
            try:
                for buffer, tensor in zip(outputs, pytree.tree_leaves(fn(*static)), strict=True):
                    if isinstance(buffer, torch.Tensor):
                        buffer.copy_(tensor)
            finally:
                graph.capture_end()
        torch.cuda.current_stream().wait_stream(self._stream)
        self.captures += 1
        return _Captured(graph, static, pytree.tree_unflatten(outputs, spec))


__all__ = ("GraphCache",)
