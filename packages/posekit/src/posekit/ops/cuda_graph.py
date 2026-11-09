"""Captured CUDA graphs over Torch and Warp work, shared by the EgoHand and SAM 3D Body executors: resource lifetimes during capture and replay."""

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeAlias

import torch
import warp as wp
from torch import Tensor

GraphPool: TypeAlias = torch.cuda._POOL_HANDLE
"""Memory pool token of ``torch.cuda.graph_pool_handle()``; graphs sharing one must not replay concurrently."""
CAPTURE_WARMUPS: int = 3
"""Eager runs on the capture stream before capture, so lazy allocations and compilation happen outside the graph."""


@contextmanager
def capture_graph(
    graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream, warp_stream: wp.Stream, pool: GraphPool | None = None
) -> Iterator[None]:
    """Keep unrelated TensorRT destructors out of the capture interval.

    Model cycles from a prior caller can otherwise be collected on any Python
    allocation inside capture. Their CUDA module unloads invalidate that graph.
    Collect before capture and restore the caller's GC setting afterwards.
    Capture errors are thread-local, as in trtkit's engine graphs.
    """
    enabled: bool = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        with (
            wp.ScopedMempool(device=str(stream.device), enable=True),
            wp.ScopedStream(warp_stream),
            torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"),
            wp.ScopedCapture(external=True),
        ):
            yield
    finally:
        if enabled:
            gc.enable()


class CapturedGraph[Outputs]:
    """One captured CUDA graph with the storage it reads and writes.

    The owner holds the static inputs that a replay copies new values into, the
    outputs the graph writes, and strong references to the external resources
    the captured work reads (Warp arrays, model buffers, events, streams). It
    resets the graph before any of those are released.
    """

    def __init__(self, run: Callable[[], Outputs], inputs: tuple[Tensor, ...], *, pool: GraphPool, keep: tuple[object, ...] = ()) -> None:
        """Warm ``run`` up on a side stream, then capture it.

        Args:
            run: Capture-safe work that reads only ``inputs`` among the changing values.
            inputs: Static input storage; :meth:`replay` copies into it in this order.
            pool: Memory pool shared with the owner's other graphs, which replay one at a time.
            keep: External resources the captured work reads.
        """
        stream: torch.cuda.Stream = torch.cuda.Stream()
        warp_stream: wp.Stream = wp.stream_from_torch(stream)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
            for _ in range(CAPTURE_WARMUPS):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with capture_graph(self.graph, stream, warp_stream, pool):
            self.outputs: Outputs = run()
        self.inputs: tuple[Tensor, ...] = inputs
        self.resources: tuple[object, ...] = (stream, warp_stream, *keep)

    def replay(self, *values: Tensor) -> Outputs:
        """Copy ``values`` into the static inputs and replay on the current stream; the outputs are overwritten by the next replay."""
        for storage, value in zip(self.inputs, values, strict=True):
            storage.copy_(value)
        self.graph.replay()
        return self.outputs

    def __del__(self) -> None:
        """Destroy the captured work while everything it reads is still alive."""
        graph: torch.cuda.CUDAGraph | None = getattr(self, "graph", None)
        if graph is not None:
            graph.reset()
