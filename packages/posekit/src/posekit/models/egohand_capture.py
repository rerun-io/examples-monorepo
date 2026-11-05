"""Coordinate Python, Torch and Warp resource lifetimes during CUDA capture."""

import gc
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import warp as wp


@contextmanager
def capture_hand_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream, warp_stream: wp.Stream) -> Iterator[None]:
    """Keep unrelated TensorRT destructors out of the capture interval.

    Model cycles from a prior caller can otherwise be collected on any Python
    allocation inside capture. Their CUDA module unloads invalidate that graph.
    Collect before capture and restore the caller's GC setting afterwards.
    """
    enabled: bool = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        with wp.ScopedMempool(device=str(stream.device), enable=True), wp.ScopedStream(warp_stream), torch.cuda.graph(graph, stream=stream), wp.ScopedCapture(external=True):
            yield
    finally:
        if enabled:
            gc.enable()
