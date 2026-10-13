"""The one GPU-consumer step for a ``Fenced`` edge value (SPEC 2.8): wait on the producer's fence, keep its tensors alive."""

import dataclasses
from typing import TypeVar, cast

import torch
from dagkit.node import Fence, Fenced

T = TypeVar("T")


def consume(fenced: Fenced[T], stream: torch.cuda.Stream | None = None) -> T:
    """Make ``stream`` (default: the current one) wait for the fence, then ``record_stream`` every CUDA tensor field of the value.

    A host value without a fence (a resumed checkpoint, a CPU test) passes through without touching CUDA.
    """
    tensors: list[torch.Tensor] = [v for f in dataclasses.fields(fenced.value) if isinstance(v := getattr(fenced.value, f.name), torch.Tensor) and v.is_cuda]  # type: ignore[arg-type]
    if fenced.fence is None and not tensors:
        return fenced.value
    stream = stream or torch.cuda.current_stream()
    if fenced.fence is not None:
        fenced.fence.wait(stream)
    for tensor in tensors:
        tensor.record_stream(stream)
    return fenced.value


def fence_after(tensor: torch.Tensor) -> Fence | None:
    """A fence on the current stream after the work that produced ``tensor``; None for a host tensor."""
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record()
    return cast(Fence, event)  # torch's Event.wait takes a Stream where dagkit's Fence protocol says object
