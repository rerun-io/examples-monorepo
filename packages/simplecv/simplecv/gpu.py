"""The one GPU-consumer step for a ``Fenced`` edge value (SPEC 2.8): wait on the producer's fence, keep its tensors alive."""

import dataclasses
from collections.abc import Iterator
from typing import TypeVar, cast

import torch
from dagkit.node import Fence, Fenced

T = TypeVar("T")


def consume(fenced: Fenced[T], stream: torch.cuda.Stream | None = None) -> T:
    """Make ``stream`` (default: the current one) wait for the fence, then ``record_stream`` every CUDA tensor of the value,
    nested dataclass fields included.

    A host value without a fence (a resumed checkpoint, a CPU test) passes through without touching CUDA.
    """
    tensors: list[torch.Tensor] = [tensor for tensor in _tensors(fenced.value) if tensor.is_cuda]
    if fenced.fence is None and not tensors:
        return fenced.value
    stream = stream or torch.cuda.current_stream()
    if fenced.fence is not None:
        fenced.fence.wait(stream)
    for tensor in tensors:
        tensor.record_stream(stream)
    return fenced.value


def _tensors(value: object) -> Iterator[torch.Tensor]:
    """``value`` when it is a tensor, else the tensors in its dataclass fields, depth first."""
    if isinstance(value, torch.Tensor):
        yield value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _tensors(getattr(value, field.name))


def fence_after(tensor: torch.Tensor) -> Fence | None:
    """A fence on the current stream after the work that produced ``tensor``; None for a host tensor."""
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record()
    return cast(Fence, event)  # torch's Event.wait takes a Stream where dagkit's Fence protocol says object
