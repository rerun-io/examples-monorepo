"""``simplecv.gpu.consume``: the fence wait and the tensors it keeps alive on the consumer's stream, nested dataclass fields
included (unit lane: a host tensor that reports CUDA; the integration test uses the caching allocator of a real GPU)."""

from dataclasses import dataclass

import pytest
import torch
from dagkit.node import Fenced

from simplecv import gpu


class OnDevice(torch.Tensor):
    """A host tensor that reports CUDA and remembers the streams ``record_stream`` was given."""

    @property
    def is_cuda(self) -> bool:  # type: ignore[override]
        return True

    def record_stream(self, stream: object) -> None:  # type: ignore[override]
        self.__dict__.setdefault("streams", []).append(stream)


@dataclass(frozen=True, slots=True)
class Tracks:
    masks: torch.Tensor
    count: int


@dataclass(frozen=True, slots=True)
class Tracked:
    tracks: Tracks
    crops: torch.Tensor
    host: torch.Tensor
    name: str


class Fence:
    def __init__(self) -> None:
        self.waited: list[object] = []

    def synchronize(self) -> None:
        raise AssertionError("a consumer waits on its stream, never on the host")

    def wait(self, stream: object = None) -> None:
        self.waited.append(stream)


def test_consume_waits_once_and_records_every_cuda_tensor_of_nested_dataclass_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    masks, crops = (torch.zeros(2).as_subclass(OnDevice) for _ in range(2))
    value = Tracked(Tracks(masks, 1), crops, torch.zeros(2), "tick")
    monkeypatch.setattr(gpu.torch.cuda, "current_stream", lambda: "consumer stream")
    fence = Fence()
    assert gpu.consume(Fenced(value, fence)) is value  # type: ignore[arg-type]
    assert fence.waited == ["consumer stream"]
    assert masks.__dict__.get("streams") == crops.__dict__.get("streams") == ["consumer stream"]


def test_a_host_value_without_a_fence_passes_through_without_touching_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    value = Tracked(Tracks(torch.zeros(2), 1), torch.zeros(2), torch.zeros(2), "resumed")
    monkeypatch.setattr(gpu.torch.cuda, "current_stream", lambda: pytest.fail("touched CUDA"))
    assert gpu.consume(Fenced(value, None)) is value


@pytest.mark.integration
def test_a_nested_cuda_tensor_stays_allocated_while_the_consumer_stream_still_uses_it() -> None:
    """The caching allocator hands a freed block to the next allocation of its stream at once, unless the tensor was
    recorded on another stream that still has work queued (PyTorch's own record_stream test)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    consumer = torch.cuda.Stream()

    def reused_after(record: bool) -> bool:
        value = Tracked(Tracks(torch.empty(1 << 20, device="cuda"), 1), torch.empty(4, device="cuda"), torch.zeros(2), "tick")
        address: int = value.tracks.masks.data_ptr()
        with torch.cuda.stream(consumer):
            if record:
                gpu.consume(Fenced(value, gpu.fence_after(value.crops)), consumer)
            torch.cuda._sleep(50_000_000)  # the consumer is still busy when the producer frees the value
        del value
        reused: bool = torch.empty(1 << 20, device="cuda").data_ptr() == address
        consumer.synchronize()
        return reused

    assert reused_after(record=False), "control: an unrecorded block is reused at once"
    assert not reused_after(record=True)
