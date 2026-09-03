"""Tests for shared Polycam data helpers."""

from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float32, UInt8
from numpy.testing import assert_array_equal
from torch import Tensor

from simplecv.data.polycam import PolycamData, PolycamDataset, prepare_polycam_batches, stack_polycam_batch


def _frame(value: int) -> PolycamData:
    frame: PolycamData = object.__new__(PolycamData)
    frame.rgb_hw3 = np.full((2, 3, 3), value, dtype=np.uint8)
    frame.original_depth_hw = np.full((192, 256), value * 1000, dtype=np.uint16)
    return frame


def _dataset(monkeypatch: pytest.MonkeyPatch, frames: list[PolycamData]) -> PolycamDataset:
    monkeypatch.setattr(PolycamDataset, "__iter__", lambda _self: iter(frames))
    dataset: PolycamDataset = object.__new__(PolycamDataset)
    dataset.cameras_path = [Path(f"{index}.json") for index in range(len(frames))]
    return dataset


def test_prepare_polycam_batches_peeks_and_honors_exact_frame_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the first batch while limiting the final batch to the frame budget."""
    frames: list[PolycamData] = [_frame(index) for index in range(1, 6)]
    dataset: PolycamDataset = _dataset(monkeypatch, frames)

    plan = prepare_polycam_batches(dataset, batch_size=2, max_frames=3, capture_path=Path("capture.zip"), description="Test")

    assert plan.first_batch == (frames[0], frames[1])
    assert plan.total_batches == 2
    assert list(plan) == [(frames[0], frames[1]), (frames[2],)]


def test_prepare_polycam_batches_rejects_empty_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the capture path when no frame can be peeked."""
    dataset: PolycamDataset = _dataset(monkeypatch, [])

    with pytest.raises(ValueError, match=r"Polycam capture empty\.zip contains no frames\."):
        prepare_polycam_batches(dataset, batch_size=8, max_frames=None, capture_path=Path("empty.zip"), description="Test")


def test_stack_polycam_batch_returns_cuda_ready_raw_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stack RGB unchanged and convert raw uint16-millimetre prompts to float metres."""
    cuda_calls: list[UInt8[Tensor, "b h w 3"] | Float32[Tensor, "b 192 256"]] = []

    def fake_cuda(
        tensor: UInt8[Tensor, "b h w 3"] | Float32[Tensor, "b 192 256"],
    ) -> UInt8[Tensor, "b h w 3"] | Float32[Tensor, "b 192 256"]:
        cuda_calls.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "cuda", fake_cuda)
    frames: tuple[PolycamData, ...] = (_frame(0), _frame(2))

    tensor_batch: tuple[UInt8[Tensor, "b h w 3"], Float32[Tensor, "b 192 256"]] = stack_polycam_batch(frames)
    rgb_bhwc: UInt8[Tensor, "b h w 3"] = tensor_batch[0]
    prompt_bhw: Float32[Tensor, "b 192 256"] = tensor_batch[1]

    assert rgb_bhwc.dtype is torch.uint8
    assert prompt_bhw.dtype is torch.float32
    assert_array_equal(rgb_bhwc.numpy(), np.stack([frame.rgb_hw3 for frame in frames]))
    assert_array_equal(prompt_bhw.numpy(), np.stack([frame.original_depth_hw for frame in frames]) / 1000.0)
    assert cuda_calls == [rgb_bhwc, prompt_bhw]
