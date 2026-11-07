"""LandmarkEstimator crops: sampling each camera's frame in place gives the same crops as the per-person frame copies."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray

from mamma.engine.types import CameraTracks, TrackedObject
from mamma.landmarks.config import DEFAULT_MAMMANET_CONFIG
from mamma.landmarks.crops import box_geometry, gpu_crop_batch
from mamma.landmarks.estimator import LandmarkEstimator
from mamma.landmarks.tensorrt_backend import OUTPUT_NAMES

_JOINTS: int = 8


class _RecordingModel:
    """Stands in for MammaNet: keeps the RGB crops it receives and returns zero landmarks."""

    def __init__(self) -> None:
        self.img_crops: list[torch.Tensor] = []

    def __call__(self, img_crops: torch.Tensor, _mask_crops: torch.Tensor) -> dict[str, torch.Tensor]:
        self.img_crops.append(img_crops.clone())
        n: int = img_crops.shape[0]
        zeros: torch.Tensor = torch.zeros(n, _JOINTS, 1, device=img_crops.device)
        return {
            OUTPUT_NAMES[0]: torch.zeros(n, _JOINTS, 3, device=img_crops.device),
            OUTPUT_NAMES[1]: zeros,
            OUTPUT_NAMES[2]: zeros,
            OUTPUT_NAMES[3]: zeros,
        }


def _track(obj_id: int, box_xyxy: list[float] | None, hw: tuple[int, int], device: str) -> TrackedObject:
    bbox: Float32[ndarray, "4"] | None = None if box_xyxy is None else np.asarray(box_xyxy, dtype=np.float32)
    return TrackedObject(obj_id=obj_id, mask=torch.ones(hw, dtype=torch.bool, device=device), bbox_xyxy=bbox, score=1.0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.integration)])
def test_hires_crops_match_per_person_frame_copies(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    generator: torch.Generator = torch.Generator().manual_seed(0)
    engine_hw: tuple[int, int] = (36, 64)
    hires_scale: float = 3.0
    frames_hires: list[UInt8[torch.Tensor, "3 hh ww"]] = [
        torch.randint(0, 256, (3, 108, 192), dtype=torch.uint8, generator=generator).to(device) for _ in range(3)
    ]
    frames: list[UInt8[torch.Tensor, "3 h w"]] = [torch.zeros(3, *engine_hw, dtype=torch.uint8, device=device) for _ in range(3)]
    boxes: dict[tuple[int, int], list[float]] = {(0, 2): [2.0, 1.0, 20.0, 30.0], (0, 7): [30.0, 4.0, 50.0, 34.0], (2, 5): [10.0, 5.0, 40.0, 33.0]}
    # Camera 0: two people (dict order differs from obj_id order); camera 1: nobody usable; camera 2: one person.
    tracks: list[CameraTracks] = [
        {7: _track(7, boxes[(0, 7)], engine_hw, device), 2: _track(2, boxes[(0, 2)], engine_hw, device)},
        {4: _track(4, None, engine_hw, device)},
        {5: _track(5, boxes[(2, 5)], engine_hw, device)},
    ]

    estimator: LandmarkEstimator = LandmarkEstimator.__new__(LandmarkEstimator)
    estimator.config = DEFAULT_MAMMANET_CONFIG
    estimator.device = device
    estimator.runner = None
    model: _RecordingModel = _RecordingModel()
    estimator.model = model  # type: ignore[assignment]
    results = estimator.estimate(frames, tracks, frames_hires=frames_hires, hires_scale=hires_scale)

    # Reference: the previous layout, one float copy of the camera frame per person, stacked.
    entries: list[tuple[int, int]] = [(0, 2), (0, 7), (2, 5)]
    geometry: list[tuple[Float64[ndarray, "2"], Float64[ndarray, "2"]]] = [
        box_geometry(np.asarray(boxes[entry], dtype=np.float32), DEFAULT_MAMMANET_CONFIG) for entry in entries
    ]
    centers: Float64[ndarray, "n 2"] = np.stack([center for center, _ in geometry])
    sizes: Float64[ndarray, "n 2"] = np.stack([size for _, size in geometry])
    stacked: Float32[torch.Tensor, "n 3 hh ww"] = torch.stack([frames_hires[cam].float() for cam, _ in entries])
    expected, _ = gpu_crop_batch(stacked, centers * hires_scale, sizes * hires_scale, None, DEFAULT_MAMMANET_CONFIG)

    assert len(model.img_crops) == 1
    assert torch.equal(model.img_crops[0], expected)
    assert [sorted(cam_results) for cam_results in results] == [[2, 7], [], [5]]
