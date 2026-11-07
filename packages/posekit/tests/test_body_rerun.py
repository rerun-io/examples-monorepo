"""What ``log_people`` sends for a still image versus a frame of a video logged once as a VideoStream (CPU-only)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rerun as rr
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from torch import Tensor

from posekit.body_rerun import log_people
from posekit.models.sam3d_body import MHR_PARAMETER_WIDTHS, ROOT_INDICES, Sam3dBodyReconstruction
from posekit.predictions import Keypoints3d
from posekit.skeletons import MHR_70

HEIGHT: int = 48
WIDTH: int = 64
ROOT: Path = Path("/world")


def one_person() -> Sam3dBodyReconstruction:
    """One synthetic person inside a 64x48 frame."""

    def zeros(*shape: int) -> Tensor:
        return torch.zeros(1, *shape, dtype=torch.float32)

    return Sam3dBodyReconstruction(
        keypoints=Keypoints3d(
            torch.full((1, 70, 2), 10.0), zeros(70), torch.ones(1, 70), torch.zeros(1, dtype=torch.int64), MHR_70, ROOT_INDICES, zeros(70, 3) + 1.0
        ),
        boxes=torch.tensor([[4.0, 4.0, 20.0, 30.0]]),
        vertices=torch.rand(1, 6, 3),
        faces=torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64),
        translation=zeros(3),
        focal_length=torch.tensor([80.0]),
        global_rotation=zeros(3),
        **{name: zeros(width) for name, width in MHR_PARAMETER_WIDTHS.items()},
        joint_coordinates=zeros(4, 3),
        joint_rotations=zeros(4, 3, 3),
        left_hand_boxes=None,
        right_hand_boxes=None,
    )


def person_mask() -> Bool[ndarray, "1 h w"]:
    mask: Bool[ndarray, "1 h w"] = np.zeros((1, HEIGHT, WIDTH), dtype=bool)
    mask[0, 4:30, 4:20] = True
    return mask


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Entity path -> last archetype logged there."""
    sent: dict[str, Any] = {}
    monkeypatch.setattr(rr, "log", lambda entity_path, archetype, **_kwargs: sent.__setitem__(str(entity_path), archetype))
    return sent


def segmentation_hw(archetype: Any) -> tuple[int, int]:
    batches: dict[str, Any] = {batch.component_descriptor().component: batch for batch in archetype.as_component_batches()}
    image_format: dict[str, Any] = batches["SegmentationImage:format"].as_arrow_array().to_pylist()[0]
    return int(image_format["height"]), int(image_format["width"])


def test_still_image_logs_the_frame(logged: dict[str, Any]) -> None:
    rgb: UInt8[ndarray, "h w 3"] = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    log_people(one_person(), rgb, ROOT, masks=person_mask())
    assert "/world/cam/pinhole/image" in logged
    assert segmentation_hw(logged["/world/cam/pinhole/pred/segmentation_ids"]) == (HEIGHT, WIDTH)


def test_video_frame_logs_overlays_without_the_frame(logged: dict[str, Any]) -> None:
    log_people(one_person(), (HEIGHT, WIDTH), ROOT, masks=person_mask())
    assert "/world/cam/pinhole/image" not in logged, "the source video is logged once as a VideoStream, not per frame"
    for entity in (
        "/world/cam/pinhole",
        "/world/cam/pinhole/pred/bbox_0",
        "/world/cam/pinhole/pred/uv_0",
        "/world/pred/kpts3d_0",
        "/world/pred/mesh_0",
    ):
        assert entity in logged, entity
    assert segmentation_hw(logged["/world/cam/pinhole/pred/segmentation_ids"]) == (HEIGHT, WIDTH)


def test_depth_clouds_need_the_frame() -> None:
    depth: Float32[ndarray, "h w"] = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    with pytest.raises(ValueError, match="Depth clouds need the frame"):
        log_people(one_person(), (HEIGHT, WIDTH), ROOT, depth_meters=depth, depth_confidence=depth, depth_intrinsics=np.eye(3, dtype=np.float32))
