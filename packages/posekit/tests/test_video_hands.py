"""Per-frame hand logging of the posekit-video-hands app (no models, rr.log captured)."""

from typing import Any

import pytest
import rerun as rr
import torch

from posekit.apis.video_hands import VideoHandsSettings, assumed_pinhole, describe_camera, log_chunk_hands, pinhole_camera
from posekit.predictions import BoxDetections, HandDetections, HandPoses3d, Keypoints3d
from posekit.skeletons import HAND_21


def _chunk() -> tuple[HandDetections, HandPoses3d]:
    """Frame 0: a left and a right hand; frame 1: only a left hand."""
    frames = torch.tensor([0, 0, 1])
    sides = ("left", "right", "left")
    detections = HandDetections(BoxDetections(torch.tensor([[0.0, 0, 10, 10], [20, 0, 30, 10], [1, 1, 11, 11]]), torch.ones(3), frames), sides)
    xyz = torch.rand((3, 21, 3)) + torch.tensor([0.0, 0.0, 1.0])
    hands = HandPoses3d(
        keypoints=Keypoints3d(xyz[..., :2] * 100, xyz[..., 2] - xyz[:, :1, 2], torch.ones((3, 21)), frames, HAND_21, (0,), xyz),
        sides=sides,
        vertices=torch.rand((3, 5, 3)),
        faces={"left": torch.tensor([[0, 2, 1]]), "right": torch.tensor([[0, 1, 2]])},
    )
    return detections, hands


def test_hands_go_to_side_slots_and_an_emptied_slot_is_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[tuple[str, Any]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype, **_: logged.append((path, archetype)))
    monkeypatch.setattr(rr, "set_time", lambda *_, **__: None)
    detections, hands = _chunk()
    live: set[str] = set()
    log_chunk_hands(frame_indices=[10, 11], timestamps_ns=[0, 1], detections=detections, hands=hands, live_slots=live)

    paths = [path for path, _ in logged]
    # Joints and meshes are logged once in 3D; the video view projects them, so no 2D keypoints.
    assert not any(path.endswith("/keypoints") for path in paths)
    for slot in ("left_0", "right_0"):
        assert f"world/camera/pinhole/boxes/{slot}" in paths
        assert f"world/camera/hands/{slot}/joints" in paths
        assert f"world/camera/hands/{slot}/mesh" in paths
    # Frame 1 has no right hand: its slot is cleared once, the left slot is logged again.
    cleared = [path for path, archetype in logged if isinstance(archetype, rr.Clear)]
    assert sorted(cleared) == sorted(
        ["world/camera/pinhole/boxes/right_0"] + [f"world/camera/hands/right_0/{name}" for name in ("joints", "mesh")]
    )
    assert live == {"left_0"}
    meshes = [archetype for path, archetype in logged if path == "world/camera/hands/right_0/mesh"]
    faces = meshes[0].triangle_indices.as_arrow_array().to_pylist()
    assert faces == [[0, 1, 2]]


def test_unset_intrinsics_come_from_the_assumed_field_of_view() -> None:
    fx, fy, cx, cy = assumed_pinhole(1920, 1080, 65.0)
    assert fx == fy == pytest.approx(1507.0, abs=0.5) and (cx, cy) == (960.0, 540.0)
    camera = pinhole_camera(VideoHandsSettings(fx=1000.0, cy=500.0), 1920, 1080)
    assert camera.lens == "PINHOLE" and (camera.width, camera.height) == (1920, 1080)
    assert camera.intrinsics.tolist() == [[1000.0, 0.0, 960.0], [0.0, fy, 500.0], [0.0, 0.0, 1.0]]


def test_status_names_the_camera() -> None:
    settings = VideoHandsSettings()
    assert describe_camera(settings, pinhole_camera(settings, 1920, 1080)) == "assumed pinhole, f = 1507 px (65° horizontal FOV)"
    given = VideoHandsSettings(fx=1200.0, fy=1210.0, cx=950.0, cy=530.0)
    assert describe_camera(given, pinhole_camera(given, 1920, 1080)) == "pinhole fx = 1200, fy = 1210, cx = 950, cy = 530 px"


def test_joints_carry_full_confidence_and_show_green(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hand models give no per-joint confidence, so every joint is logged at 1.0: the confidence gradient's green.
    logged: list[tuple[str, Any]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype, **_: logged.append((path, archetype)))
    monkeypatch.setattr(rr, "set_time", lambda *_, **__: None)
    detections, hands = _chunk()
    log_chunk_hands(frame_indices=[10, 11], timestamps_ns=[0, 1], detections=detections, hands=hands, live_slots=set())

    joints = next(archetype for path, archetype in logged if path == "world/camera/hands/left_0/joints")
    batches = {batch.component_descriptor().component: batch.as_arrow_array().to_pylist() for batch in joints.as_component_batches()}
    assert batches["simplecv.KeypointConfidence3D:confidences"] == [1.0] * 21
    assert {packed >> 8 for packed in batches["Points3D:colors"]} == {0x00FF00}
