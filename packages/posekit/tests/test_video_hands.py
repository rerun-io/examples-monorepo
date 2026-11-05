"""Per-frame hand logging of the posekit-video-hands app (no models, rr.log captured)."""

from typing import Any

import pytest
import rerun as rr
import torch

from posekit.apis.video_hands import log_chunk_hands
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
    log_chunk_hands(frame_indices=[10, 11], timestamps_ns=[0, 1], detections=detections, hands=hands, keypoint_threshold=0.3, live_slots=live)

    paths = [path for path, _ in logged]
    for slot in ("left_0", "right_0"):
        assert f"world/camera/image/hands/{slot}/box" in paths
        assert f"world/camera/image/hands/{slot}/keypoints" in paths
        assert f"world/camera/hands/{slot}/joints" in paths
        assert f"world/camera/hands/{slot}/mesh" in paths
    # Frame 1 has no right hand: its slot is cleared once, the left slot is logged again.
    cleared = [path for path, archetype in logged if isinstance(archetype, rr.Clear)]
    assert sorted(cleared) == sorted(
        [f"world/camera/image/hands/right_0/{name}" for name in ("box", "keypoints")] + [f"world/camera/hands/right_0/{name}" for name in ("joints", "mesh")]
    )
    assert live == {"left_0"}
    meshes = [archetype for path, archetype in logged if path == "world/camera/hands/right_0/mesh"]
    faces = meshes[0].triangle_indices.as_arrow_array().to_pylist()
    assert faces == [[0, 1, 2]]
