"""Camera and anatomical contracts, independent of model weights."""

import cv2
import numpy as np
import pytest

pytest.importorskip("egohand_inference", reason="requires the egohand Pixi environment")

from posekit.models.egohand_geometry import (
    OAKCamera,
    coco_hand_indices,
    detector_hand_side,
    soma_to_hand21,
)


def test_oak_projection_retains_all_fourteen_coefficients() -> None:
    intrinsics = np.array([[600.0, 0.0, 640.0], [0.0, 601.0, 360.0], [0.0, 0.0, 1.0]])
    distortion = np.array([0.03, -0.01, 0.002, -0.003, 0.004, 0.005, -0.002, 0.001, 0.001, -0.002, 0.003, -0.001, 0.02, -0.03])
    points = np.array([[0.3, -0.2, 1.0], [-0.7, 0.4, 1.5], [0.0, 0.0, 2.0]])
    camera = OAKCamera(intrinsics, distortion, 1280, 720)
    expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsics, distortion)
    np.testing.assert_allclose(camera.project(points), expected[:, 0], atol=1e-8)
    rays = camera.unproject(expected[:, 0])
    np.testing.assert_allclose(rays[:, :2] / rays[:, 2:], points[:, :2] / points[:, 2:], atol=1e-7)


def test_raw_detector_class_names_match_upstream_and_unknown_labels_fail() -> None:
    assert detector_hand_side(1) == "left"
    assert detector_hand_side(2) == "right"
    with pytest.raises(ValueError, match="hand label"):
        detector_hand_side(0)


def test_soma_mapping_skips_extra_metacarpals_for_coco_and_quest21() -> None:
    # Unique coordinates make the independently specified anatomical order visible.
    points = np.repeat(np.arange(25, dtype=np.float32)[:, None], 3, axis=1)
    expected = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    np.testing.assert_array_equal(soma_to_hand21(points)[:, 0], expected)
    assert coco_hand_indices("left") == tuple(range(91, 112))
    assert coco_hand_indices("right") == tuple(range(112, 133))


def test_identity_assignment_precedes_crop_mirroring() -> None:
    import torch

    from posekit.models.egohand import HandDetections, resolve_hand_identities
    from posekit.predictions import BoxDetections

    # Released detector at 4.128 s: class 1 covers the wearer's right hand.
    raw = HandDetections(
        BoxDetections(
            torch.tensor([[743.0, 349.0, 964.0, 551.0], [71.0, 339.0, 276.0, 571.0], [745.0, 350.0, 966.0, 552.0], [73.0, 341.0, 278.0, 573.0]]),
            torch.tensor([0.89, 0.81, 0.90, 0.82]),
            torch.tensor([0, 0, 1, 1]),
        ),
        ("left", "right", "left", "right"),
    )
    result = resolve_hand_identities(raw, frame_count=2)
    assert result.sides == ("left", "right", "left", "right")
    torch.testing.assert_close(result.boxes.xyxy, raw.boxes.xyxy[[1, 0, 3, 2]])
    # Resolving an already-correct pair must not swap it again.
    stable = resolve_hand_identities(result, frame_count=2)
    torch.testing.assert_close(stable.boxes.xyxy, result.boxes.xyxy)


def test_identity_assignment_can_recover_two_hands_with_one_class() -> None:
    import torch

    from posekit.models.egohand import HandDetections, resolve_hand_identities
    from posekit.predictions import BoxDetections

    raw = HandDetections(
        BoxDetections(torch.tensor([[100.0, 200.0, 200.0, 300.0], [700.0, 200.0, 800.0, 300.0]]), torch.tensor([0.9, 0.8]), torch.tensor([0, 0])),
        ("right", "right"),
    )
    assigned = resolve_hand_identities(raw, frame_count=1)
    assert assigned.sides == ("left", "right")
    torch.testing.assert_close(assigned.boxes.xyxy, raw.boxes.xyxy)
