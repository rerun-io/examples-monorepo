"""Hand sphere geometry: boxes of a known sphere in every view, triangulation back from 1-2 labeled views, and the 1-view
depth search against the other views' proposals."""

from __future__ import annotations

import math

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from posekit.labeler.rig import Kb4Rig
from posekit.labeler.schema import Box
from posekit.labeler.sphere import HAND_RADIUS_M, HandSphere, box_iou, lift_box, solve_views, sphere_box, triangulate

YAWS_DEG: tuple[float, ...] = (-70.0, -25.0, 25.0, 70.0, 180.0)
"""Four cameras look forward and down around the head; the last looks backwards and never sees the hands."""
TRUTH: HandSphere = HandSphere(np.array([0.04, 0.22, 0.32]), 0.09)


def look(yaw_deg: float, pitch_deg: float) -> Float64[ndarray, "3 3"]:
    """Rig-from-camera rotation of a camera looking at yaw (about +y) and pitch (down), rig frame x right, y down, z forward."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward: Float64[ndarray, "3"] = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    right: Float64[ndarray, "3"] = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(forward, right), forward], axis=-1)


def head_rig() -> Kb4Rig:
    """Five 1280x720 KB4 fisheyes on a 9 cm head, RoboCap-like (wide, slightly non-equidistant lenses)."""
    count: int = len(YAWS_DEG)
    cam_from_rig: Float64[ndarray, "c 4 4"] = np.tile(np.eye(4), (count, 1, 1))
    for cam, yaw in enumerate(YAWS_DEG):
        rig_from_cam: Float64[ndarray, "3 3"] = look(yaw, 35.0 if yaw != 180.0 else 0.0)
        origin: Float64[ndarray, "3"] = 0.09 * np.array([math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw))])
        cam_from_rig[cam, :3, :3] = rig_from_cam.T
        cam_from_rig[cam, :3, 3] = -rig_from_cam.T @ origin
    return Kb4Rig(
        names=tuple(f"cam{cam}" for cam in range(count)),
        size_wh=np.tile([1280.0, 720.0], (count, 1)),
        focal=np.full((count, 2), 380.0),
        principal=np.tile([640.0, 360.0], (count, 1)),
        kb4=np.tile([0.02, -0.004, 0.001, 0.0], (count, 1)),
        cam_from_rig=cam_from_rig,
    )


def truth_boxes(rig: Kb4Rig, sphere: HandSphere = TRUTH) -> dict[int, Box]:
    boxes: dict[int, Box] = {}
    for cam in rig.cams:
        box: Box | None = sphere_box(rig, cam, sphere)
        if box is not None:
            boxes[cam] = box
    return boxes


def test_on_axis_box_matches_the_equidistant_formula() -> None:
    rig: Kb4Rig = Kb4Rig(
        ("a",), np.array([[1920.0, 1080.0]]), np.array([[600.0, 600.0]]), np.array([[960.0, 540.0]]), np.zeros((1, 4)), np.eye(4)[None]
    )
    box: Box | None = sphere_box(rig, 0, HandSphere(np.array([0.0, 0.0, 0.5]), 0.1))
    assert box is not None
    half: float = 600.0 * math.asin(0.1 / 0.5)  # k = 0: radius on the image = f * theta, theta = the cone's half angle
    for got, want in zip((box.x0, box.y0, box.x1, box.y1), (960.0 - half, 540.0 - half, 960.0 + half, 540.0 + half), strict=True):
        assert got == pytest.approx(want, abs=0.5)


def test_sphere_behind_a_camera_has_no_box() -> None:
    rig: Kb4Rig = head_rig()
    boxes: dict[int, Box] = truth_boxes(rig)
    assert 4 not in boxes
    assert {1, 2} <= set(boxes)


@pytest.mark.parametrize("labeled", [(1, 2), (0, 2), (1, 3), (0, 1, 2, 3)])
def test_triangulate_recovers_the_sphere_and_the_other_views(labeled: tuple[int, ...]) -> None:
    rig: Kb4Rig = head_rig()
    boxes: dict[int, Box] = truth_boxes(rig)
    sphere: HandSphere = triangulate(rig, {cam: boxes[cam] for cam in labeled})
    assert np.linalg.norm(sphere.centre - TRUTH.centre) < 0.005
    assert sphere.radius == pytest.approx(TRUTH.radius, abs=0.004)
    for cam, want in boxes.items():
        got: Box | None = sphere_box(rig, cam, sphere)
        assert got is not None and box_iou(got, want) > 0.93, (cam, got, want)


def test_triangulate_tolerates_hand_drawn_noise() -> None:
    rig: Kb4Rig = head_rig()
    boxes: dict[int, Box] = truth_boxes(rig)
    noise: Float64[ndarray, "2 4"] = np.random.default_rng(3).normal(0.0, 4.0, (2, 4))
    noisy: dict[int, Box] = {
        cam: Box(*(np.array([b.x0, b.y0, b.x1, b.y1]) + noise[i]).tolist()) for i, (cam, b) in enumerate((c, boxes[c]) for c in (1, 2))
    }
    sphere: HandSphere = triangulate(rig, noisy)
    # Edge noise mostly trades depth against size; the derived boxes, which is what the person sees, stay close.
    assert np.linalg.norm(sphere.centre - TRUTH.centre) < 0.03
    for cam in (0, 3):
        got: Box | None = sphere_box(rig, cam, sphere)
        assert got is not None and box_iou(got, boxes[cam]) > 0.8, (cam, got, boxes[cam])


def test_one_view_depth_from_the_other_views_proposals() -> None:
    rig: Kb4Rig = head_rig()
    boxes: dict[int, Box] = truth_boxes(rig)
    distractor: Box = Box(50.0, 50.0, 150.0, 140.0)
    proposals: dict[int, list[Box]] = {2: [distractor, boxes[2]], 3: [boxes[3]], 0: [distractor]}
    choice = lift_box(rig, 1, boxes[1], proposals)
    assert choice.mode == "proposals"
    assert choice.confidence > 0.8
    assert set(choice.support) == {2, 3}
    assert np.linalg.norm(choice.sphere.centre - TRUTH.centre) < 0.02


def test_one_view_without_proposals_falls_back_to_the_hand_size_prior() -> None:
    rig: Kb4Rig = head_rig()
    boxes: dict[int, Box] = truth_boxes(rig, HandSphere(TRUTH.centre, HAND_RADIUS_M))
    choice = lift_box(rig, 1, boxes[1], {})
    assert choice.mode == "prior"
    assert choice.confidence == 0.0
    assert choice.sphere.radius == pytest.approx(HAND_RADIUS_M, abs=1e-3)
    assert np.linalg.norm(choice.sphere.centre - TRUTH.centre) < 0.01


def test_solve_views_derives_every_unlabeled_view() -> None:
    rig: Kb4Rig = head_rig()
    boxes: dict[int, Box] = truth_boxes(rig)
    solved = solve_views(rig, {1: boxes[1], 2: boxes[2]}, {})
    assert solved.mode == "triangulated"
    assert solved.labeled == (1, 2)
    assert solved.boxes[1] == boxes[1] and solved.boxes[2] == boxes[2]  # labeled views keep the person's box
    assert solved.boxes[4] is None
    assert box_iou(solved.boxes[3] or Box(0, 0, 1, 1), boxes[3]) > 0.93


def test_short_baseline_pair_reports_low_confidence() -> None:
    """Two cameras 2 cm apart (RoboCap's stacked front and eye cameras) barely fix depth: the solve says so."""
    rig: Kb4Rig = head_rig()
    stacked: ndarray = rig.cam_from_rig.copy()
    stacked[0] = stacked[1]
    stacked[0, 1, 3] -= 0.02  # 2 cm below cam 1, same orientation
    near: Kb4Rig = Kb4Rig(rig.names, rig.size_wh, rig.focal, rig.principal, rig.kb4, stacked)
    boxes: dict[int, Box] = truth_boxes(near)
    assert solve_views(near, {0: boxes[0], 1: boxes[1]}, {}).confidence < 0.5
    assert solve_views(near, {1: boxes[1], 2: boxes[2]}, {}).confidence == 1.0
