"""Hand sphere geometry and its track: boxes of a known sphere in every view, triangulation back from 1-2 labeled views,
the 1-view depth search against the other views' proposals, and one sphere per hand carried through time."""

from __future__ import annotations

import math

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

from posekit.labeler.rig import Kb4Rig
from posekit.labeler.schema import Box, HandSide, Proposal, TrackBox, ViewLabel
from posekit.labeler.sphere import HAND_RADIUS_M, HandSphere, box_iou, lift_box, solve_views, sphere_box, triangulate
from posekit.labeler.spheretrack import SpherePropagator, SphereStep

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


# -- the sphere track (posekit.labeler.spheretrack) --------------------------------------------------------------------

STEP_NS: int = 200_000_000
CAMS: list[int] = [0, 1, 2, 3, 4]


def moving(step: int) -> HandSphere:
    """The left hand moving 3 cm right and 1 cm forward per grid step."""
    return HandSphere(np.array([-0.06 + 0.03 * step, 0.22, 0.30 + 0.01 * step]), 0.09)


def proposals_of(rig: Kb4Rig, sphere: HandSphere, *, flip: bool = False, skip: frozenset[int] = frozenset()) -> list[Proposal]:
    """EgoHand boxes of the sphere in every camera that sees it (``flip`` = EgoHand calls it the other side)."""
    side: HandSide = "right" if flip else "left"
    out: list[Proposal] = []
    for cam in rig.cams:
        box: Box | None = sphere_box(rig, cam, sphere)
        if box is not None and cam not in skip:
            out.append(Proposal(cam, side, box, 0.8, "egohand"))
    return out


def person(cam: int, box: Box) -> ViewLabel:
    return ViewLabel(cam, "left", "corrected", box, "person", "pablo")


def left(step: SphereStep, cam: int) -> TrackBox:
    return next(b for b in step.frame.boxes if b.cam == cam and b.side == "left")


def run(rig: Kb4Rig, steps: int, seed_cams: tuple[int, ...], skip_at: dict[int, frozenset[int]] | None = None) -> list[SphereStep]:
    propagator: SpherePropagator = SpherePropagator(rig)
    out: list[SphereStep] = []
    previous: SphereStep | None = None
    for k in range(steps):
        truth: HandSphere = moving(k)
        labels: list[ViewLabel] = []
        if k == 0:
            for cam in seed_cams:
                box: Box | None = sphere_box(rig, cam, truth)
                assert box is not None
                labels.append(person(cam, box))
        skip: frozenset[int] = (skip_at or {}).get(k, frozenset())
        previous = propagator.step(k * STEP_NS, CAMS, proposals_of(rig, truth, flip=k % 2 == 1, skip=skip), labels, previous)
        out.append(previous)
    return out


def test_two_view_seed_is_tracked_in_every_view() -> None:
    rig: Kb4Rig = head_rig()
    steps: list[SphereStep] = run(rig, 10, (1, 2))
    assert left(steps[0], 1).origin == "reviewed" and left(steps[0], 0).origin == "sphere"
    for k, step in enumerate(steps):
        sphere: HandSphere = step.hands["left"].sphere
        assert np.linalg.norm(sphere.centre - moving(k).centre) < 0.02, k  # the pull towards the last centre lags a moving hand
        for cam in (0, 1, 2, 3):
            got: TrackBox = left(step, cam)
            want: Box | None = sphere_box(rig, cam, moving(k))
            assert want is not None and got.box is not None and box_iou(got.box, want) > 0.85, (k, cam)
        assert left(step, 4).box is None  # the backward camera never sees the hand


def test_one_missing_step_holds_then_recovers() -> None:
    rig: Kb4Rig = head_rig()
    steps: list[SphereStep] = run(rig, 6, (1, 2), skip_at={3: frozenset(CAMS)})
    assert left(steps[3], 1).origin == "held"
    assert left(steps[4], 1).origin == "sphere"
    assert np.linalg.norm(steps[5].hands["left"].sphere.centre - moving(5).centre) < 0.03  # catching up after the 6 cm gap


def test_two_missing_steps_end_the_track() -> None:
    rig: Kb4Rig = head_rig()
    steps: list[SphereStep] = run(rig, 6, (1, 2), skip_at={3: frozenset(CAMS), 4: frozenset(CAMS)})
    assert "left" not in steps[4].hands
    assert left(steps[4], 1).origin != "sphere"


def test_one_view_seed_uses_the_proposals_for_depth() -> None:
    rig: Kb4Rig = head_rig()
    steps: list[SphereStep] = run(rig, 4, (1,))
    assert np.linalg.norm(steps[0].hands["left"].sphere.centre - moving(0).centre) < 0.02
    assert np.linalg.norm(steps[3].hands["left"].sphere.centre - moving(3).centre) < 0.02


def test_saved_derived_label_shows_as_derived() -> None:
    rig: Kb4Rig = head_rig()
    box: Box | None = sphere_box(rig, 0, moving(0))
    assert box is not None
    derived: ViewLabel = ViewLabel(0, "left", "accepted", box, "rig", "pablo")
    step: SphereStep = SpherePropagator(rig).step(0, CAMS, [], [derived], None)
    assert left(step, 0).origin == "derived"


def test_two_close_hands_never_share_a_box() -> None:
    """Both hands at the sink: each camera's EgoHand box goes to one hand only, so the tracks do not merge."""
    rig: Kb4Rig = head_rig()
    hands: dict[HandSide, HandSphere] = {
        "left": HandSphere(np.array([-0.05, 0.22, 0.32]), 0.07),
        "right": HandSphere(np.array([0.09, 0.22, 0.32]), 0.07),
    }
    labels: list[ViewLabel] = []
    for side, sphere in hands.items():
        for cam in (1, 2):
            box: Box | None = sphere_box(rig, cam, sphere)
            assert box is not None
            labels.append(ViewLabel(cam, side, "corrected", box, "person", "pablo"))
    propagator: SpherePropagator = SpherePropagator(rig)
    step: SphereStep = propagator.step(0, CAMS, [], labels, None)
    # Next step: only the right hand is detected (the left one is hidden), and EgoHand calls it "left".
    only_right: list[Proposal] = [Proposal(p.cam, "left", p.box, p.score, p.model) for p in proposals_of(rig, hands["right"])]
    step = propagator.step(STEP_NS, CAMS, only_right, [], step)
    assert np.linalg.norm(step.hands["right"].sphere.centre - hands["right"].centre) < 0.01
    assert step.hands["left"].held == 1  # held where it was, not pulled onto the right hand
