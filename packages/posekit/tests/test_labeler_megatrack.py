"""MEgATrack propagation on a synthetic rig (posekit.labeler.rig, posekit.labeler.megatrack) with a fake keypoint network."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray

from posekit.labeler.megatrack import CropRequest, MegaTrackPropagator, MegaTrackStep
from posekit.labeler.proposals import iou
from posekit.labeler.rig import Kb4Rig, ViewKeypoints, fuse_views, keypoint_box
from posekit.labeler.schema import Box, HandSide, Proposal, TrackBox, ViewLabel

STEP: int = 200_000_000
TIMES: list[int] = [i * STEP for i in range(6)]
CAMS: list[int] = [0, 1, 2]


def make_rig(kb4: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> Kb4Rig:
    """Three forward-looking 1920x1080 KB4 cameras 10 cm apart along x (cam 1 turned 10 degrees about y)."""
    cam_from_rig: Float64[ndarray, "3 4 4"] = np.tile(np.eye(4), (3, 1, 1))
    for cam, x in enumerate((-0.1, 0.0, 0.1)):
        cam_from_rig[cam, 0, 3] = -x
    angle: float = math.radians(10.0)
    cam_from_rig[1, :3, :3] = np.array([[math.cos(angle), 0.0, -math.sin(angle)], [0.0, 1.0, 0.0], [math.sin(angle), 0.0, math.cos(angle)]])
    return Kb4Rig(
        names=("a", "b", "c"),
        size_wh=np.tile([1920.0, 1080.0], (3, 1)),
        focal=np.full((3, 2), 600.0),
        principal=np.tile([960.0, 540.0], (3, 1)),
        kb4=np.tile(np.asarray(kb4, dtype=np.float64), (3, 1)),
        cam_from_rig=cam_from_rig,
    )


def hand_at(x: float, y: float = 0.0, z: float = 0.5) -> Float64[ndarray, "21 3"]:
    """21 points spread over a 9 cm square around (x, y, z), rig frame."""
    grid: Float64[ndarray, "21 2"] = np.stack(np.meshgrid(np.linspace(-0.045, 0.045, 7), np.linspace(-0.045, 0.045, 3)), -1).reshape(-1, 2)
    offsets: Float64[ndarray, "21"] = 0.01 * np.sin(np.arange(21, dtype=np.float64))
    return np.stack([x + grid[:, 0], y + grid[:, 1], z + offsets], axis=-1)


def view(rig: Kb4Rig, cam: int, points_rig: Float64[ndarray, "21 3"], presence: float = 1.0) -> ViewKeypoints:
    """Perfect keypoints of a 3D hand in one camera."""
    points_cam: Float64[ndarray, "21 3"] = rig.to_cam(cam, points_rig)
    distance: Float64[ndarray, "21"] = np.linalg.norm(points_cam, axis=-1)
    return ViewKeypoints(cam, rig.project(cam, points_cam), np.ones(21), (distance - distance.mean()) * 1000.0, presence)


def test_kb4_maps_the_axis_to_the_principal_point_and_45_degrees_to_f_pi_over_4() -> None:
    rig: Kb4Rig = make_rig()
    pixels: Float64[ndarray, "2 2"] = rig.project(0, np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]))
    np.testing.assert_allclose(pixels, [[960.0, 540.0], [960.0 + 600.0 * math.pi / 4.0, 540.0]], atol=1e-9)


def test_kb4_unproject_inverts_project_beyond_90_degrees() -> None:
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    directions: Float64[ndarray, "4 3"] = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 1.0], [1.0, 1.0, 0.2], [1.0, 0.0, -0.1]])
    rays: Float64[ndarray, "4 3"] = rig.unproject(2, rig.project(2, directions))
    np.testing.assert_allclose(rays, directions / np.linalg.norm(directions, axis=-1, keepdims=True), atol=1e-9)


def test_fusion_recovers_the_hand_and_names_the_view_that_saw_another_hand() -> None:
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    hand: Float64[ndarray, "21 3"] = hand_at(0.02)
    fused = fuse_views(rig, [view(rig, 0, hand), view(rig, 1, hand), view(rig, 2, hand_at(0.25))])
    assert fused is not None
    np.testing.assert_allclose(fused.points_rig, hand, atol=1e-6)
    assert (fused.inliers, fused.outliers) == ((0, 1), (2,))


def test_two_views_of_different_hands_do_not_fuse() -> None:
    rig: Kb4Rig = make_rig()
    assert fuse_views(rig, [view(rig, 0, hand_at(0.0)), view(rig, 2, hand_at(0.0, y=0.2))]) is None


def test_the_box_is_the_square_around_the_grown_enclosing_circle() -> None:
    rig: Kb4Rig = make_rig()
    box, pixels = keypoint_box(rig, 1, hand_at(0.0))
    assert box is not None
    centre_x, centre_y = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    radius: float = float(np.max(np.hypot(pixels[:, 0] - centre_x, pixels[:, 1] - centre_y)))
    assert math.isclose(box.x1 - box.x0, box.y1 - box.y0)
    assert 1.19 * radius <= (box.x1 - box.x0) / 2.0 <= 1.21 * radius  # the farthest keypoint is on the circle, grown by 1.2
    assert keypoint_box(rig, 1, hand_at(0.0, z=-0.5))[0] is None  # behind the camera


class FakeKeyNet:
    """Sees the true hand of each side when a crop is aimed within ``reach_px`` of it; ``wrong`` cams see ``wrong_hand``."""

    def __init__(self, rig: Kb4Rig, truth: Callable[[int], dict[HandSide, Float64[ndarray, "21 3"]]], reach_px: float = 150.0) -> None:
        self.rig: Kb4Rig = rig
        self.truth: Callable[[int], dict[HandSide, Float64[ndarray, "21 3"]]] = truth
        self.reach_px: float = reach_px
        self.now: int = 0
        self.wrong: dict[int, Float64[ndarray, "21 3"]] = {}
        self.calls: list[CropRequest] = []

    def estimate(self, frames_rgb: Mapping[int, UInt8[ndarray, "h w 3"]], requests: Sequence[CropRequest]) -> list[ViewKeypoints | None]:
        out: list[ViewKeypoints | None] = []
        for request in requests:
            self.calls.append(request)
            hand: Float64[ndarray, "21 3"] | None = self.wrong.get(request.cam, self.truth(self.now).get(request.side))
            if hand is None:
                out.append(ViewKeypoints(request.cam, np.zeros((21, 2)), np.zeros(21), np.zeros(21), 0.0))
                continue
            seen: ViewKeypoints = view(self.rig, request.cam, hand)
            if request.points_cam is not None:
                aim: Float64[ndarray, "2"] = self.rig.project(request.cam, request.points_cam).mean(axis=0)
            else:
                assert request.box is not None
                aim = np.array([(request.box.x0 + request.box.x1) / 2.0, (request.box.y0 + request.box.y1) / 2.0])
            near: bool = bool(np.linalg.norm(seen.uv.mean(axis=0) - aim) <= self.reach_px)
            out.append(seen if near else ViewKeypoints(request.cam, seen.uv, seen.confidence, seen.d_rel_mm, 0.0))
        return out


class Frames:
    """Blank frames; tells the fake network which time it is looking at."""

    def __init__(self, network: FakeKeyNet) -> None:
        self.network: FakeKeyNet = network

    def frame(self, cam: int, t_ns: int) -> UInt8[ndarray, "h w 3"]:
        self.network.now = t_ns
        return np.zeros((4, 4, 3), dtype=np.uint8)


def run(
    truth: Callable[[int], dict[HandSide, Float64[ndarray, "21 3"]]],
    labels: dict[int, list[ViewLabel]],
    proposals: dict[int, list[Proposal]] | None = None,
    wrong: dict[int, dict[int, Float64[ndarray, "21 3"]]] | None = None,
) -> tuple[list[MegaTrackStep], FakeKeyNet]:
    """Propagate over ``TIMES``; ``labels``/``proposals``/``wrong`` are keyed by grid index."""
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    network: FakeKeyNet = FakeKeyNet(rig, truth)
    propagator: MegaTrackPropagator = MegaTrackPropagator(rig, Frames(network), network)
    steps: list[MegaTrackStep] = []
    for index, t_ns in enumerate(TIMES):
        network.wrong = (wrong or {}).get(index, {})
        steps.append(propagator.step(t_ns, CAMS, (proposals or {}).get(index, []), labels.get(index, []), steps[-1] if steps else None))
    return steps, network


def at(steps: list[MegaTrackStep], index: int, cam: int, side: HandSide) -> TrackBox:
    return next(box for box in steps[index].frame.boxes if (box.cam, box.side) == (cam, side))


def reviewer_box(rig: Kb4Rig, cam: int, side: HandSide, hand: Float64[ndarray, "21 3"]) -> ViewLabel:
    box: Box | None = keypoint_box(rig, cam, hand)[0]
    assert box is not None
    return ViewLabel(cam, side, "corrected", box, "agent", "astra")


def moving(t_ns: int) -> dict[HandSide, Float64[ndarray, "21 3"]]:
    """The left hand moves 2 cm right per grid step."""
    return {"left": hand_at(-0.05 + 0.1 * t_ns / 1e9)}


def test_reviewer_boxes_in_two_views_seed_a_track_in_every_view() -> None:
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    seed: list[ViewLabel] = [reviewer_box(rig, 0, "left", moving(0)["left"]), reviewer_box(rig, 2, "left", moving(0)["left"])]
    steps, _ = run(moving, {0: seed})
    assert [at(steps, 0, cam, "left").origin for cam in CAMS] == ["reviewed", "fused", "reviewed"]
    for index in range(1, len(TIMES)):
        for cam in CAMS:
            found: TrackBox = at(steps, index, cam, "left")
            expected: Box | None = keypoint_box(rig, cam, moving(TIMES[index])["left"])[0]
            assert found.origin == "fused" and found.box is not None and expected is not None
            assert iou(found.box, expected) > 0.95
        assert at(steps, index, 0, "right").origin == "none"


def test_the_track_follows_the_hand_through_frames_between_grid_times() -> None:
    # The hand moves 30 cm per grid step: a crop aimed at the previous grid position misses it, the substeps do not.
    def fast(t_ns: int) -> dict[HandSide, Float64[ndarray, "21 3"]]:
        return {"left": hand_at(-0.3 + 1.5 * t_ns / 1e9, z=0.8)}

    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    steps, network = run(fast, {0: [reviewer_box(rig, 0, "left", fast(0)["left"]), reviewer_box(rig, 1, "left", fast(0)["left"])]})
    assert at(steps, 2, 1, "left").origin == "fused"
    assert len({request.points_cam.tobytes() for request in network.calls if request.points_cam is not None and request.cam == 1}) > len(TIMES)


def test_a_view_that_sees_another_hand_is_flagged() -> None:
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    def still(t_ns: int) -> dict[HandSide, Float64[ndarray, "21 3"]]:
        return {"left": hand_at(0.0)}

    seed: list[ViewLabel] = [reviewer_box(rig, 0, "left", hand_at(0.0)), reviewer_box(rig, 1, "left", hand_at(0.0))]
    steps, _ = run(still, {0: seed}, wrong={2: {2: hand_at(0.0, y=0.03, z=0.45)}})
    assert [at(steps, 1, cam, "left").origin for cam in CAMS] == ["fused"] * 3
    assert [at(steps, 2, cam, "left").origin for cam in CAMS] == ["disagree"] * 3
    assert steps[2].results["left"].fusion is not None and steps[2].results["left"].fusion.outliers == (2,)
    assert at(steps, 3, 2, "left").origin == "fused"


def test_a_reviewer_box_reseeds_the_side_on_another_hand() -> None:
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    def two(t_ns: int) -> dict[HandSide, Float64[ndarray, "21 3"]]:
        return {"left": hand_at(-0.08) if t_ns < TIMES[3] else hand_at(0.12)}

    labels: dict[int, list[ViewLabel]] = {
        0: [reviewer_box(rig, 0, "left", hand_at(-0.08)), reviewer_box(rig, 1, "left", hand_at(-0.08))],
        3: [reviewer_box(rig, 1, "left", hand_at(0.12)), reviewer_box(rig, 2, "left", hand_at(0.12))],
    }
    steps, _ = run(two, labels)
    assert steps[3].results["left"].seeded
    target: Box | None = keypoint_box(rig, 0, hand_at(0.12))[0]
    found: Box | None = at(steps, 4, 0, "left").box
    assert target is not None and found is not None
    assert iou(found, target) > 0.95


def test_a_no_hand_label_hides_its_view_and_a_lost_hand_falls_back_to_egohand() -> None:
    rig: Kb4Rig = make_rig((0.05, -0.02, 0.01, -0.003))
    def gone(t_ns: int) -> dict[HandSide, Float64[ndarray, "21 3"]]:
        return {"left": hand_at(0.0)} if t_ns < TIMES[3] else {}

    seed: list[ViewLabel] = [reviewer_box(rig, 0, "left", hand_at(0.0)), reviewer_box(rig, 1, "left", hand_at(0.0))]
    proposal: Proposal = Proposal(2, "left", Box(10.0, 10.0, 110.0, 110.0), 0.9, "egohand")
    steps, _ = run(gone, {0: seed, 1: [ViewLabel(2, "left", "no_hand", None, "agent", "astra")]}, proposals={4: [proposal]})
    assert (at(steps, 1, 2, "left").origin, at(steps, 1, 2, "left").box) == ("reviewed", None)
    assert at(steps, 1, 0, "left").origin == "fused"
    assert "left" not in steps[3].hands and at(steps, 3, 0, "left").origin == "none"
    assert (at(steps, 4, 2, "left").origin, at(steps, 4, 2, "left").box) == ("detected", proposal.box)
