"""The tracker's state machine with fake networks: acquisition, round robin, the two-view choice and the track end.

The scene: four 636x480 pinhole cameras at the rig origin, yawed -0.9, -0.3, 0.3 and 0.9 rad, a still headset, and the
generic hand model with both hands held still in front. The fake detector reports the ground-truth circle on scripted
frames; the fake KeyNet returns the exact projected ground truth with a scripted presence. The fit is the real one.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import pytest
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch
from torch import Tensor

import handtrack.tracker as tracker_module
from handtrack.fit.observations import HandObservation
from handtrack.fit.pose_fit import FitConfig, FitResult
from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.geometry.letterbox import Letterbox, letterbox_for
from handtrack.hand.pose import HandPose, Side, generic_hand_model, landmarks
from handtrack.labels.circles import enclosing_circles, square_boxes
from handtrack.labels.crops import apply_affine, crop_boxes, crop_from_net
from handtrack.labels.heatmaps import render_distance, render_heatmaps
from handtrack.labels.keypoint_input import relative_distances
from handtrack.models.detnet import Detections
from handtrack.models.keynet import KeyNetF, KeyNetOutput
from handtrack.results import BoxSource
from handtrack.tracker import ROBUST_TRACKER_CONFIG, CropRequest, FrameResult, KeyNetEstimator, KeypointEstimate, Tracker, TrackerConfig

YAWS: tuple[float, ...] = (-0.9, -0.3, 0.3, 0.9)
WRISTS: tuple[tuple[float, float, float], tuple[float, float, float]] = ((-0.3, 0.0, 0.3), (0.3, 0.0, 0.3))
"""Left and right wrist positions; the fingers point towards the middle."""
VISIBLE: tuple[list[int], list[int]] = ([21, 21, 14, 0], [0, 15, 21, 21])
"""Keypoints inside each camera, per hand, in this scene."""


def _rotation(axis: int, angle: float) -> Float32[Tensor, "3 3"]:
    c, s = math.cos(angle), math.sin(angle)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    matrix: Float32[Tensor, "3 3"] = torch.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


def _rig() -> CameraRig:
    cam_from_rig: Float32[Tensor, "4 4 4"] = torch.eye(4).repeat(4, 1, 1)
    for camera, yaw in enumerate(YAWS):
        cam_from_rig[camera, :3, :3] = _rotation(1, yaw).T
    return CameraRig(
        names=tuple(f"/world/rig_00/cam_0{camera}" for camera in range(4)),
        image_size=torch.tensor([[636.0, 480.0]] * 4),
        cam_from_rig=cam_from_rig,
        focal=torch.tensor([[240.0, 240.0]] * 4),
        principal=torch.tensor([[317.5, 239.5]] * 4),
        fisheye62=None,
    )


@dataclass(frozen=True, slots=True)
class Scene:
    """Ground truth of the still scene, in the net frame of every camera."""

    rig: CameraRig
    letterboxes: tuple[Letterbox, ...]
    model: HandModelTorch
    poses: tuple[HandPose, HandPose]
    landmarks: Float32[Tensor, "2 21 3"]
    net_xy: Float32[Tensor, "4 2 21 2"]
    points_cam: Float32[Tensor, "4 2 21 3"]
    visible: Int64[Tensor, "4 2"]
    circles: Float32[Tensor, "4 2 3"]


def _scene() -> Scene:
    rig: CameraRig = _rig()
    letterboxes: tuple[Letterbox, ...] = tuple(letterbox_for(636, 480) for _ in range(4))
    model: HandModelTorch = generic_hand_model()
    poses: tuple[HandPose, HandPose] = (
        HandPose(_rotation(0, 1.2), torch.tensor(WRISTS[0]), torch.zeros(22)),
        HandPose(_rotation(0, 1.2), torch.tensor(WRISTS[1]), torch.zeros(22)),
    )
    points: Float32[Tensor, "2 21 3"] = torch.stack([landmarks(model, poses[side], side) for side in Side])
    points_cam: Float32[Tensor, "4 2 21 3"] = torch.stack([world_to_cameras(rig, torch.eye(4), points[side]) for side in Side], dim=1)
    pixels: Float32[Tensor, "4 2 21 2"] = torch.stack([project(rig, points_cam[:, side]) for side in Side], dim=1)
    net_xy: Float32[Tensor, "4 2 21 2"] = torch.stack([letterboxes[camera].to_net(pixels[camera]) for camera in range(4)])
    inside: Bool[Tensor, "4 2 21"] = torch.stack([in_front(points_cam[:, side]) & inside_image(rig, pixels[:, side]) for side in Side], dim=1)
    circles: Float32[Tensor, "4 2 3"] = torch.from_numpy(enclosing_circles(net_xy.numpy(), in_front(points_cam).numpy()))
    return Scene(rig, letterboxes, model, poses, points, net_xy, points_cam, inside.sum(-1), circles)


@dataclass(slots=True)
class FakeDetector:
    """Reports the ground-truth circle of the scripted (frame, camera, side) detections; records every call."""

    scene: Scene
    detections: set[tuple[int, int, int]] = field(default_factory=set)
    calls: list[tuple[int, int]] = field(default_factory=list)

    def __call__(self, images: UInt8[Tensor, "b 480 640"], frame: int, cameras: Int64[Tensor, "b"]) -> Detections:
        circle: Float32[Tensor, "b 2 3"] = self.scene.circles[cameras]
        probability: Float32[Tensor, "b 2"] = torch.tensor(
            [[0.9 if (frame, int(camera), side) in self.detections else 0.1 for side in Side] for camera in cameras]
        )
        self.calls.extend((frame, int(camera)) for camera in cameras)
        return Detections(circle=circle, probability=probability, present=probability > 0.5, box=square_boxes(circle))


@dataclass(slots=True)
class FakeKeyNet:
    """Returns the exact ground-truth keypoints of the requested (camera, side); presence 0.9 unless scripted."""

    scene: Scene
    presence: Callable[[int, int, int], float] = lambda frame, camera, side: 0.9
    calls: list[tuple[int, CropRequest]] = field(default_factory=list)

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        self.calls.append((frame, request))
        return KeypointEstimate(
            points_net=self.scene.net_xy[request.camera, request.side],
            d_rel_mm=relative_distances(self.scene.points_cam[request.camera, request.side], torch.ones(len(request.camera))),
            presence=torch.tensor([self.presence(frame, int(camera), int(side)) for camera, side in zip(request.camera, request.side, strict=True)]),
            confidence=torch.ones((len(request.camera), 21)),
        )


def _run(tracker: Tracker, frames: int) -> list[FrameResult]:
    images: UInt8[Tensor, "4 480 640"] = torch.zeros((4, 480, 640), dtype=torch.uint8)
    return [tracker.step(frame, images, torch.eye(4)) for frame in range(frames)]


def _views(keynet: FakeKeyNet, frame: int, side: Side) -> list[int]:
    return sorted(
        int(camera) for f, request in keynet.calls if f == frame for camera, s in zip(request.camera, request.side, strict=True) if int(s) == side
    )


def test_scene_counts() -> None:
    assert _scene().visible.T.tolist() == [VISIBLE[0], VISIBLE[1]]


def test_detnet_runs_round_robin_while_a_hand_is_untracked() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene)
    keynet: FakeKeyNet = FakeKeyNet(scene)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet), 6)
    assert detector.calls == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 0), (5, 1)]
    assert [result.detnet_camera for result in results] == [0, 1, 2, 3, 0, 1]
    assert keynet.calls == []
    assert not any(bool(result.tracked.any()) for result in results)


def test_acquisition_then_stereo_tracking_then_no_detnet() -> None:
    scene: Scene = _scene()
    # The left hand is found in cam_00 on frame 0; the right hand in cam_01 on frame 1 (its round-robin camera).
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT), (1, 1, Side.RIGHT)})
    keynet: FakeKeyNet = FakeKeyNet(scene)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet), 4)

    assert results[0].tracked.tolist() == [True, False]
    assert results[0].box_source[:, Side.LEFT].tolist() == [BoxSource.DETNET, 0, 0, 0]
    assert _views(keynet, 0, Side.LEFT) == [0]
    first: CropRequest = keynet.calls[0][1]
    assert bool((first.keypoint_input == 0).all()), "an acquisition runs KeyNet with the zero keypoint input"
    torch.testing.assert_close(results[0].landmarks[Side.LEFT], scene.landmarks[Side.LEFT], atol=2e-3, rtol=0.0)

    # Frame 1: the left hand's boxes come from its pose in every camera that sees it; KeyNet runs on its two best views.
    assert results[1].tracked.tolist() == [True, True]
    assert results[1].box_source[:, Side.LEFT].tolist() == [BoxSource.TRACKED, BoxSource.TRACKED, BoxSource.TRACKED, 0]
    assert _views(keynet, 1, Side.LEFT) == [0, 1]
    assert _views(keynet, 1, Side.RIGHT) == [1]
    tracked_inputs: Float32[Tensor, "n 63"] = keynet.calls[1][1].keypoint_input[keynet.calls[1][1].side == Side.LEFT]
    assert bool((tracked_inputs != 0).any(dim=-1).all()), "a tracked hand's keypoint input comes from its extrapolated pose"
    assert results[1].detnet_camera == 1

    # Frame 2 on: both hands tracked, no DetNet; the right hand in its two best views.
    assert results[2].detnet_camera == -1 and results[3].detnet_camera == -1
    assert detector.calls == [(0, 0), (1, 1)]
    assert _views(keynet, 2, Side.RIGHT) == [2, 3]
    assert _views(keynet, 3, Side.LEFT) == [0, 1]
    torch.testing.assert_close(results[3].landmarks, scene.landmarks, atol=1e-3, rtol=0.0)
    assert bool(torch.isfinite(results[3].fit_energy).all())


def test_low_presence_drops_the_view_then_the_track() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})

    def presence(frame: int, camera: int, side: int) -> float:
        if side == Side.LEFT and frame == 2 and camera == 1:
            return 0.3  # one view below 0.5: it stays out of the fit
        if side == Side.LEFT and frame == 3:
            return 0.2  # every view below 0.5: the track ends
        return 0.9

    keynet: FakeKeyNet = FakeKeyNet(scene, presence=presence)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet), 5)
    left_observation = results[2].observations[Side.LEFT]
    assert left_observation is not None and len(left_observation.views) == 1
    assert results[2].tracked[Side.LEFT]
    assert _views(keynet, 3, Side.LEFT) == [0, 1]
    assert not results[3].tracked[Side.LEFT]
    assert results[3].poses[Side.LEFT] is None
    assert bool(torch.isnan(results[3].landmarks[Side.LEFT]).all())
    torch.testing.assert_close(results[3].presence[:2, Side.LEFT], torch.tensor([0.2, 0.2]))
    # Dropped: no tracked boxes on the next frame, and DetNet looks for it again (cam_00 was frame 0; frames 1-3 kept cycling).
    assert results[4].box_source[:, Side.LEFT].tolist() == [0, 0, 0, 0]
    assert [result.detnet_camera for result in results] == [0, 1, 2, 3, 0]


def test_end_on_view_rejection_drops_a_track_left_with_one_view() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})

    def presence(frame: int, camera: int, side: int) -> float:
        return 0.3 if side == Side.LEFT and frame == 2 and camera == 1 else 0.9  # one of two views rejected at frame 2

    keynet: FakeKeyNet = FakeKeyNet(scene, presence=presence)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet,
                                              TrackerConfig(end_on_view_rejection=True)), 4)
    assert results[1].tracked[Side.LEFT] and not results[2].tracked[Side.LEFT]  # the one-view frame ends the track
    assert results[3].box_source[:, Side.LEFT].tolist() == [0, 0, 0, 0]  # DetNet looks for it again


def test_rescue_rejected_view_keeps_a_track_whose_recut_view_passes() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    asked: list[int] = []

    def presence(frame: int, camera: int, side: int) -> float:
        if side == Side.LEFT and frame == 2 and camera == 1:
            asked.append(frame)
            return 0.3 if len(asked) == 1 else 0.9  # the planned crop misses; the crop re-cut around the one-view fit holds the hand
        return 0.9

    keynet: FakeKeyNet = FakeKeyNet(scene, presence=presence)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet,
                                              TrackerConfig(end_on_view_rejection=True, rescue_rejected_view=True)), 4)
    assert all(bool(result.tracked[Side.LEFT]) for result in results)
    frame_two: list[CropRequest] = [request for frame, request in keynet.calls if frame == 2]
    assert len(frame_two) == 2, "the rescue is one more KeyNet call on the same frame"
    assert sorted(frame_two[1].camera[frame_two[1].side == Side.LEFT].tolist()) == [0, 1]
    left_observation = results[2].observations[Side.LEFT]
    assert left_observation is not None and len(left_observation.views) == 2
    torch.testing.assert_close(results[2].presence[:2, Side.LEFT], torch.tensor([0.9, 0.9]))
    torch.testing.assert_close(results[2].landmarks[Side.LEFT], scene.landmarks[Side.LEFT], atol=2e-3, rtol=0.0)


def test_rescue_rejected_view_still_ends_a_track_whose_recut_view_fails() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})

    def presence(frame: int, camera: int, side: int) -> float:
        return 0.3 if side == Side.LEFT and frame == 2 and camera == 1 else 0.9

    keynet: FakeKeyNet = FakeKeyNet(scene, presence=presence)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet,
                                              TrackerConfig(end_on_view_rejection=True, rescue_rejected_view=True)), 4)
    assert results[1].tracked[Side.LEFT] and not results[2].tracked[Side.LEFT]
    assert len([frame for frame, _ in keynet.calls if frame == 2]) == 2  # tried once more, then ended
    assert results[3].box_source[:, Side.LEFT].tolist() == [0, 0, 0, 0]


def test_a_detection_that_keynet_rejects_is_not_tracked() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    keynet: FakeKeyNet = FakeKeyNet(scene, presence=lambda frame, camera, side: 0.1)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet), 2)
    assert results[0].box_source[0, Side.LEFT] == BoxSource.DETNET
    assert not bool(results[0].tracked.any())
    assert results[1].detnet_camera == 1


def test_a_lost_headset_pose_resets_the_tracks() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    tracker: Tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, FakeKeyNet(scene))
    images: UInt8[Tensor, "4 480 640"] = torch.zeros((4, 480, 640), dtype=torch.uint8)
    assert tracker.step(0, images, torch.eye(4)).tracked[Side.LEFT]
    lost: FrameResult = tracker.step(1, images, torch.full((4, 4), torch.nan))
    assert not bool(lost.tracked.any()) and lost.detnet_camera == -1
    assert tracker.step(2, images, torch.eye(4)).detnet_camera == 1


class ScriptedKeyNet(KeyNetF):
    """KeyNet whose heatmaps peak exactly at scripted crop points."""

    def __init__(self, points_crop: Float32[Tensor, "n 21 2"], d_rel_mm: Float32[Tensor, "n 21"], presence_logit: Float32[Tensor, "n"]) -> None:
        super().__init__()
        self.points_crop: Float32[Tensor, "n 21 2"] = points_crop
        self.d_rel_mm: Float32[Tensor, "n 21"] = d_rel_mm
        self.scripted_logit: Float32[Tensor, "n"] = presence_logit

    def forward(self, crop: Float32[Tensor, "b 1 96 96"], keypoints: Float32[Tensor, "b 63"]) -> KeyNetOutput:
        return KeyNetOutput(heatmaps=render_heatmaps(self.points_crop), distance=render_distance(self.d_rel_mm), presence_logit=self.scripted_logit)


def test_keynet_estimator_undoes_the_crop_and_the_right_hand_mirror() -> None:
    generator: torch.Generator = torch.Generator().manual_seed(3)
    points_net: Float32[Tensor, "2 21 2"] = torch.tensor([300.0, 200.0]) + 60.0 * torch.rand((2, 21, 2), generator=generator)
    circles: Float32[Tensor, "2 3"] = torch.from_numpy(enclosing_circles(points_net.numpy(), torch.ones((2, 21), dtype=torch.bool).numpy()))
    maps: Float32[Tensor, "2 3 3"] = crop_from_net(crop_boxes(circles), torch.tensor([False, True]))
    d_rel: Float32[Tensor, "2 21"] = torch.linspace(-60.0, 60.0, 42).reshape(2, 21)
    model: ScriptedKeyNet = ScriptedKeyNet(apply_affine(maps, points_net), d_rel, torch.tensor([2.0, -2.0]))
    request: CropRequest = CropRequest(
        camera=torch.tensor([1, 2]), side=torch.tensor([0, 1]), crop_from_net=maps, keypoint_input=torch.zeros((2, 63))
    )
    estimate: KeypointEstimate = KeyNetEstimator(model)(torch.zeros((4, 480, 640), dtype=torch.uint8), 0, request)
    torch.testing.assert_close(estimate.points_net, points_net, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(estimate.d_rel_mm, d_rel, atol=1e-2, rtol=0.0)
    torch.testing.assert_close(estimate.presence, torch.tensor([2.0, -2.0]).sigmoid())


def test_a_hand_outside_every_image_is_dropped_without_keynet() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    keynet: FakeKeyNet = FakeKeyNet(scene)
    tracker: Tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet)
    images: UInt8[Tensor, "4 480 640"] = torch.zeros((4, 480, 640), dtype=torch.uint8)
    assert tracker.step(0, images, torch.eye(4)).tracked[Side.LEFT]
    turned: Float32[Tensor, "4 4"] = torch.eye(4)
    turned[:3, :3] = _rotation(1, math.pi)  # the headset looks the other way: the hand is behind every camera
    away: FrameResult = tracker.step(1, images, turned)
    assert not away.tracked[Side.LEFT]
    assert away.box_source[:, Side.LEFT].tolist() == [0, 0, 0, 0]
    assert _views(keynet, 1, Side.LEFT) == []
    assert away.detnet_camera == 1, "the right hand was still untracked, so DetNet ran; the left hand is looked for from frame 2"


def test_a_keypoint_with_an_empty_heatmap_is_left_out_of_the_fit() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    honest: FakeKeyNet = FakeKeyNet(scene)

    def dead_wrist(images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        estimate: KeypointEstimate = honest(images, frame, request)
        points: Float32[Tensor, "n 21 2"] = estimate.points_net.clone()
        points[:, 5] = 0.0  # an empty wrist heatmap decodes to the crop corner
        confidence: Float32[Tensor, "n 21"] = estimate.confidence.clone()
        confidence[:, 5] = 0.0
        return KeypointEstimate(points, estimate.d_rel_mm, estimate.presence, confidence)

    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, dead_wrist), 3)
    observation = results[2].observations[Side.LEFT]
    assert observation is not None and all(float(view.weights[5]) == 0.0 and float(view.weights.sum()) == 20.0 for view in observation.views)
    torch.testing.assert_close(results[2].landmarks[Side.LEFT], scene.landmarks[Side.LEFT], atol=3e-3, rtol=0.0)


def test_a_fit_out_of_reach_ends_the_track() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    config: TrackerConfig = TrackerConfig(max_reach_m=0.2)  # the scene's wrists are 0.42 m from the headset
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, FakeKeyNet(scene), config), 2)
    assert results[0].box_source[0, Side.LEFT] == BoxSource.DETNET
    assert not results[0].tracked[Side.LEFT] and results[0].poses[Side.LEFT] is None
    assert results[1].detnet_camera == 1


def test_crop_refinement_recuts_an_off_centre_acquisition_crop() -> None:
    scene: Scene = _scene()
    offset: Scene = replace(scene, circles=scene.circles + torch.tensor([30.0, 0.0, 0.0]))  # DetNet's circle 30 px off the hand
    for shift, calls in ((None, 1), (0.1, 2)):
        detector: FakeDetector = FakeDetector(offset, detections={(0, 0, Side.LEFT)})
        keynet: FakeKeyNet = FakeKeyNet(scene)
        result: FrameResult = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet, TrackerConfig(refine_shift=shift)), 1)[0]
        assert len(keynet.calls) == calls
        assert result.tracked[Side.LEFT]
    refined_crop: torch.Tensor = keynet.calls[1][1].crop_from_net[0]
    first_crop: torch.Tensor = keynet.calls[0][1].crop_from_net[0]
    assert not torch.allclose(refined_crop, first_crop), "the second pass cuts a new crop around the fitted pose"


@pytest.mark.parametrize("config", [TrackerConfig(min_keypoint_confidence=2.0), TrackerConfig(fit=FitConfig(init_iterations=0))])
def test_unconverged_acquisition_is_not_tracked(config: TrackerConfig) -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT), (1, 1, Side.LEFT)})
    keynet: FakeKeyNet = FakeKeyNet(scene)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet, config), 2)
    assert not any(bool(result.tracked.any()) for result in results)
    assert all(result.poses == (None, None) for result in results)
    assert all(bool(torch.isnan(result.fit_energy).all()) for result in results)
    assert all(bool((request.keypoint_input == 0).all()) for _, request in keynet.calls)


def test_stationary_acquires_and_finite_unconverged_tracking_keeps_its_pose(monkeypatch: pytest.MonkeyPatch) -> None:
    scene: Scene = _scene()

    def fitted(
        model: HandModelTorch, phi: float, hands: list[HandObservation], previous: list[HandPose | None], config: FitConfig,
    ) -> list[FitResult]:
        return [
            FitResult(scene.poses[hand.side], 0.0, 0.0, 0.0, 0.0, 1, prior is None, "stationary" if prior is None else "iterations")
            for hand, prior in zip(hands, previous, strict=True)
        ]

    monkeypatch.setattr(tracker_module, "fit_pose", fitted)
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    keynet: FakeKeyNet = FakeKeyNet(scene)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet), 2)
    assert [result.tracked.tolist() for result in results] == [[True, False], [True, False]]
    assert _views(keynet, 1, Side.LEFT) == [0, 1]


@pytest.mark.parametrize("max_views", [-1, 0, 3])
def test_tracker_rejects_unsupported_view_counts(max_views: int) -> None:
    with pytest.raises(ValueError, match="max_views"):
        TrackerConfig(max_views=max_views)


@pytest.mark.parametrize(("acquisition_presence", "first_reported"), [(0.95, 1), (0.6, 2)])
def test_unsure_acquisitions_wait_longer_before_they_are_reported(acquisition_presence: float, first_reported: int) -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    keynet: FakeKeyNet = FakeKeyNet(scene, presence=lambda frame, camera, side: acquisition_presence if frame == 0 else 0.95)
    config: TrackerConfig = TrackerConfig(confirm_frames=1, confirm_frames_unsure=2, confident_presence=0.9)
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, keynet, config), 4)
    assert [bool(result.tracked[Side.LEFT]) for result in results] == [frame >= first_reported for frame in range(4)]


def test_the_robust_preset_tracks_the_fake_scene() -> None:
    scene: Scene = _scene()
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    results: list[FrameResult] = _run(Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, FakeKeyNet(scene), ROBUST_TRACKER_CONFIG), 6)
    # confirm_frames 2: reported from the 3rd frame; the damped guess still converges onto the exact keypoints
    assert [bool(result.tracked[Side.LEFT]) for result in results] == [False, False, True, True, True, True]
    torch.testing.assert_close(results[5].landmarks[Side.LEFT], scene.landmarks[Side.LEFT], atol=2e-3, rtol=0.0)


@pytest.mark.parametrize("target", ["previous", "guess"])
def test_temporal_target_chooses_the_fits_start(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    scene: Scene = _scene()
    seen: list[HandPose | None] = []
    real_fit = tracker_module.fit_pose

    def recording(model: HandModelTorch, phi: float, hands: list[HandObservation], previous: list[HandPose | None], config: FitConfig) -> list[FitResult]:
        seen[:] = list(previous)
        return real_fit(model, phi, hands, previous, config)

    monkeypatch.setattr(tracker_module, "fit_pose", recording)
    detector: FakeDetector = FakeDetector(scene, detections={(0, 0, Side.LEFT)})
    tracker: Tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, FakeKeyNet(scene), TrackerConfig(temporal_target=target))
    images: UInt8[Tensor, "4 480 640"] = torch.zeros((4, 480, 640), dtype=torch.uint8)
    for frame in range(3):
        tracker.step(frame, images, torch.eye(4))
    # frame 2 was fitted from θ(t−1) (now ``before``) or from its planning guess (now ``guess``, an extrapolated pose object)
    history = tracker.history[Side.LEFT]
    assert history.guess is not None and history.guess is not history.before
    assert seen[0] is (history.before if target == "previous" else history.guess)
