"""UmeTrack predictions feed the same camera observations and LM fit as KeyNet."""

from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from test_tracker import FakeDetector, Scene, _scene
from torch import Tensor

from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import letterbox_for
from handtrack.hand.pose import Side
from handtrack.reference.upstream import Crops, Pose
from handtrack.tracker import CropRequest, Tracker, TrackerConfig
from handtrack.umetrack import UmeTrackEstimator


@dataclass
class FakeCamera:
    camera_to_world_xf: Float64[ndarray, "4 4"] = field(default_factory=lambda: np.eye(4))
    f: tuple[float, float] = (240.0, 240.0)
    c: tuple[float, float] = (317.5, 239.5)
    width: int = 636
    height: int = 480

    def world_to_eye(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
        return points

    def eye_to_window(self, points: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 2"]:
        return points[:, :2] / points[:, 2:] * self.f + self.c


@dataclass
class FakePoseStage:
    scene: Scene
    requests: list[dict[int, Pose]] = field(default_factory=list)
    acquisitions: list[Float64[ndarray, "c 2 3"]] = field(default_factory=list)
    calls: list[Crops] = field(default_factory=list)

    def set_frame(self, world_from_rig: Float32[Tensor, "4 4"]) -> None:
        pass

    def pose_crops(self, poses: dict[int, Pose]) -> Crops:
        self.requests.append(poses)
        # Camera objects are opaque to the adapter; only their indices matter.
        return {hand: {camera: FakeCamera() for camera in range(4)} for hand in poses}

    def circle_crops(self, circles: Float64[ndarray, "c 2 3"], scores: Float64[ndarray, "c 2"], scale: float) -> Crops:
        self.acquisitions.append(circles.copy())
        return {hand: {camera: FakeCamera() for camera in range(len(scores)) if scores[camera, hand] > 0.5}
                for hand in range(2) if (scores[:, hand] > 0.5).any()}

    def track(self, images: list[UInt8[ndarray, "h w"]], crops: Crops) -> dict[int, Pose]:
        self.calls.append(crops)
        assert all(image.shape == (480, 636) for image in images)
        result: dict[int, Pose] = {}
        for hand in crops:
            transform: Float64[ndarray, "4 4"] = np.eye(4)
            transform[:3, :3] = self.scene.poses[hand].rotation.numpy()
            transform[:3, 3] = self.scene.poses[hand].translation.numpy() * 1000.0
            result[hand] = Pose(np.zeros(22), transform)
        return result

    def landmarks(self, pose: Pose, hand: int) -> Float32[ndarray, "21 3"]:
        return self.scene.landmarks[hand].numpy() * np.float32(1000.0)


def test_umetrack_acquires_from_circle_then_uses_our_pose_and_multiview_fit() -> None:
    scene: Scene = _scene()
    stage: FakePoseStage = FakePoseStage(scene)
    detector: FakeDetector = FakeDetector(scene, {(frame, camera, int(side)) for frame in range(4) for camera in range(4) for side in Side})
    estimator: UmeTrackEstimator = UmeTrackEstimator(stage, scene.rig, scene.letterboxes, 1.0, 0.87)
    tracker: Tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, estimator)
    images: UInt8[Tensor, "4 480 640"] = torch.zeros((4, 480, 640), dtype=torch.uint8)
    acquired = tracker.step(0, images, torch.eye(4))
    tracker.step(1, images, torch.eye(4))
    tracker.step(2, images, torch.eye(4))
    tracked = tracker.step(3, images, torch.eye(4))
    assert acquired.tracked.tolist() == [True, False]
    assert tracked.tracked.tolist() == [True, True]
    assert set(stage.calls[0]) == {0, 1}
    assert stage.requests[0] == {}
    assert set(stage.requests[3]) == {0, 1}
    np.testing.assert_allclose(stage.acquisitions[0][0, 0, :2], scene.circles[0, 0, :2].numpy() - [2.0, 0.0])
    assert len(stage.calls[3][0]) == len(stage.calls[3][1]) == 2
    torch.testing.assert_close(tracked.landmarks, scene.landmarks, atol=0.002, rtol=0.0)
    for side in Side:
        observation = tracked.observations[side]
        assert observation is not None and len(observation.views) == 2
        assert all(float(view.weights.sum()) > 0.0 for view in observation.views)


def test_detnet_misses_end_umetrack_and_reacquisition_has_no_pose_guess() -> None:
    scene: Scene = _scene()
    stage: FakePoseStage = FakePoseStage(scene)
    detector: FakeDetector = FakeDetector(scene, {(0, 0, 0), (4, 0, 0)})
    estimator: UmeTrackEstimator = UmeTrackEstimator(stage, scene.rig, scene.letterboxes, 1.0, 0.87)
    tracker: Tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector, estimator,
                               TrackerConfig(umetrack_miss_frames=2))
    images: UInt8[Tensor, "4 480 640"] = torch.zeros((4, 480, 640), dtype=torch.uint8)
    results = [tracker.step(frame, images, torch.eye(4)) for frame in range(5)]
    assert [bool(result.tracked[0]) for result in results] == [True, True, False, False, True]
    assert stage.requests[-1] == {}


def test_one_confirmed_view_resets_misses_and_excludes_unconfirmed_views() -> None:
    scene: Scene = _scene()
    detector = FakeDetector(scene, {(0, 0, 0), (2, 1, 0)})
    tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector,
        UmeTrackEstimator(FakePoseStage(scene), scene.rig, scene.letterboxes, 1.0, 0.87),
        TrackerConfig(umetrack_miss_frames=2, umetrack_presence_threshold=0.8))
    images = torch.zeros((4, 480, 640), dtype=torch.uint8)
    results = [tracker.step(frame, images, torch.eye(4)) for frame in range(5)]
    assert [bool(result.tracked[0]) for result in results] == [True, True, True, True, False]
    confirmed = results[2].observations[0]
    assert confirmed is not None and len(confirmed.views) == 1
    assert confirmed.views[0].camera.names == (scene.rig.names[1],)


def test_missing_network_pose_drops_immediately_even_with_detector_support() -> None:
    class MissingPose(FakePoseStage):
        def track(self, images: list[UInt8[ndarray, "h w"]], crops: Crops) -> dict[int, Pose]:
            return super().track(images, crops) if not self.calls else {}

    scene = _scene()
    detector = FakeDetector(scene, {(frame, camera, 0) for frame in range(2) for camera in range(4)})
    tracker = Tracker(scene.rig, scene.letterboxes, scene.model, 1.0, detector,
        UmeTrackEstimator(MissingPose(scene), scene.rig, scene.letterboxes, 1.0, 0.87))
    images = torch.zeros((4, 480, 640), dtype=torch.uint8)
    assert tracker.step(0, images, torch.eye(4)).tracked[0]
    lost = tracker.step(1, images, torch.eye(4))
    assert not lost.tracked[0] and lost.poses[0] is None


def test_show3d_adapter_uses_native_images_and_returns_unmirrored_scaled_distances() -> None:
    class Stage(FakePoseStage):
        def track(self, images: list[UInt8[ndarray, "h w"]], crops: Crops) -> dict[int, Pose]:
            assert len(images) == 2 and images[0].shape == (1280, 1024)
            assert int(images[0][123, 456]) == 231
            return {1: Pose(np.zeros(22), np.eye(4))}

        def landmarks(self, pose: Pose, hand: int) -> Float32[ndarray, "21 3"]:
            z = np.linspace(400.0, 600.0, 21, dtype=np.float32)
            return np.stack([z * 0.2, z * 0.4, z], axis=-1)

    rig = CameraRig(("/world/rig_01/cam_00", "/world/rig_01/cam_01"), torch.tensor([[1024.0, 1280.0]] * 2),
        torch.eye(4).repeat(2, 1, 1), torch.tensor([[1000.0, 1000.0]] * 2), torch.tensor([[511.5, 639.5]] * 2), None)
    letters = (letterbox_for(1024, 1280),) * 2
    native = torch.zeros((1280, 1024), dtype=torch.uint8)
    native[123, 456] = 231
    request = CropRequest(torch.tensor([0, 1]), torch.tensor([1, 1]), torch.eye(3).repeat(2, 1, 1), torch.zeros((2, 63)),
                          world_from_rig=torch.eye(4), circles=torch.tensor([[132.0, 333.25, 30.0]] * 2), native_images=(native, native))
    estimator = UmeTrackEstimator(Stage(_scene()), rig, letters, 2.0, 0.87)
    images = torch.zeros((2, 480, 640), dtype=torch.uint8)
    estimate = estimator(images, 0, request)
    torch.testing.assert_close(estimate.points_net, torch.tensor([132.0, 333.25]).expand(2, 21, 2), atol=1e-4, rtol=0.0)
    torch.testing.assert_close(estimate.d_rel_mm[:, [0, -1]], torch.tensor([[-54.772255, 54.772255]] * 2), atol=1e-3, rtol=0.0)
    assert estimate.confidence.eq(1.0).all() and estimate.presence.eq(1.0).all()
    with pytest.raises(ValueError, match="native camera images"):
        from dataclasses import replace
        estimator(images, 1, replace(request, native_images=()))
