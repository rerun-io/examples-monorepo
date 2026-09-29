"""UmeTrack's pretrained pose network as observations for our tracker and LM fit.

The upstream network sees at most two perspective views per hand (its trained
limit). Our tracker supplies its extrapolated pose and chooses the views. On
acquisition, the frozen circle calibration maps DetNet circles to crops. All
hands share one inference batch; no view is inferred independently. Upstream
temporal memory is reset on each call: only our tracker owns tracking history.
"""

import atexit
import os
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, world_to_cameras
from handtrack.geometry.letterbox import Letterbox
from handtrack.hand.pose import HandPose, Side
from handtrack.labels.keypoint_input import relative_distances
from handtrack.reference.upstream import Crops, Pose, PoseStage
from handtrack.tracker import CropRequest, KeypointEstimate

_DUMP: dict[tuple[int, int], np.ndarray] = {}
if os.environ.get("HT_UMETRACK_DUMP"):
    atexit.register(lambda: np.savez(os.environ["HT_UMETRACK_DUMP"], keys=np.array(list(_DUMP), dtype=np.int64).reshape(-1, 2),
                                     points=np.stack(list(_DUMP.values())) if _DUMP else np.zeros((0, 21, 3), np.float32)))


@runtime_checkable
class PoseNetwork(Protocol):
    """Perspective crop and pose inference boundary, replaceable on CPU."""

    def set_frame(self, world_from_rig: Float32[Tensor, "4 4"]) -> None: ...
    def pose_crops(self, poses: dict[int, Pose]) -> Crops: ...
    def circle_crops(self, circles: Float64[ndarray, "c 2 3"], scores: Float64[ndarray, "c 2"], scale: float) -> Crops: ...
    def track(self, images: list[UInt8[ndarray, "h w"]], crops: Crops) -> dict[int, Pose]: ...
    def landmarks(self, pose: Pose, hand: int) -> Float32[ndarray, "21 3"]: ...


class RigPoseNetwork:
    """Use reference crop/inference code with our rig; never read ground-truth hand poses."""

    def __init__(self, stage: PoseStage, rig: CameraRig, angles: tuple[float, ...]) -> None:
        self.stage: PoseStage = stage
        self.rig: CameraRig = rig
        self.stage.angles = list(angles)

    def set_frame(self, world_from_rig: Float32[Tensor, "4 4"]) -> None:
        transforms: Float64[ndarray, "c 4 4"] = (
            world_from_rig.numpy().astype(np.float64)[None] @ np.linalg.inv(self.rig.cam_from_rig.numpy().astype(np.float64))
        )
        transforms[:, :3, 3] *= 1000.0
        self.stage.cameras = []
        for camera in range(len(self.rig.names)):
            width, height = self.rig.image_size[camera].tolist()
            coefficients: list[float] = [] if self.rig.fisheye62 is None else self.rig.fisheye62[camera, [0, 1, 2, 3, 6, 7, 4, 5]].tolist()
            constructor = self.stage.api.camera.PinholePlaneCameraModel if self.rig.fisheye62 is None else self.stage.api.camera.Fisheye62CameraModel
            self.stage.cameras.append(constructor(int(width), int(height), tuple(self.rig.focal[camera].tolist()),
                tuple(self.rig.principal[camera].tolist()), coefficients, camera_to_world_xf=transforms[camera]))

    def pose_crops(self, poses: dict[int, Pose]) -> Crops:
        return self.stage.pose_crops(poses, all_views=True)

    def circle_crops(self, circles: Float64[ndarray, "c 2 3"], scores: Float64[ndarray, "c 2"], scale: float) -> Crops:
        return self.stage.circle_crops(circles, scores, scale)

    def track(self, images: list[UInt8[ndarray, "h w"]], crops: Crops) -> dict[int, Pose]:
        if self.stage.tracker is not None:
            self.stage.tracker.reset_history()
        return self.stage.track(images, crops)

    def landmarks(self, pose: Pose, hand: int) -> Float32[ndarray, "21 3"]:
        return self.stage.landmarks(pose, hand)


class UmeTrackEstimator:
    """Project a multi-view network pose into KeypointEstimate; presence is checked by DetNet in Tracker."""

    def __init__(self, stage: PoseNetwork, rig: CameraRig, letterboxes: tuple[Letterbox, ...], phi: float, circle_scale: float) -> None:
        self.stage: PoseNetwork = stage
        self.rig: CameraRig = rig
        self.letterboxes: tuple[Letterbox, ...] = letterboxes
        self.phi: float = phi
        self.circle_scale: float = circle_scale

    def __call__(self, images: UInt8[Tensor, "c 480 640"], frame: int, request: CropRequest) -> KeypointEstimate:
        if request.world_from_rig is None or request.circles is None:
            raise ValueError("UmeTrack needs the tracker's headset pose and circles")
        native: list[UInt8[ndarray, "h w"]] = []
        if request.native_images:
            native = [np.ascontiguousarray(image.cpu().numpy()) for image in request.native_images]
        else:
            for camera, letterbox in enumerate(self.letterboxes):
                if letterbox.quarter_turn_cw or letterbox.scale != 1.0:
                    raise ValueError("UmeTrack needs native camera images for resized/rotated inputs")
                start: int = int(letterbox.pad_x)
                native.append(np.ascontiguousarray(images[camera, :, start:start + letterbox.source_width].cpu().numpy()))
        self.stage.set_frame(request.world_from_rig)
        guesses: dict[int, Pose] = {}
        for side in Side:
            guess: HandPose | None = request.poses[side]
            if guess is not None:
                wrist: Float64[ndarray, "4 4"] = np.eye(4)
                wrist[:3, :3] = guess.rotation.numpy()
                wrist[:3, 3] = guess.translation.numpy().astype(np.float64) * 1000.0
                guesses[int(side)] = Pose(guess.joint_angles.numpy().astype(np.float64), wrist)
        crops: Crops = self.stage.pose_crops(guesses)
        circles: Float64[ndarray, "c 2 3"] = np.full((len(self.letterboxes), 2, 3), np.nan)
        scores: Float64[ndarray, "c 2"] = np.zeros((len(self.letterboxes), 2))
        for index, (camera, side) in enumerate(zip(request.camera.tolist(), request.side.tolist(), strict=True)):
            if side not in guesses:
                letterbox = self.letterboxes[camera]
                circles[camera, side, :2] = letterbox.from_net(request.circles[index, :2]).numpy()
                circles[camera, side, 2] = float(request.circles[index, 2]) / letterbox.scale
                scores[camera, side] = 1.0
        crops.update(self.stage.circle_crops(circles, scores, self.circle_scale))
        # Preserve the tracker's camera choice and upstream ascending camera order.
        crops = {hand: {camera: crop for camera, crop in sorted(per_hand.items())
                        if bool(((request.side == hand) & (request.camera == camera)).any())}
                 for hand, per_hand in crops.items()}
        crops = {hand: per_hand for hand, per_hand in crops.items() if per_hand}
        poses: dict[int, Pose] = self.stage.track(native, crops) if crops else {}
        count: int = len(request.camera)
        points_net: Float32[Tensor, "n 21 2"] = torch.zeros((count, 21, 2))
        distances: Float32[Tensor, "n 21"] = torch.zeros((count, 21))
        confidence: Float32[Tensor, "n 21"] = torch.zeros((count, 21))
        presence: Float32[Tensor, "n"] = torch.zeros(count)
        for hand, pose in poses.items():
            points_world: Float32[Tensor, "21 3"] = torch.from_numpy(self.stage.landmarks(pose, hand)) / 1000.0
            _DUMP.setdefault((frame, hand), points_world.numpy().copy())  # debug: HT_UMETRACK_DUMP=<npz> keeps raw network landmarks
            if not bool(torch.isfinite(points_world).all()):
                continue
            points_cam: Float32[Tensor, "c 21 3"] = world_to_cameras(self.rig, request.world_from_rig, points_world)
            pixels: Float32[Tensor, "c 21 2"] = project(self.rig, points_cam)
            valid: Bool[Tensor, "c 21"] = in_front(points_cam) & inside_image(self.rig, pixels) & torch.isfinite(pixels).all(dim=-1)
            relative: Float32[Tensor, "c 21"] = relative_distances(points_cam, torch.full((len(self.letterboxes),), self.phi))
            for index in torch.nonzero(request.side == hand).flatten().tolist():
                camera = int(request.camera[index])
                points_net[index] = torch.nan_to_num(self.letterboxes[camera].to_net(pixels[camera]))
                distances[index] = relative[camera]
                confidence[index] = valid[camera].float()
                presence[index] = float(bool(valid[camera].any()))
        return KeypointEstimate(points_net, distances, presence, confidence, uses_detnet_presence=True)
