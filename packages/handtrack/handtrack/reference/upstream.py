"""One lazy loader for the external UmeTrack checkout; no vendoring or monkey patches."""
import importlib
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import numpy as np
import torch
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from simplecv.umetrack_temp.generic_hand_model_torch import HandModelTorch

from handtrack.labels.circles import enclosing_circles
from handtrack.reference.catalog import ReferenceLabels
from handtrack.reference.geometry import FisheyeRays, ProjectionCamera, circle_crop, select_views

Crops: TypeAlias = dict[int, dict[int, ProjectionCamera]]


@dataclass(frozen=True, slots=True)
class Pose:
    joint_angles: Float64[ndarray, "22"]
    """Upstream joint angles, radians."""
    wrist_xform: Float64[ndarray, "4 4"]
    """World from wrist, mm, right-hand mirror not yet applied."""
    hand_confidence: float = 1.0
    """Upstream confidence."""


@dataclass(frozen=True, slots=True)
class UmeTrack:
    camera: ModuleType
    """lib.common.camera."""
    tracker: ModuleType
    """lib.tracker.tracker."""
    crop: ModuleType
    """lib.tracker.perspective_crop."""
    hand: ModuleType
    """lib.common.hand."""
    loader: ModuleType
    """lib.models.model_loader."""


def load_umetrack(root: Path, shim: Path | None = None) -> UmeTrack:
    """Load one checkout, rejecting collision with an already imported top-level lib package."""
    root = root.resolve()
    if not (root / "lib/tracker/tracker.py").is_file():
        raise FileNotFoundError(f"UmeTrack checkout missing: {root}")
    existing: ModuleType | None = sys.modules.get("lib")
    if existing is not None and root / "lib" not in [Path(path).resolve() for path in existing.__path__]:
        raise RuntimeError("Another checkout owns the top-level lib package; use a fresh process")
    shim = root.parent / "shim" if shim is None else shim.resolve()
    if (shim / "pytorch3d/transforms.py").is_file() and str(shim) not in sys.path:
        sys.path.insert(0, str(shim))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return UmeTrack(*(importlib.import_module(name) for name in (
        "lib.common.camera", "lib.tracker.tracker", "lib.tracker.perspective_crop", "lib.common.hand", "lib.models.model_loader")))


class PoseStage:
    """Unchanged upstream HandTracker, with typed poses at the package boundary."""

    def __init__(self, api: UmeTrack, labels: ReferenceLabels | None, weights: Path | None,
                 *, hand_model: HandModelTorch | None = None, device: torch.device | None = None) -> None:
        self.api: UmeTrack = api
        self.labels: ReferenceLabels | None = labels
        profile: HandModelTorch | None = labels.hand_model if labels is not None else hand_model
        if profile is None:
            raise ValueError("PoseStage requires a hand model")
        # A6 torch.Tensor(list) uses float32 even for index fields. Preserve it here.
        self.hand_model = api.hand.HandModel(**{field.name: getattr(profile, field.name).float() for field in fields(profile)})
        self.tracker = None
        if weights is not None:
            model = api.loader.load_pretrained_model(str(weights))
            model.eval()
            self.tracker = api.tracker.HandTracker(model, api.tracker.HandTrackerOpts())
            if device is not None:
                # Upstream auto-selects CUDA; respect the pipeline's explicit device.
                self.tracker._device = str(device)
                model.to(device)
        self.cameras: list[ProjectionCamera] = []
        self.angles: list[float] = [] if labels is None else [camera.angle for camera in labels.cameras]

    def set_frame(self, row: int) -> None:
        assert self.labels is not None
        self.cameras = [self.api.camera.Fisheye62CameraModel(spec.width, spec.height, spec.focal, spec.principal,
                       spec.coefficients, camera_to_world_xf=self.labels.camera_to_world[row, index])
                        for index, spec in enumerate(self.labels.cameras)]

    def ground_truth(self, row: int) -> dict[int, Pose]:
        assert self.labels is not None
        return {hand: Pose(self.labels.joints[row, hand], self.labels.wrists[row, hand], float(self.labels.confidence[row, hand]))
                for hand in range(2) if self.labels.confidence[row, hand] > 0}

    def pose_crops(self, poses: dict[int, Pose], *, all_views: bool = False) -> Crops:
        """L0 and L3: upstream options and upstream camera ordering, unchanged."""
        if all_views:
            # Our tracker can request a partial view. A pose spanning a crop's
            # horizon has no perspective crop; discard that view, not the hand.
            crops: Crops = {}
            for hand, pose in poses.items():
                per_hand: dict[int, ProjectionCamera] = {}
                for index, (camera, angle) in enumerate(zip(self.cameras, self.angles, strict=True)):
                    try:
                        candidate = self.api.crop.gen_crop_cameras_from_pose([camera], [angle], self.hand_model, pose, hand,
                            63, (96, 96), max_view_num=1, sort_camera_index=True, focal_multiplier=0.95,
                            mirror_right_hand=True, min_required_vis_landmarks=1)
                    except ValueError:
                        continue
                    if candidate:
                        per_hand[index] = candidate[0]
                if per_hand:
                    crops[hand] = per_hand
            return crops
        if self.tracker is not None:
            return self.tracker.gen_crop_cameras(self.cameras, self.angles, self.hand_model, poses, min_num_crops=1)
        return {hand: crops for hand, pose in poses.items() if pose.hand_confidence >= 0.5
                if (crops := self.api.crop.gen_crop_cameras_from_pose(self.cameras, self.angles, self.hand_model, pose, hand,
                    63, (96, 96), max_view_num=2, sort_camera_index=True, focal_multiplier=0.95,
                    mirror_right_hand=True, min_required_vis_landmarks=19))}

    def landmarks(self, pose: Pose, hand: int) -> Float32[ndarray, "21 3"]:
        """Upstream skinning output in mm."""
        return np.asarray(self.api.crop.landmarks_from_hand_pose(self.hand_model, pose, hand), dtype=np.float32)

    def gt_circles(self, poses: dict[int, Pose]) -> tuple[Float64[ndarray, "4 2 3"], Float64[ndarray, "4 2"]]:
        circles: Float64[ndarray, "4 2 3"] = np.full((4, 2, 3), np.nan)
        visible: Float64[ndarray, "4 2"] = np.zeros((4, 2))
        for hand, pose in poses.items():
            points: Float32[ndarray, "21 3"] = self.landmarks(pose, hand)
            for index, camera in enumerate(self.cameras):
                eye = camera.world_to_eye(points.astype(np.float64))
                pixels = camera.eye_to_window(eye)
                visible[index, hand] = np.count_nonzero((eye[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= camera.width - 1)
                                                       & (pixels[:, 1] >= 0) & (pixels[:, 1] <= camera.height - 1))
                if np.isfinite(pixels).all():
                    circles[index, hand] = enclosing_circles(pixels.astype(np.float32), np.ones(21, dtype=bool))
        return circles, visible

    def circle_crops(self, circles: Float64[ndarray, "c 2 3"], scores: Float64[ndarray, "c 2"],
                     scale: float, gt: bool = False) -> Crops:
        crops: Crops = {}
        for hand in range(2):
            per_hand: dict[int, ProjectionCamera] = {}
            eligible_scores: Float64[ndarray, "c"] = scores[:, hand].copy()
            eligible_scores[~np.isfinite(circles[:, hand]).all(axis=-1) | (circles[:, hand, 2] <= 0)] = -np.inf
            for index in select_views(eligible_scores.tolist(), 19.0 if gt else 0.5, strict=not gt):
                try:
                    parameters = circle_crop(FisheyeRays(self.cameras[index]), circles[index, hand], self.angles[index], hand, scale)
                except ValueError:
                    # An invalid detected circle provides no usable view. GT failures must be visible.
                    if gt:
                        raise
                    continue
                per_hand[index] = self.api.camera.PinholePlaneCameraModel(parameters.size, parameters.size, (parameters.focal,) * 2,
                                  ((parameters.size - 1) / 2,) * 2, [], camera_to_world_xf=parameters.camera_to_world)
            if per_hand:
                crops[hand] = per_hand
        return crops

    def track(self, images: list[UInt8[ndarray, "h w"]], crops: Crops) -> dict[int, Pose]:
        sample = self.api.tracker.InputFrame(views=[self.api.tracker.ViewData(image, camera, angle)
               for image, camera, angle in zip(images, self.cameras, self.angles, strict=True)])
        if self.tracker is None:
            raise RuntimeError("Pose inference requires pretrained weights")
        result = self.tracker.track_frame(sample, self.hand_model, crops)
        return {int(hand): Pose(np.asarray(pose.joint_angles, dtype=np.float64), np.asarray(pose.wrist_xform, dtype=np.float64), float(pose.hand_confidence))
                for hand, pose in result.hand_poses.items()}

    def crop_diagnostics(self, poses: dict[int, Pose], circles: Float64[ndarray, "4 2 3"], visible: Float64[ndarray, "4 2"]) -> tuple[list[float], list[float]]:
        """Unit-circle vs upstream 63-point focal ratios and optical-axis angles (degrees)."""
        ratios: list[float] = []
        angles: list[float] = []
        crops: Crops = self.pose_crops(poses)
        for hand, per_hand in crops.items():
            for index, upstream in per_hand.items():
                if visible[index, hand] < 19:
                    continue
                unit = circle_crop(FisheyeRays(self.cameras[index]), circles[index, hand], self.angles[index], hand, 1.0)
                ratios.append(float(upstream.f[0]) / unit.focal)
                cosine: float = float(np.dot(upstream.camera_to_world_xf[:3, 2], unit.camera_to_world[:3, 2]))
                angles.append(float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))))
        return ratios, angles
