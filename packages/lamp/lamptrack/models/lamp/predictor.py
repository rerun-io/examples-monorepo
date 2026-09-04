"""PoseKit-backed streaming wrapper around the vendored LAMP tracker."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from posekit.models import AnnotatedDetectorConfig, AnnotatedPose2dConfig, PersonDetector, TopDownPose2d
from posekit.models.rtdetr import RtDetrDetectorConfig
from posekit.models.vitpose import VitPoseConfig
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.skeletons import COCO_17
from torch import Tensor

from lamptrack.cameras import PerCameraCalibration, RigCamera, gravity_aligned_world_transform
from lamptrack.third_party.lamp.core.types import Detection2D
from lamptrack.third_party.lamp.models.lifter import Lifter, LifterSettings, SnippetData
from lamptrack.third_party.lamp.tracking.tracker import LampTracker as UpstreamLampTracker

_NUM_VIEWS = 4
_NUM_COCO_KEYPOINTS = 17
_NUM_SMPL_JOINTS = 24
_NUM_BETAS = 10
_DEFAULT_CHECKPOINT = Path(__file__).parents[3] / "data" / "checkpoints" / "lamp_smpl_aria_gen2.pt"
_DEFAULT_SMPL_MODEL = Path(__file__).parents[3] / "data" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"


@dataclass(frozen=True, slots=True)
class Frameset:
    """One synchronized capture from the four-camera rig."""

    timestamp_ns: int
    images: UInt8[ndarray, "v h w 3"]
    world_T_rig: Float64[ndarray, "4 4"]


@dataclass(frozen=True, slots=True)
class PersonState:
    """Smoothed LAMP output for one person's current temporal window."""

    track_id: int
    timestamps_ns: Int64[ndarray, "t"]
    joints_world: Float32[ndarray, "t 24 3"]
    betas: Float32[ndarray, "10"]
    root_T: Float32[ndarray, "t 4 4"]
    rotations: Float32[ndarray, "t 24 3 3"]


@dataclass(frozen=True, slots=True)
class LampTimings:
    """Per-stage wall-clock time for one :meth:`LampTracker.step` call."""

    detector_ms: float = 0.0
    pose_ms: float = 0.0
    tracker_ms: float = 0.0
    lifter_ms: float = 0.0
    smoothing_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class LampStep:
    """2D observations and smoothed 3D people produced for one frameset."""

    timestamp_ns: int
    boxes_by_camera: dict[int, BoxDetections]
    keypoints_by_camera: dict[int, Keypoints2d]
    people: dict[int, PersonState]
    timings: LampTimings


@dataclass(frozen=True, slots=True)
class LampConfig:
    """Configuration for PoseKit detection, LAMP lifting, and SMPL output."""

    checkpoint: Path | None = None
    """Released LAMP state dict; defaults to the package artifact path."""
    smpl_model_path: Path | None = None
    """Chumpy-free neutral SMPL model; defaults to the package artifact path."""
    detector: AnnotatedDetectorConfig = field(default_factory=RtDetrDetectorConfig)
    """PoseKit person detector configuration."""
    pose: AnnotatedPose2dConfig = field(
        default_factory=lambda: VitPoseConfig(model_id="usyd-community/vitpose-plus-base", dataset_index=0)
    )
    """PoseKit COCO-17 estimator; expert zero is the checkpoint's COCO head."""
    window: int = 20
    """Temporal window length in frames; the released model accepts at most 20."""
    keypoint_conf_min: float = 0.5
    """Confidence threshold converted to LAMP's binary validity channel."""
    floor_z: float | None = None
    """Optional floor height in the gravity-aligned world frame, in metres."""

    def __post_init__(self) -> None:
        """Validate values that affect the fixed released architecture."""
        if not 1 <= self.window <= 20:
            raise ValueError(f"window must be in [1, 20], got {self.window}")
        if not 0.0 <= self.keypoint_conf_min <= 1.0:
            raise ValueError(f"keypoint_conf_min must be in [0, 1], got {self.keypoint_conf_min}")

    def setup(self, device: str | torch.device = "cuda") -> LampTracker:
        """Load PoseKit stages and the released LAMP checkpoint.

        Args:
            device: Torch inference device shared by all learned stages.

        Returns:
            Tracker awaiting a four-camera :meth:`LampTracker.configure_cameras`
            call before its first frameset.

        Raises:
            FileNotFoundError: If either non-committed model artifact is absent.
        """
        resolved = torch.device(device)
        checkpoint = self.checkpoint or _DEFAULT_CHECKPOINT
        smpl_model_path = self.smpl_model_path or _DEFAULT_SMPL_MODEL
        missing = [path for path in (checkpoint, smpl_model_path) if not path.is_file()]
        if missing:
            missing_lines = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(f"LAMP model artifact(s) are missing:\n{missing_lines}\nRun `pixi run -e lamp lamp-download-artifacts`.")

        detector_config = replace(self.detector, device=str(resolved)) if isinstance(self.detector, RtDetrDetectorConfig) else self.detector
        pose_config = replace(self.pose, device=str(resolved)) if isinstance(self.pose, VitPoseConfig) else self.pose
        detector = detector_config.setup()
        pose = pose_config.setup()
        lifter_settings = LifterSettings(snippet_length=self.window, kp_thres_for_binary=self.keypoint_conf_min)
        lifter = Lifter.from_checkpoint(
            checkpoint,
            smpl_model_path,
            device=str(resolved),
            settings=lifter_settings,
            capture_cuda_graph=False,
        )
        lifter.set_floor_plane(self.floor_z)
        return LampTracker(
            detector=detector,
            pose=pose,
            lifter=lifter,
            device=resolved,
            window=self.window,
            keypoint_conf_min=self.keypoint_conf_min,
            floor_z=self.floor_z,
        )


class LampTracker:
    """Stateful four-view PoseKit detector, LAMP tracker, lifter, and smoother."""

    def __init__(
        self,
        *,
        detector: PersonDetector,
        pose: TopDownPose2d,
        lifter: Lifter,
        device: torch.device,
        window: int = 20,
        keypoint_conf_min: float = 0.5,
        floor_z: float | None = None,
    ) -> None:
        """Store loaded stages; camera calibration is supplied separately."""
        self._detector = detector
        self._pose = pose
        self._lifter = lifter
        self._device = device
        self._window = window
        self._keypoint_conf_min = keypoint_conf_min
        self._lifter.set_floor_plane(floor_z)
        self._tracker = UpstreamLampTracker(num_cameras=_NUM_VIEWS)
        self._cameras: tuple[RigCamera, ...] = ()
        self._calibrations: dict[int, PerCameraCalibration | None] = {}
        self._T_gravity_world = gravity_aligned_world_transform(np.array([0.0, 0.0, -9.81], dtype=np.float64))

    def configure_cameras(self, cameras: Sequence[RigCamera]) -> None:
        """Set the fixed four-camera rig calibration and reset track state.

        Args:
            cameras: Cameras in LAMP view order; index zero is the anchor.
        """
        if len(cameras) != _NUM_VIEWS:
            raise ValueError(f"LAMP requires exactly {_NUM_VIEWS} cameras, got {len(cameras)}")
        self._cameras = tuple(cameras)
        self._calibrations = {index: camera.as_lamp_calibration(index) for index, camera in enumerate(cameras)}
        self._tracker.reset()

    @property
    def faces(self) -> ndarray:
        """Static neutral-SMPL triangle indices with shape ``(13776, 3)``."""
        faces = self._lifter.smpl_faces
        if faces is None:
            raise RuntimeError("The configured lifter does not expose SMPL faces.")
        return faces

    def smpl_vertices(self, state: PersonState) -> Float32[ndarray, "t 6890 3"]:
        """Skin a smoothed window using its stable shape and fused rotations."""
        steps = len(state.timestamps_ns)
        betas = np.repeat(state.betas[None], steps, axis=0)
        _, vertices = self._lifter.forward_smpl_geometry(
            betas=betas,
            global_orient_rotmat=state.rotations[:, 0],
            body_pose_rotmat=state.rotations[:, 1:],
            transl=state.root_T[:, :3, 3],
        )
        if vertices.shape != (steps, 6890, 3):
            raise RuntimeError(f"Expected SMPL vertices {(steps, 6890, 3)}, got {vertices.shape}")
        return vertices

    def step(self, frameset: Frameset) -> LampStep:
        """Advance detection, association, lifting, and smoothing by one frameset."""
        started = time.perf_counter()
        if len(self._cameras) != _NUM_VIEWS:
            raise RuntimeError("Call configure_cameras() with exactly four cameras before step().")
        if frameset.images.shape[0] != _NUM_VIEWS:
            raise ValueError(f"LAMP requires exactly four images, got shape {frameset.images.shape}")

        frames: UInt8[Tensor, "4 h w 3"] = torch.from_numpy(np.ascontiguousarray(frameset.images)).to(self._device)
        stage = time.perf_counter()
        detections = self._detector(frames)
        detector_ms = (time.perf_counter() - stage) * 1000.0
        stage = time.perf_counter()
        keypoints = self._pose(frames, detections)
        pose_ms = (time.perf_counter() - stage) * 1000.0
        self._validate_posekit_outputs(detections, keypoints)

        upstream_detections, rows_by_camera = self._to_upstream_detections(detections, keypoints, frameset.timestamp_ns)
        world_T_cameras = {
            index: (frameset.world_T_rig @ np.linalg.inv(camera.cam_T_rig)).astype(np.float32)
            for index, camera in enumerate(self._cameras)
        }
        previous_lift_times = {person_id: person.last_lifted_ts for person_id, person in self._tracker.people.items()}
        stage = time.perf_counter()
        self._tracker.track_frameset(upstream_detections, world_T_cameras, self._calibrations, frameset.timestamp_ns)
        tracker_ms = (time.perf_counter() - stage) * 1000.0

        snippets = self._tracker.get_snippets_for_lifting(
            snippet_length=self._window,
            T_gravityWorld_world=self._T_gravity_world,
            kp_thres=self._keypoint_conf_min,
            num_views=_NUM_VIEWS,
        )
        self._convert_snippets_to_virtual_pinhole(snippets)
        stage = time.perf_counter()
        lifted = self._lifter.lift_all_steps_batched(snippets)
        lifter_ms = (time.perf_counter() - stage) * 1000.0
        stage = time.perf_counter()
        for person_id, skeletons in lifted.items():
            self._tracker.attach_skeletons(
                person_id,
                skeletons,
                T_world_cams=world_T_cameras,
                T_gravityWorld_world=self._T_gravity_world,
            )
        self._tracker.merge_lifted_tracks(
            current_ts_ns=frameset.timestamp_ns,
            prev_last_lifted_ts=previous_lift_times,
        )
        smoothing_ms = (time.perf_counter() - stage) * 1000.0

        boxes_by_camera, keypoints_by_camera = self._posekit_outputs_by_camera(
            detections, keypoints, rows_by_camera, upstream_detections
        )
        people = self._public_people()
        total_ms = (time.perf_counter() - started) * 1000.0
        return LampStep(
            timestamp_ns=frameset.timestamp_ns,
            boxes_by_camera=boxes_by_camera,
            keypoints_by_camera=keypoints_by_camera,
            people=people,
            timings=LampTimings(detector_ms, pose_ms, tracker_ms, lifter_ms, smoothing_ms, total_ms),
        )

    @staticmethod
    def _validate_posekit_outputs(detections: BoxDetections, keypoints: Keypoints2d) -> None:
        """Check the row alignment and COCO-17 skeleton LAMP requires."""
        if detections.num_detections != keypoints.num_instances:
            raise ValueError(
                f"PoseKit detector/keypoint rows differ: {detections.num_detections} boxes and {keypoints.num_instances} poses"
            )
        if not torch.equal(detections.frame_indices, keypoints.frame_indices):
            raise ValueError("PoseKit detector and keypoint frame_indices are not row-aligned.")
        if keypoints.skeleton != COCO_17 or int(keypoints.xy.shape[1]) != _NUM_COCO_KEYPOINTS:
            raise ValueError(f"LAMP requires COCO-17 keypoints, got {keypoints.skeleton.name!r} with {keypoints.xy.shape[1]} points")

    def _to_upstream_detections(
        self,
        detections: BoxDetections,
        keypoints: Keypoints2d,
        timestamp_ns: int,
    ) -> tuple[dict[int, list[Detection2D]], dict[int, list[int]]]:
        """Copy flattened PoseKit rows into per-camera mutable tracker records."""
        boxes = detections.xyxy_numpy()
        box_scores = detections.scores.detach().cpu().numpy().astype(np.float32, copy=False)
        points = keypoints.xy_numpy()
        point_scores = keypoints.scores_numpy()
        frame_indices = detections.frame_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        per_camera: dict[int, list[Detection2D]] = {index: [] for index in range(_NUM_VIEWS)}
        rows: dict[int, list[int]] = {index: [] for index in range(_NUM_VIEWS)}
        for row, camera_index_raw in enumerate(frame_indices):
            camera_index = int(camera_index_raw)
            if not 0 <= camera_index < _NUM_VIEWS:
                raise ValueError(f"PoseKit frame index {camera_index} is outside the four-view frameset")
            points_with_scores = np.concatenate([points[row], point_scores[row, :, None]], axis=1).astype(np.float32)
            per_camera[camera_index].append(
                Detection2D(
                    box_xyxy=boxes[row].copy(),
                    box_score=float(box_scores[row]),
                    keypoints=points_with_scores,
                    cam_idx=camera_index,
                    timestamp_ns=timestamp_ns,
                )
            )
            rows[camera_index].append(row)
        return per_camera, rows

    def _convert_snippets_to_virtual_pinhole(self, snippets: dict[int, SnippetData]) -> None:
        """Preserve KB4 rays while selecting LAMP's eager four-value path."""
        for snippet in snippets.values():
            for view_index, camera in enumerate(self._cameras):
                view = snippet.kp2ds_per_view[view_index]
                valid = view[..., 2] > 0.0
                converted = np.zeros_like(view)
                converted[..., 2] = view[..., 2]
                converted[..., :2][valid] = camera.to_virtual_pinhole(view[..., :2][valid])
                snippet.kp2ds_per_view[view_index] = converted
                snippet.cam_params_per_view[view_index] = np.tile(camera.lifter_params(), (len(view), 1))

    def _posekit_outputs_by_camera(
        self,
        detections: BoxDetections,
        keypoints: Keypoints2d,
        rows_by_camera: dict[int, list[int]],
        upstream: dict[int, list[Detection2D]],
    ) -> tuple[dict[int, BoxDetections], dict[int, Keypoints2d]]:
        """Return PoseKit containers split by view and decorated with track IDs."""
        boxes_by_camera: dict[int, BoxDetections] = {}
        keypoints_by_camera: dict[int, Keypoints2d] = {}
        for camera_index in range(_NUM_VIEWS):
            rows = torch.as_tensor(rows_by_camera[camera_index], dtype=torch.long, device=detections.xyxy.device)
            track_ids = torch.as_tensor(
                [record.track_id if record.track_id is not None else -1 for record in upstream[camera_index]],
                dtype=torch.long,
                device=detections.xyxy.device,
            )
            boxes_by_camera[camera_index] = BoxDetections(
                xyxy=detections.xyxy[rows],
                scores=detections.scores[rows],
                frame_indices=detections.frame_indices[rows],
                masks=detections.masks[rows] if detections.masks is not None else None,
                track_ids=track_ids,
            )
            keypoints_by_camera[camera_index] = Keypoints2d(
                xy=keypoints.xy[rows],
                scores=keypoints.scores[rows],
                frame_indices=keypoints.frame_indices[rows],
                skeleton=keypoints.skeleton,
                uncertainty=keypoints.uncertainty[rows] if keypoints.uncertainty is not None else None,
            )
        return boxes_by_camera, keypoints_by_camera

    def _public_people(self) -> dict[int, PersonState]:
        """Copy each lifted track's current smoothed window into owned arrays."""
        people: dict[int, PersonState] = {}
        for track_id, tracked in self._tracker.people.items():
            rows = [(timestamp, state.skeleton) for timestamp, state in sorted(tracked.ts_to_states.items()) if state.skeleton is not None]
            rows = rows[-self._window :]
            if not rows:
                continue
            timestamps = np.asarray([timestamp for timestamp, _ in rows], dtype=np.int64)
            skeletons = [skeleton for _, skeleton in rows]
            joints = np.stack([skeleton.kp_world for skeleton in skeletons]).astype(np.float32)
            roots = np.stack([skeleton.T_world_pelvis for skeleton in skeletons]).astype(np.float32)
            rotations = np.stack([skeleton.joints_rot_mat for skeleton in skeletons]).astype(np.float32)
            betas = tracked.shape_estimate.astype(np.float32, copy=True)
            if betas.shape != (_NUM_BETAS,) or joints.shape[1:] != (_NUM_SMPL_JOINTS, 3):
                continue
            people[track_id] = PersonState(track_id, timestamps, joints, betas, roots, rotations)
        return people


__all__ = ("Frameset", "LampConfig", "LampStep", "LampTimings", "LampTracker", "PersonState")
