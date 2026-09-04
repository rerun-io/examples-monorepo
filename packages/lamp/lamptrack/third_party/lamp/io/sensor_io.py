# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VRS + MPS adapters built on `projectaria_tools`."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from lamptrack.third_party.lamp.core.se3 import as_4x4_f32
from lamptrack.third_party.lamp.core.types import CameraOrientation, Frameset
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import DeviceVersion
from projectaria_tools.core.mps import MpsDataPaths, MpsDataProvider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId

# Calibration containers


@dataclass(slots=True)
class PerCameraCalibration:
    """Calibration for a single camera, exposed as a plain dataclass."""

    cam_idx: int
    label: str
    image_size: tuple[int, int]  # (width, height)
    intrinsic_matrix: np.ndarray  # 3x3 float32 (K-matrix from fx/fy/cx/cy)
    T_device_camera: np.ndarray  # 4x4 float32
    distortion_model: str
    project: Callable[[np.ndarray], np.ndarray | None] | None  # 3D->2D
    unproject: Callable[[np.ndarray], np.ndarray | None] | None  # 2D->3D ray
    projection_params: np.ndarray = field(  # raw `cc.get_projection_params()`
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )


class CameraCalibration:
    """Container indexable by integer `cam_idx`."""

    def __init__(self, per_cam: dict[int, PerCameraCalibration]) -> None:
        self._per_cam = per_cam

    def camera(self, cam_idx: int) -> PerCameraCalibration:
        return self._per_cam[cam_idx]

    @property
    def cam_indices(self) -> list[int]:
        return sorted(self._per_cam.keys())

    @property
    def labels(self) -> dict[int, str]:
        return {idx: c.label for idx, c in self._per_cam.items()}


# Camera-set helpers — view-count agnostic, no hard-coded device branches

# Labels we hand to LAMP. ET cameras / IMU / audio are filtered out.
_RELEVANT_CAMERA_PREFIXES: tuple[str, ...] = ("slam", "camera-slam")
_RGB_LABEL = "camera-rgb"


def _is_relevant_camera(label: str) -> bool:
    """True for cameras that LAMP can use (SLAM monochrome + RGB)."""
    if label == _RGB_LABEL:
        return True
    return any(label.startswith(p) for p in _RELEVANT_CAMERA_PREFIXES)


def _order_camera_labels(labels: list[str]) -> list[str]:
    """Stable cam_idx assignment: SLAMs alphabetical, then RGB last."""
    slams = sorted(lbl for lbl in labels if "slam" in lbl)
    rgbs = sorted(lbl for lbl in labels if lbl == _RGB_LABEL)
    return slams + rgbs


def _device_family_from_version(
    version: Any,
) -> Literal["aria_gen1", "aria_gen2", "unknown"]:
    """Map projectaria_tools `DeviceVersion` to a friendly family name."""
    family: Literal["aria_gen1", "aria_gen2", "unknown"]
    if version == DeviceVersion.Gen1:
        family = "aria_gen1"
    elif version == DeviceVersion.Gen2:
        family = "aria_gen2"
    else:
        family = "unknown"
    return family


def _default_camera_orientation(device_family: str, label: str) -> CameraOrientation:
    """Default camera orientation for each Aria generation."""
    if device_family == "aria_gen1" and "slam" in label:
        return CameraOrientation.ROTATE_90_CW
    return CameraOrientation.UPRIGHT


# VRS loader


FLAT_VRS_NAME = "video.vrs"
FLAT_CLOSED_LOOP_TRAJECTORY_NAME = "closed_loop_trajectory.csv"
FLAT_ONLINE_CALIBRATION_NAME = "online_calibration.jsonl"
FLAT_SEMIDENSE_POINTS_NAME = "semidense_points.csv.gz"


@dataclass(frozen=True, slots=True)
class RecordingPaths:
    """Resolved files for one flat LAMP recording folder."""

    root: Path
    vrs: Path
    closed_loop_trajectory: Path
    online_calibration: Path
    semidense_points: Path


def resolve_recording_paths(recording_dir: Path) -> RecordingPaths:
    """Resolve the required files in a flat recording folder."""
    root = Path(recording_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"recording folder does not exist: {root}")

    paths = RecordingPaths(
        root=root,
        vrs=root / FLAT_VRS_NAME,
        closed_loop_trajectory=root / FLAT_CLOSED_LOOP_TRAJECTORY_NAME,
        online_calibration=root / FLAT_ONLINE_CALIBRATION_NAME,
        semidense_points=root / FLAT_SEMIDENSE_POINTS_NAME,
    )
    missing = [
        path.name
        for path in (
            paths.vrs,
            paths.closed_loop_trajectory,
            paths.online_calibration,
            paths.semidense_points,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"recording folder {root} is missing required file(s): {', '.join(missing)}"
        )
    return paths


@dataclass(frozen=True, slots=True)
class VrsFrameIndex:
    """A decode-ready VRS frame index and its canonical timestamp."""

    frame_idx: int
    timestamp_ns: int


class VrsLoader:
    """Read a VRS recording: per-camera streams, calibration, orientations."""

    def __init__(
        self,
        vrs_path: Path,
        *,
        decode_camera_indices: Sequence[int] | None = None,
    ) -> None:
        self._vrs_path: Path = Path(vrs_path)
        self._decode_camera_indices = (
            {int(idx) for idx in decode_camera_indices}
            if decode_camera_indices is not None
            else None
        )
        self._provider = data_provider.create_vrs_data_provider(str(vrs_path))
        if self._provider is None:
            raise FileNotFoundError(f"failed to open VRS file: {vrs_path}")

        self._device_calib = self._provider.get_device_calibration()
        if self._device_calib is None:
            raise RuntimeError(f"VRS file has no device calibration: {vrs_path}")

        # cam_idx ordering: SLAMs alphabetical, then RGB last.
        all_labels = list(self._device_calib.get_camera_labels())
        relevant = [lbl for lbl in all_labels if _is_relevant_camera(lbl)]
        ordered = _order_camera_labels(relevant)
        self._idx_to_label: dict[int, str] = dict(enumerate(ordered))
        self._label_to_idx: dict[str, int] = {
            lbl: i for i, lbl in self._idx_to_label.items()
        }

        # Map cam_idx -> stream_id for image fetching.
        self._idx_to_stream: dict[int, StreamId] = {}
        for sid in self._provider.get_all_streams():
            label = self._provider.get_label_from_stream_id(sid)
            if label in self._label_to_idx:
                self._idx_to_stream[self._label_to_idx[label]] = sid

        # Build the calibration container.
        self._device_family: Literal["aria_gen1", "aria_gen2", "unknown"] = (
            _device_family_from_version(self._device_calib.get_device_version())
        )
        per_cam: dict[int, PerCameraCalibration] = {}
        for idx, label in self._idx_to_label.items():
            cc = self._device_calib.get_camera_calib(label)
            if cc is None:
                continue
            f = cc.get_focal_lengths()
            c = cc.get_principal_point()
            K = np.array(
                [[f[0], 0.0, c[0]], [0.0, f[1], c[1]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            size = cc.get_image_size()
            try:
                proj_params = np.asarray(cc.get_projection_params(), dtype=np.float32)
            except (AttributeError, RuntimeError):  # pragma: no cover
                # Some camera models (linear) don't expose distortion params;
                # fall back to an empty vector and let the consumer decide.
                proj_params = np.zeros(0, dtype=np.float32)
            per_cam[idx] = PerCameraCalibration(
                cam_idx=idx,
                label=label,
                image_size=(int(size[0]), int(size[1])),
                intrinsic_matrix=K,
                T_device_camera=as_4x4_f32(cc.get_transform_device_camera()),
                distortion_model=str(cc.get_model_name()),
                project=cc.project,
                unproject=cc.unproject,
                projection_params=proj_params,
            )
        self._calibration = CameraCalibration(per_cam)

    @property
    def path(self) -> Path:
        """The on-disk path this loader was opened from (the ctor `vrs_path`)."""
        return self._vrs_path

    @property
    def device_family(self) -> Literal["aria_gen1", "aria_gen2", "unknown"]:
        return self._device_family

    @property
    def camera_indices(self) -> list[int]:
        return sorted(self._idx_to_label.keys())

    @property
    def rgb_camera_indices(self) -> list[int]:
        """Cam indices whose label matches the `_RGB_LABEL` constant."""
        return sorted(
            idx for idx, label in self._idx_to_label.items() if label == _RGB_LABEL
        )

    @property
    def slam_camera_indices(self) -> list[int]:
        indices = sorted(
            idx for idx, label in self._idx_to_label.items() if "slam" in label
        )
        if self._decode_camera_indices is None:
            return indices
        return [idx for idx in indices if idx in self._decode_camera_indices]

    @property
    def num_framesets(self) -> int:
        slam_indices = self.slam_camera_indices
        if not slam_indices:
            return 0
        return min(
            self._provider.get_num_data(self._idx_to_stream[idx])
            for idx in slam_indices
        )

    @property
    def calibration(self) -> CameraCalibration:
        return self._calibration

    @property
    def camera_orientations(self) -> dict[int, CameraOrientation]:
        return {
            idx: _default_camera_orientation(self._device_family, label)
            for idx, label in self._idx_to_label.items()
        }

    @property
    def labels(self) -> dict[int, str]:
        """`cam_idx -> label` for diagnostic prints / smoke scripts."""
        return dict(self._idx_to_label)

    def frameset_timestamps_ns(self) -> list[int]:
        """Per-frame DEVICE_TIME timestamps of the anchor SLAM stream."""
        slam_indices = self.slam_camera_indices
        if not slam_indices:
            return []
        return self._timestamp_list_ns(slam_indices[0])[: self.num_framesets]

    def iter_frame_indices(
        self,
        max_framesets: int | None = None,
        min_frame_gap_ns: int = 0,
        start_offset_ns: int = 0,
    ) -> Iterator[VrsFrameIndex]:
        """Yield frame indices after start-offset and FPS throttling."""
        slam_indices = self.slam_camera_indices
        if not slam_indices:
            return

        anchor_ts = self._timestamp_list_ns(slam_indices[0])
        n_total = min(self.num_framesets, len(anchor_ts))
        if n_total == 0:
            return

        first_ts = anchor_ts[0]
        last_yielded_ts = -1
        yielded = 0
        for frame_idx in range(n_total):
            if max_framesets is not None and yielded >= max_framesets:
                break
            ts_ns = anchor_ts[frame_idx]
            if start_offset_ns > 0 and (ts_ns - first_ts) < start_offset_ns:
                continue
            if (
                min_frame_gap_ns > 0
                and last_yielded_ts >= 0
                and (ts_ns - last_yielded_ts) < min_frame_gap_ns
            ):
                continue
            last_yielded_ts = ts_ns
            yielded += 1
            yield VrsFrameIndex(frame_idx=frame_idx, timestamp_ns=ts_ns)

    def iter_framesets(
        self,
        max_framesets: int | None = None,
        min_frame_gap_ns: int = 0,
        start_offset_ns: int = 0,
    ) -> Iterator[Frameset]:
        """Yield synced per-camera SLAM `Frameset`s, with optional timestamp skipping."""
        slam_indices = self.slam_camera_indices
        if not slam_indices:
            return

        for frame in self.iter_frame_indices(
            max_framesets=max_framesets,
            min_frame_gap_ns=min_frame_gap_ns,
            start_offset_ns=start_offset_ns,
        ):
            fs = self._decode_frameset_at_index(frame.frame_idx, slam_indices)
            if fs is None:
                continue
            assert fs.timestamp_ns == frame.timestamp_ns, (
                "anchor cam capture_timestamp_ns "
                f"({fs.timestamp_ns}) != get_timestamps_ns[{frame.frame_idx}] "
                f"({frame.timestamp_ns}); the throttle timestamp source diverged "
                "from the emitted ts"
            )
            yield fs

    def frameset_at_index(self, idx: int) -> Frameset | None:
        """Decode a SINGLE frameset by frame index (no iteration / throttle)."""
        slam_indices = self.slam_camera_indices
        if not slam_indices:
            return None
        if idx < 0 or idx >= self.num_framesets:
            return None
        return self._decode_frameset_at_index(idx, slam_indices)

    def decode_camera_image(
        self, frame_idx: int, cam_idx: int
    ) -> tuple[int, np.ndarray]:
        """Decode one camera image and return `(timestamp_ns, raw_array)`."""
        sid = self._idx_to_stream[cam_idx]
        img_data, rec = self._provider.get_image_data_by_index(sid, frame_idx)
        return int(rec.capture_timestamp_ns), img_data.to_numpy_array()

    def _decode_frameset_at_index(
        self, idx: int, slam_indices: Sequence[int]
    ) -> Frameset | None:
        per_cam_images: dict[int, np.ndarray] = {}
        ts_ns_canonical: int | None = None
        for cam_idx in slam_indices:
            ts_ns, image = self.decode_camera_image(idx, cam_idx)
            per_cam_images[cam_idx] = _to_uint8_hwc3(image)
            if ts_ns_canonical is None:
                ts_ns_canonical = ts_ns

        assert ts_ns_canonical is not None  # invariant: at least one SLAM cam
        return Frameset(timestamp_ns=ts_ns_canonical, images=per_cam_images)

    def _timestamp_list_ns(self, cam_idx: int) -> list[int]:
        ts = self._provider.get_timestamps_ns(
            self._idx_to_stream[cam_idx], TimeDomain.DEVICE_TIME
        )
        return [int(t) for t in ts]


def _to_uint8_hwc3(arr: np.ndarray) -> np.ndarray:
    """Normalize an image to `(H, W, 3) uint8`. Grayscale -> repeat channels."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"unsupported image shape {arr.shape}, dtype {arr.dtype}")


# MPS loader


class MpsLoader:
    """Read flat-folder MPS files: pose trajectory, calibration, and points."""

    def __init__(self, recording_dir: Path) -> None:
        self._recording_paths = resolve_recording_paths(recording_dir)
        self._provider = MpsDataProvider(_mps_data_paths(self._recording_paths))
        if not self._provider.has_closed_loop_poses():
            raise FileNotFoundError(
                f"recording folder {recording_dir} has no usable closed-loop trajectory "
                f"at {self._recording_paths.closed_loop_trajectory}."
            )
        self._T_gravityWorld_world = self._compute_T_gravityWorld_world()

    def _compute_T_gravityWorld_world(self) -> np.ndarray:
        """Build T_gw_w from MPS gravity vector (taken at the first valid pose).

        Gravity-world frame: +Z = up (anti-gravity). We rotate world so that
        normalized(-gravity_world) maps to +Z, with translation = 0.
        """
        from projectaria_tools.core.sensor_data import TimeQueryOptions

        # Find the first valid pose by binary search inside the trajectory range.
        # MPS trajectories don't expose a direct first-ts API, so query CLOSEST
        # at t=0 and walk back.
        pose = self._provider.get_closed_loop_pose(0, TimeQueryOptions.AFTER)
        if pose is None:
            return np.eye(4, dtype=np.float32)

        g = np.asarray(pose.gravity_world, dtype=np.float32)
        z_axis = -g / max(np.linalg.norm(g), 1e-9)  # gravity-world's +Z in world

        # Pick any vector not parallel to z_axis to build x.
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(helper, z_axis)) > 0.99:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x_axis = helper - np.dot(helper, z_axis) * z_axis
        x_axis /= max(np.linalg.norm(x_axis), 1e-9)
        y_axis = np.cross(z_axis, x_axis)

        # R_world_gw has those axes as columns; invert (= transpose) for gw <- world.
        R_world_gw = np.column_stack([x_axis, y_axis, z_axis])
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R_world_gw.T
        return out

    @property
    def T_gravityWorld_world(self) -> np.ndarray:
        return self._T_gravityWorld_world

    def query_T_world_device(self, timestamp_ns: int) -> np.ndarray | None:
        pose: Any = self._provider.get_interpolated_closed_loop_pose(timestamp_ns)
        if pose is None:
            return None
        return as_4x4_f32(pose.transform_world_device)

    def semidense_points(
        self,
        threshold_dep: float = 0.05,
        threshold_invdep: float = 0.001,
        max_count: int = 200_000,
    ) -> np.ndarray | None:
        """Load MPS semidense world-frame points with confidence filtering."""
        if not self._provider.has_semidense_point_cloud():
            return None
        cloud = self._provider.get_semidense_point_cloud()
        n_raw = len(cloud)
        if n_raw == 0:
            return None

        # One Python pass to pull (x, y, z, dist_std, inv_dist_std) per point.
        # Repeated `np.fromiter` calls would re-iterate the cloud; assemble a
        # single structured array instead.
        xyz = np.empty((n_raw, 3), dtype=np.float32)
        dist = np.empty(n_raw, dtype=np.float32)
        inv_dist = np.empty(n_raw, dtype=np.float32)
        for i, p in enumerate(cloud):
            pw = p.position_world
            xyz[i, 0] = pw[0]
            xyz[i, 1] = pw[1]
            xyz[i, 2] = pw[2]
            dist[i] = p.distance_std
            inv_dist[i] = p.inverse_distance_std

        keep = (dist <= threshold_dep) & (inv_dist <= threshold_invdep)
        xyz = xyz[keep]
        if xyz.shape[0] == 0:
            return None
        if max_count and xyz.shape[0] > max_count:
            idx = np.random.default_rng(0).choice(
                xyz.shape[0], max_count, replace=False
            )
            xyz = xyz[idx]
        return xyz


def _mps_data_paths(paths: RecordingPaths) -> MpsDataPaths:
    out = MpsDataPaths()
    out.root = str(paths.root)
    out.slam.closed_loop_trajectory = str(paths.closed_loop_trajectory)
    out.slam.online_calibrations = str(paths.online_calibration)
    out.slam.semidense_points = str(paths.semidense_points)
    return out
