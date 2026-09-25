"""Read-only Aria Gen2 Pilot streams and MPS tables on their native clocks."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from projectaria_tools.core.calibration import CameraCalibration, CameraModelType, DeviceCalibration, device_calibration_from_json_string
from scipy.spatial.transform import Rotation

from dataforge import hands
from dataforge.vrs_hevc import VrsHevcReader

CAMERAS: tuple[tuple[str, str, int], ...] = (
    ("214-1", "camera-rgb", 10),
    ("1201-1", "slam-front-left", 30),
    ("1201-2", "slam-front-right", 30),
    ("1201-3", "slam-side-left", 30),
    ("1201-4", "slam-side-right", 30),
)
"""Native stream IDs, factory labels and nominal container rates."""
SEQUENCES: tuple[str, ...] = ("clean_0", "cook_0", "eat_0", "eat_1", "eat_2", "eat_3", "play_0", "play_1", "play_2", "play_3", "walk_0", "walk_1")
"""Release v1.0 sequence names."""


def numeric_columns(path: Path, columns: list[str]) -> Float64[ndarray, "n c"]:
    """Read selected CSV stream columns; no dataset-owned JSON copy is made."""
    with path.open() as stream:
        header: list[str] = next(csv.reader(stream))
        try:
            selected: list[int] = [header.index(name) for name in columns]
            return np.loadtxt(stream, delimiter=",", usecols=selected, ndmin=2, dtype=np.float64)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error


def poses_from_columns(values: Float64[ndarray, "n 7"]) -> Float64[ndarray, "n 4 4"]:
    """Translation and xyzw quaternion to SE(3); invalid inputs remain missing."""
    valid: Bool[ndarray, "n"] = np.isfinite(values).all(axis=1) & (np.linalg.norm(values[:, 3:], axis=1) > 1e-12)
    result: Float64[ndarray, "n 4 4"] = np.full((len(values), 4, 4), np.nan)
    result[valid] = np.eye(4)
    result[valid, :3, :3] = Rotation.from_quat(values[valid, 3:]).as_matrix() if valid.any() else np.empty((0, 3, 3))
    result[valid, :3, 3] = values[valid, :3]
    return result


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Every native closed-loop row, including its shipped quality score."""

    times_ns: Int64[ndarray, "n"]
    """Device microseconds converted to nanoseconds."""
    poses: Float64[ndarray, "n 4 4"]
    """World-from-device matrices; invalid rows become NaN."""
    quality: Float64[ndarray, "n"]
    """Shipped quality, including 0.0 and 0.5."""

    def __post_init__(self) -> None:
        if len(self.times_ns) != len(self.poses) or len(self.times_ns) != len(self.quality):
            raise ValueError("trajectory columns differ in length")
        if not len(self.times_ns) or np.any(np.diff(self.times_ns) <= 0):
            raise ValueError("trajectory timestamps must be nonempty and strictly increasing")
        finite: Bool[ndarray, "n"] = np.isfinite(self.poses).all(axis=(1, 2))
        valid: Bool[ndarray, "n"] = finite.copy()
        rotation: Float64[ndarray, "m 3 3"] = self.poses[finite, :3, :3]
        valid[finite] &= np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        valid[finite] &= np.isclose(rotation @ np.swapaxes(rotation, 1, 2), np.eye(3), atol=1e-6).all(axis=(1, 2))
        valid &= np.isclose(self.poses[:, 3], [0.0, 0.0, 0.0, 1.0], atol=1e-6).all(axis=1)
        self.poses[~valid] = np.nan

    def at(self, times_ns: Int64[ndarray, "m"]) -> Float64[ndarray, "m 4 4"]:
        """Slerp/lerp inside valid <=2 ms brackets; never extrapolate or clamp."""
        result: Float64[ndarray, "m 4 4"] = np.full((len(times_ns), 4, 4), np.nan)
        right: Int64[ndarray, "m"] = np.searchsorted(self.times_ns, times_ns)
        bounded: Int64[ndarray, "m"] = np.minimum(right, len(self.times_ns) - 1)
        exact: Bool[ndarray, "m"] = (right < len(self.times_ns)) & (self.times_ns[bounded] == times_ns)
        result[exact] = self.poses[bounded[exact]]
        between: Int64[ndarray, "k"] = np.flatnonzero((~exact) & (right > 0) & (right < len(self.times_ns)))
        high: Int64[ndarray, "k"] = right[between]
        low: Int64[ndarray, "k"] = high - 1
        gaps: Int64[ndarray, "k"] = self.times_ns[high] - self.times_ns[low]
        good: Bool[ndarray, "k"] = (
            (gaps <= 2_000_000) & np.isfinite(self.poses[low]).all(axis=(1, 2)) & np.isfinite(self.poses[high]).all(axis=(1, 2))
        )
        between, low, high = between[good], low[good], high[good]
        if len(between):
            fraction: Float64[ndarray, "k"] = (times_ns[between] - self.times_ns[low]) / gaps[good]
            start: Rotation = Rotation.from_matrix(self.poses[low, :3, :3])
            end: Rotation = Rotation.from_matrix(self.poses[high, :3, :3])
            result[between] = np.eye(4)
            result[between, :3, :3] = (start * Rotation.from_rotvec((start.inv() * end).as_rotvec() * fraction[:, None])).as_matrix()
            result[between, :3, 3] = self.poses[low, :3, 3] * (1.0 - fraction[:, None]) + self.poses[high, :3, 3] * fraction[:, None]
        return result


def read_trajectory(path: Path) -> Trajectory:
    """Keep every closed-loop row; quality does not change pose availability."""
    columns: list[str] = [
        "tracking_timestamp_us",
        *[f"t{axis}_world_device" for axis in "xyz"],
        *[f"q{axis}_world_device" for axis in "xyzw"],
        "quality_score",
    ]
    values: Float64[ndarray, "n 9"] = numeric_columns(path, columns)
    return Trajectory(values[:, 0].astype(np.int64) * 1000, poses_from_columns(values[:, 1:8]), values[:, 8])


@dataclass(frozen=True, slots=True)
class HandSamples:
    """Dense world measurements, one row per MPS hand row."""

    times_ns: Int64[ndarray, "n"]
    """Native 30 Hz MPS timestamps, with dropped rows preserved as gaps."""
    positions: Float32[ndarray, "n 133 3"]
    """COCO-133 world metres; uncovered slots and missing hands are NaN."""
    confidence: Float32[ndarray, "n 133"]
    """Shipped hand confidence repeated over its covered slots."""
    wrists: Float64[ndarray, "n 2 4 4"]
    """World-from-wrist per side."""
    normals: Float64[ndarray, "n 2 2 3"]
    """World palm and wrist normals per side; no translation applied."""
    scores: Float64[ndarray, "n 2"]
    """Shipped per-hand confidence (including -1 for missing)."""
    device_poses: Float64[ndarray, "n 4 4"]
    """Interpolated world-from-device at the hand timestamp."""


def read_hands(path: Path, trajectory: Trajectory, stop_ns: int | None = None) -> HandSamples:
    """Read every native MPS row and transform device landmarks at its own time."""
    columns: list[str] = ["tracking_timestamp_us"]
    for side in ("left", "right"):
        columns += [f"{side}_tracking_confidence"]
        columns += [f"t{axis}_{side}_landmark_{joint}_device" for joint in range(21) for axis in "xyz"]
        columns += [f"{kind}{axis}_{side}_device_wrist" for kind, axes in (("t", "xyz"), ("q", "xyzw")) for axis in axes]
        columns += [f"n{axis}_{side}_{part}_device" for part in ("palm", "wrist") for axis in "xyz"]
    values: Float64[ndarray, "n 155"] = numeric_columns(path, columns)
    times: Int64[ndarray, "n"] = values[:, 0].astype(np.int64) * 1000
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: hand timestamps must increase")
    keep: Bool[ndarray, "n"] = np.ones(len(times), dtype=np.bool_) if stop_ns is None else times <= stop_ns
    values, times = values[keep], times[keep]
    poses: Float64[ndarray, "n 4 4"] = trajectory.at(times)
    valid_pose: Bool[ndarray, "n"] = np.isfinite(poses).all(axis=(1, 2))
    landmarks: Float32[ndarray, "n 2 21 3"] = np.full((len(times), 2, 21, 3), np.nan, dtype=np.float32)
    wrists: Float64[ndarray, "n 2 4 4"] = np.full((len(times), 2, 4, 4), np.nan)
    normals: Float64[ndarray, "n 2 2 3"] = np.full((len(times), 2, 2, 3), np.nan)
    scores: Float64[ndarray, "n 2"] = np.stack([values[:, 1], values[:, 78]], axis=1)
    for side in range(2):
        offset = 1 + 77 * side
        present = valid_pose & (scores[:, side] != -1.0)
        points = values[:, offset + 1 : offset + 64].reshape(-1, 21, 3)
        landmarks[present, side] = (np.einsum("nij,nkj->nki", poses[present, :3, :3], points[present]) + poses[present, None, :3, 3]).astype(
            np.float32
        )
        wrists[present, side] = poses[present] @ poses_from_columns(values[present, offset + 64 : offset + 71])
        normals[present, side] = np.einsum("nij,nkj->nki", poses[present, :3, :3], values[present, offset + 71 : offset + 77].reshape(-1, 2, 3))
    positions: Float32[ndarray, "n 133 3"] = np.full((len(times), 133, 3), np.nan, dtype=np.float32)
    confidence: Float32[ndarray, "n 133"] = np.zeros((len(times), 133), dtype=np.float32)
    for index in range(len(times)):
        positions[index], confidence[index] = hands.coco133_from_hands(landmarks[index], scores[index].astype(np.float32))
    positions, confidence = hands.confidence_rule(positions, confidence)
    return HandSamples(times, positions, confidence, wrists, normals, scores, poses)


@dataclass(frozen=True, slots=True)
class Camera:
    """One native, unrotated camera and its full factory lens model."""

    stream_id: str
    """VRS identifier."""
    label: str
    """Factory label."""
    fps: int
    """Nominal container rate only; Rerun uses capture timestamps."""
    calibration: CameraCalibration
    """FISHEYE624 including thin prism."""
    times_ns: Int64[ndarray, "n"]
    """Own native capture clock, optionally preview-limited."""
    source_count: int
    """Full stream length."""


@dataclass(frozen=True, slots=True)
class Scene:
    """One opened sequence, with independent camera and label clocks."""

    source: Path
    """Read-only directory."""
    calibration: DeviceCalibration
    """Factory device frame shared by MPS and sensors."""
    cameras: list[Camera]
    """RGB, front stereo, side stereo."""
    trajectory: Trajectory
    """Full native trajectory, including interpolation neighbours outside preview."""
    hands: HandSamples
    """Native hand rows, optionally cut at the preview's last camera timestamp."""
    stop_ns: int | None
    """Preview cutoff, or full sequence."""


def read_scene(source: Path, frame_limit: int | None = None) -> Scene:
    """Open factory calibration and each clock independently; never decode VRS images."""
    if frame_limit is not None and frame_limit < 1:
        raise ValueError("frame_limit must be positive")
    first: VrsHevcReader = VrsHevcReader(source / "video.vrs", CAMERAS[0][0])
    factory: DeviceCalibration | None = device_calibration_from_json_string(first.description.file_tags["calib_json"])
    if factory is None:
        raise ValueError(f"{source}: missing factory calibration")
    cameras: list[Camera] = []
    for stream_id, label, fps in CAMERAS:
        calibration: CameraCalibration | None = factory.get_camera_calib(label)
        if calibration is None or calibration.get_model_name() != CameraModelType.FISHEYE624:
            raise ValueError(f"{source}/{label}: missing FISHEYE624 factory calibration")
        reader: VrsHevcReader = first if stream_id == CAMERAS[0][0] else VrsHevcReader(source / "video.vrs", stream_id)
        width, height = reader.image_size()
        factory_width, factory_height = (int(value) for value in calibration.get_image_size())
        if (factory_width, factory_height) != (width, height):
            # camera-rgb's factory model is for the full 4032x3024 sensor; the stream is the
            # 2560x1920 "pov_downscaled" image. Same rescale as projectaria-tools' VRS provider.
            scale: float = width / factory_width
            if round(factory_height * scale) != height:
                raise ValueError(f"{source}/{label}: stream {width}x{height} is not a uniform scale of {factory_width}x{factory_height}")
            calibration = calibration.rescale(np.array([width, height]), scale)
        times: Int64[ndarray, "n"] = reader.capture_timestamps()
        if not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError(f"{source}/{stream_id}: empty or unordered camera clock")
        cameras.append(Camera(stream_id, label, fps, calibration, times[:frame_limit], len(times)))
    stop: int | None = max(int(camera.times_ns[-1]) for camera in cameras) if frame_limit is not None else None
    trajectory: Trajectory = read_trajectory(source / "mps/slam/closed_loop_trajectory.csv")
    hand_samples: HandSamples = read_hands(source / "mps/hand_tracking/hand_tracking_results.csv", trajectory, stop)
    return Scene(source, factory, cameras, trajectory, hand_samples, stop)
