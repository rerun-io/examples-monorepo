"""Read-only Aria Gen2 Pilot streams and MPS tables on their native clocks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from simplecv.camera_parameters import Fisheye624Parameters
from simplecv.sensors.camera import fisheye624

from dataforge import aria, hands
from dataforge.aria import Trajectory, numeric_columns, poses_from_columns, read_trajectory
from dataforge.clocks import nearest_framesets
from dataforge.vrs import VrsFile
from dataforge.vrs_hevc import VrsHevcReader

CAMERAS: tuple[tuple[str, str, int], ...] = (
    ("214-1", "camera-rgb", 10),
    ("1201-1", "slam-front-left", 30),
    ("1201-2", "slam-front-right", 30),
    ("1201-3", "slam-side-left", 30),
    ("1201-4", "slam-side-right", 30),
)
"""Native stream IDs, factory labels and nominal container rates."""
FRAME_CLOCK_CAMERA: int = 1
"""slam-front-left supplies the shared frame_index clock."""
IMUS: tuple[tuple[aria.AriaStreamId, str], ...] = (("1202-1", "imu-left"), ("1202-2", "imu-right"))
"""Gen2 IMU stream IDs and factory labels; Gen1's ``aria.STREAM_LABELS`` has these two swapped."""
SEQUENCES: tuple[str, ...] = ("clean_0", "cook_0", "eat_0", "eat_1", "eat_2", "eat_3", "play_0", "play_1", "play_2", "play_3", "walk_0", "walk_1")
"""Release v1.0 sequence names."""


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
    frame_indices: Int64[ndarray, "n"]
    """Nearest slam-front-left frame per row, for the ``frame_index`` timeline."""


def hand_columns(side: hands.Side) -> dict[str, list[str]]:
    """One side's named MPS columns."""
    return {
        "confidence": [f"{side}_tracking_confidence"],
        "landmarks": [f"t{axis}_{side}_landmark_{joint}_device" for joint in range(21) for axis in "xyz"],
        "wrist": [f"{kind}{axis}_{side}_device_wrist" for kind, axes in (("t", "xyz"), ("q", "xyzw")) for axis in axes],
        "normals": [f"n{axis}_{side}_{part}_device" for part in ("palm", "wrist") for axis in "xyz"],
    }


@dataclass(frozen=True, slots=True)
class HandColumns:
    """One side's numeric MPS columns, before world transformation."""

    confidence: Float64[ndarray, "n"]
    """Shipped tracking confidence, including -1 for an absent hand."""
    landmarks: Float64[ndarray, "n 21 3"]
    """Device-frame landmark positions in metres."""
    wrist: Float64[ndarray, "n 7"]
    """Device-from-wrist translation and xyzw quaternion."""
    normals: Float64[ndarray, "n 2 3"]
    """Device-frame palm and wrist normals."""


def read_hands(path: Path, trajectory: Trajectory, frame_clock: Int64[ndarray, "f"], stop_ns: int | None = None) -> HandSamples:
    """Read every native MPS row and transform device landmarks at its own time."""
    names: list[str] = ["tracking_timestamp_us", *[name for side in hands.HAND_SIDES for group in hand_columns(side.name).values() for name in group]]
    table: Float64[ndarray, "n c"] = numeric_columns(path, names)
    position: dict[str, int] = {name: index for index, name in enumerate(names)}
    times: Int64[ndarray, "n"] = table[:, 0].astype(np.int64) * 1000
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: hand timestamps must increase")
    keep: Bool[ndarray, "n"] = np.ones(len(times), dtype=np.bool_) if stop_ns is None else times <= stop_ns
    times, table = times[keep], table[keep]
    poses: Float64[ndarray, "n 4 4"] = trajectory.at(times)
    valid_pose: Bool[ndarray, "n"] = np.isfinite(poses).all(axis=(1, 2))
    landmarks: Float32[ndarray, "n 2 21 3"] = np.full((len(times), 2, 21, 3), np.nan, dtype=np.float32)
    wrists: Float64[ndarray, "n 2 4 4"] = np.full((len(times), 2, 4, 4), np.nan)
    normals: Float64[ndarray, "n 2 2 3"] = np.full((len(times), 2, 2, 3), np.nan)
    scores: Float64[ndarray, "n 2"] = np.empty((len(times), 2))
    for index, side in enumerate(hands.HAND_SIDES):
        group: dict[str, list[str]] = hand_columns(side.name)
        columns: HandColumns = HandColumns(
            confidence=table[:, position[group["confidence"][0]]],
            landmarks=table[:, [position[name] for name in group["landmarks"]]].reshape(-1, 21, 3),
            wrist=table[:, [position[name] for name in group["wrist"]]],
            normals=table[:, [position[name] for name in group["normals"]]].reshape(-1, 2, 3),
        )
        scores[:, index] = columns.confidence
        present: Bool[ndarray, "n"] = valid_pose & (scores[:, index] != -1.0)
        rotation: Float64[ndarray, "p 3 3"] = poses[present, :3, :3]
        points: Float64[ndarray, "p 21 3"] = columns.landmarks[present]
        landmarks[present, index] = (np.einsum("nij,nkj->nki", rotation, points) + poses[present, None, :3, 3]).astype(np.float32)
        wrists[present, index] = poses[present] @ poses_from_columns(columns.wrist[present])
        normals[present, index] = np.einsum("nij,nkj->nki", rotation, columns.normals[present])
    positions, confidence = hands.confidence_rule(*hands.coco133_from_hands(landmarks, scores.astype(np.float32)))
    return HandSamples(times, positions, confidence, wrists, normals, scores, poses, nearest_framesets(frame_clock, times))


@dataclass(frozen=True, slots=True)
class Camera:
    """One native, unrotated camera and its full factory lens model."""

    stream_id: str
    """VRS identifier."""
    label: str
    """Factory label."""
    fps: int
    """Nominal container rate only; Rerun uses capture timestamps."""
    calibration: Fisheye624Parameters
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
    vrs: VrsFile
    """``video.vrs``, its description and record offsets read once."""
    calibration: aria.DeviceCalibration
    """Factory device frame shared by MPS and sensors."""
    cameras: list[Camera]
    """RGB, front stereo, side stereo."""
    trajectory: Trajectory
    """Full native trajectory, including interpolation neighbours outside preview."""
    hands: HandSamples
    """Native hand rows, optionally cut at the preview's last camera timestamp."""
    stop_ns: int | None
    """Preview cutoff, or full sequence."""

    @property
    def frame_clock(self) -> Int64[ndarray, "n"]:
        """Reference timestamps for every frame_index row."""
        return self.cameras[FRAME_CLOCK_CAMERA].times_ns


def read_scene(source: Path, frame_limit: int | None = None) -> Scene:
    """Open factory calibration and each clock independently; never decode VRS images."""
    vrs: VrsFile = VrsFile(source / "video.vrs")
    factory: aria.DeviceCalibration = aria.read_device_calibration(vrs)
    cameras: list[Camera] = []
    for stream_id, label, fps in CAMERAS:
        calibration: Fisheye624Parameters = factory.camera(label)
        reader: VrsHevcReader = VrsHevcReader(vrs, stream_id)
        width, height = vrs.image_size(stream_id)
        if (calibration.width, calibration.height) != (width, height):
            # camera-rgb's factory model is for the full 4032x3024 sensor; the stream is the
            # 2560x1920 "pov_downscaled" image, rescaled by the width ratio (the SDK's VRS
            # provider rounds that ratio to 0.635).
            scale: float = width / calibration.width
            if round(calibration.height * scale) != height:
                raise ValueError(f"{source}/{label}: stream {width}x{height} is not a uniform scale of {calibration.width}x{calibration.height}")
            calibration = fisheye624.rescale(calibration, width=width, height=height, scale=scale)
        times: Int64[ndarray, "n"] = reader.capture_timestamps()
        if not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError(f"{source}/{stream_id}: empty or unordered camera clock")
        cameras.append(Camera(stream_id, label, fps, calibration, times[:frame_limit], len(times)))
    stop: int | None = max(int(camera.times_ns[-1]) for camera in cameras) if frame_limit is not None else None
    trajectory: Trajectory = read_trajectory(source / "mps/slam/closed_loop_trajectory.csv")
    hand_samples: HandSamples = read_hands(source / "mps/hand_tracking/hand_tracking_results.csv", trajectory, cameras[FRAME_CLOCK_CAMERA].times_ns, stop)
    return Scene(source, vrs, factory, cameras, trajectory, hand_samples, stop)
