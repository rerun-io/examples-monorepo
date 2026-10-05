"""What hot3d, the Aria Gen2 pilot and LaMAria share of an Aria recording: its device calibration, streams and projection.

* **The device calibration.** The VRS ``calib_json`` file tag, read with
  pyserde into simplecv's ``Fisheye624Parameters`` and ``SE3`` poses, the way
  projectaria-tools reads it (its per-device camera defaults and its stream
  rescale included), so every parameter and pose has the SDK's bits.
* **The streams.** Frames and IMU samples in VRS record order, read by
  ``dataforge.vrs``, on Aria's DEVICE clock in nanoseconds.
* **The projection.** ``project_to_calibration`` sends world points through the
  full FISHEYE624 model.

LaMAria's rig convention (imu-right as the body frame) and its ground-truth
formats live in ``datasets/lamaria_source.py``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import field, serde
from simplecv.camera_parameters import Fisheye624Parameters
from simplecv.se3 import SE3
from simplecv.sensors.camera import fisheye624

from dataforge.logging_toolkit import ImuChannel
from dataforge.records import decode
from dataforge.vrs import ImuRecords, VrsFile, VrsImageReader

# ── streams ───────────────────────────────────────────────────────────────

AriaStreamId: TypeAlias = Literal["1201-1", "1201-2", "214-1", "1202-1", "1202-2"]
"""The five Aria Gen1 streams a converter reads (Gen2 reuses the ids); the rest (magnetometer,
barometer, GPS, WiFi, Bluetooth) are left in the file."""

SLAM_LEFT_STREAM_ID: AriaStreamId = "1201-1"
"""camera-slam-left: 640x480 gray at 20 fps."""
SLAM_RIGHT_STREAM_ID: AriaStreamId = "1201-2"
"""camera-slam-right: 640x480 gray at 20 fps."""
RGB_STREAM_ID: AriaStreamId = "214-1"
"""camera-rgb: 1408x1408 RGB at 10 fps, stored JPEG-compressed in the VRS."""
IMU_RIGHT_STREAM_ID: AriaStreamId = "1202-1"
"""imu-right at 1 kHz (LaMAria's body frame)."""
IMU_LEFT_STREAM_ID: AriaStreamId = "1202-2"
"""imu-left at 800 Hz."""

STREAM_LABELS: dict[AriaStreamId, str] = {
    SLAM_LEFT_STREAM_ID: "camera-slam-left",
    SLAM_RIGHT_STREAM_ID: "camera-slam-right",
    RGB_STREAM_ID: "camera-rgb",
    IMU_RIGHT_STREAM_ID: "imu-right",
    IMU_LEFT_STREAM_ID: "imu-left",
}
"""Stream id → the label the device calibration is keyed by."""

IMU_STREAM_IDS: tuple[AriaStreamId, ...] = (IMU_RIGHT_STREAM_ID, IMU_LEFT_STREAM_ID)
"""IMU streams in the order they become ``imu_00``, ``imu_01``."""

AriaImage: TypeAlias = UInt8[ndarray, "h w"] | UInt8[ndarray, "h w 3"]
"""One decoded frame: gray for the SLAM cameras, RGB for camera-rgb."""

TimedImage: TypeAlias = tuple[int, AriaImage]
"""A frame with the device-clock capture timestamp, in nanoseconds, that VRS recorded for it."""

ImuSamples: TypeAlias = tuple[ImuChannel, ImuChannel]
"""One IMU's gyro (rad/s) and accel (m/s^2) channels, in that order."""


# ── the device calibration ────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """A camera's image size and field of view, which Aria calibration JSON leaves to the reader."""

    width: int
    """Factory image width in pixels."""
    height: int
    """Factory image height in pixels."""
    max_solid_angle: float
    """Half-angle of the projectable cone, in radians."""
    valid_radius: float | None
    """Pixel radius about the principal point that projections must fall inside; ``None`` for none."""


CAMERA_CONFIGS: dict[str, dict[str, CameraConfig]] = {
    "gen1": {
        "camera-slam-left": CameraConfig(640, 480, 1.4, 330.0),
        "camera-slam-right": CameraConfig(640, 480, 1.4, 330.0),
        "camera-rgb": CameraConfig(2880, 2880, 1.0, 1415.0),
    },
    "gen2": {
        "slam-front-left": CameraConfig(512, 512, 1.4, 300.0),
        "slam-front-right": CameraConfig(512, 512, 1.4, 300.0),
        "slam-side-left": CameraConfig(512, 512, 1.4, 300.0),
        "slam-side-right": CameraConfig(512, 512, 1.4, 300.0),
        "camera-rgb": CameraConfig(4032, 3024, 1.4, 2202.0),
    },
}
"""FISHEYE624 camera defaults per device generation, from projectaria-tools 2.3 ``ConfigData.cpp``.

The factory JSON only carries them under an optional ``ConfigData`` key, which
no Aria recording seen so far has."""

GEN1_STREAM_RESCALES: dict[tuple[str, tuple[int, int], tuple[int, int]], tuple[float, tuple[float, float]]] = {
    ("camera-rgb", (2880, 2880), (1408, 1408)): (0.5, (32.0, 32.0)),
    ("camera-rgb", (2880, 2880), (704, 704)): (0.25, (32.0, 32.0)),
}
"""(label, factory size, stream size) → (scale, crop offset) for Gen1 streams recorded below
sensor resolution, as projectaria-tools' VRS provider applies them (``AriaCalibRescaleAndCrop.cpp``)."""


@serde
@dataclass(frozen=True, slots=True)
class _JsonPose:
    """``T_Device_<Sensor>``: a translation and a ``[w, [x, y, z]]`` quaternion."""

    translation: tuple[float, float, float] = field(rename="Translation")
    """Metres."""
    unit_quaternion: tuple[float, tuple[float, float, float]] = field(rename="UnitQuaternion")
    """Scalar first, then the vector part."""

    def se3(self) -> SE3:
        """The pose, normalised as projectaria-tools does on load."""
        w, (x, y, z) = self.unit_quaternion
        return SE3.from_quaternion(np.array([w, x, y, z], dtype=np.float64), np.array(self.translation, dtype=np.float64))


@serde
@dataclass(frozen=True, slots=True)
class _JsonProjection:
    """A camera's lens model name and its parameters."""

    name: str = field(rename="Name")
    """``FisheyeRadTanThinPrism`` is FISHEYE624; eye-tracking cameras are ``KannalaBrandtK3``."""
    params: list[float] = field(rename="Params")
    """``[f, cx, cy, k1..k6, p1, p2, s1..s4]`` for FISHEYE624."""


@serde
@dataclass(frozen=True, slots=True)
class _JsonCameraConfig:
    """The optional per-camera ``ConfigData`` that overrides ``CAMERA_CONFIGS``."""

    width: int = field(rename="ImageWidth")
    """Image width in pixels."""
    height: int = field(rename="ImageHeight")
    """Image height in pixels."""
    max_solid_angle: float = field(rename="MaxSolidAngle")
    """Radians."""
    valid_radius: float | None = field(rename="ValidRadius", default=None)
    """Pixels."""


@serde
@dataclass(frozen=True, slots=True)
class _JsonCamera:
    """One ``CameraCalibrations`` entry, as far as a converter reads it."""

    label: str = field(rename="Label")
    """Sensor label."""
    projection: _JsonProjection = field(rename="Projection")
    """Lens model."""
    device_T_camera: _JsonPose = field(rename="T_Device_Camera")
    """Camera pose in the device frame."""
    config: _JsonCameraConfig | None = field(rename="ConfigData", default=None)
    """Image size and field of view, when the file states them."""


@serde
@dataclass(frozen=True, slots=True)
class _JsonImu:
    """One ``ImuCalibrations`` entry, as far as a converter reads it (its pose, not its rectification)."""

    label: str = field(rename="Label")
    """Sensor label."""
    device_T_imu: _JsonPose = field(rename="T_Device_Imu")
    """IMU pose in the device frame."""


@serde
@dataclass(frozen=True, slots=True)
class _JsonDeviceClass:
    """``DeviceClassInfo``: which Aria generation wrote the file."""

    device_class: str = field(rename="DeviceClass")
    """``Ariane``/``Aria…`` for Gen1, ``Oatmeal`` for Gen2."""


@serde
@dataclass(frozen=True, slots=True)
class _JsonDeviceCalibration:
    """A VRS ``calib_json`` tag; magnetometer, barometer, microphone and the rest are the file's own."""

    device_class: _JsonDeviceClass = field(rename="DeviceClassInfo")
    """Device generation."""
    cameras: list[_JsonCamera] = field(rename="CameraCalibrations")
    """Every camera, eye tracking included."""
    imus: list[_JsonImu] = field(rename="ImuCalibrations")
    """Every IMU."""


@dataclass(frozen=True, slots=True)
class DeviceCalibration:
    """An Aria factory calibration: its FISHEYE624 cameras and its IMU poses, in the device frame."""

    cameras: dict[str, Fisheye624Parameters]
    """FISHEYE624 cameras by label, at factory resolution; ``rig_T_cam`` is ``device_T_camera``."""
    device_T_imu: dict[str, SE3]
    """IMU poses by label."""

    @classmethod
    def from_json(cls, document: str, where: str) -> DeviceCalibration:
        """Read a ``calib_json`` document, taking image sizes and fields of view from ``CAMERA_CONFIGS``.

        Raises:
            ValueError: The document is not a device calibration, or names a
                device generation or a FISHEYE624 camera with no known defaults.
        """
        parsed: _JsonDeviceCalibration = decode(_JsonDeviceCalibration, document, source=f"{where}: invalid Aria device calibration")
        device_class: str = parsed.device_class.device_class
        generation: str | None = "gen1" if device_class[:4].lower() == "aria" else "gen2" if device_class.lower() == "oatmeal" else None
        if generation is None:
            raise ValueError(f"{where}: unknown Aria device class {device_class}")
        cameras: dict[str, Fisheye624Parameters] = {}
        for camera in parsed.cameras:
            if camera.projection.name != "FisheyeRadTanThinPrism":
                continue
            if len(camera.projection.params) != 15:
                raise ValueError(f"{where}: {camera.label} has {len(camera.projection.params)} FISHEYE624 parameters, not 15")
            config: CameraConfig | None = (
                CAMERA_CONFIGS[generation].get(camera.label)
                if camera.config is None
                else CameraConfig(camera.config.width, camera.config.height, camera.config.max_solid_angle, camera.config.valid_radius)
            )
            if config is None:
                raise ValueError(f"{where}: no {generation} camera defaults for {camera.label}")
            cameras[camera.label] = Fisheye624Parameters(
                name=camera.label,
                width=config.width,
                height=config.height,
                params=np.array(camera.projection.params, dtype=np.float64),
                rig_T_cam=camera.device_T_camera.se3(),
                max_solid_angle=config.max_solid_angle,
                valid_radius=config.valid_radius,
            )
        return cls(cameras=cameras, device_T_imu={imu.label: imu.device_T_imu.se3() for imu in parsed.imus})

    def camera(self, label: str) -> Fisheye624Parameters:
        """One FISHEYE624 camera; raises ValueError naming the label when the file has none."""
        if label not in self.cameras:
            raise ValueError(f"no FISHEYE624 {label} calibration in this device calibration")
        return self.cameras[label]

    def imu(self, label: str) -> SE3:
        """One IMU's ``device_T_imu``; raises ValueError naming the label when the file has none."""
        if label not in self.device_T_imu:
            raise ValueError(f"no {label} calibration in this device calibration")
        return self.device_T_imu[label]


def read_device_calibration(vrs: VrsFile) -> DeviceCalibration:
    """The factory calibration a VRS carries in its ``calib_json`` file tag.

    Raises:
        ValueError: The file carries none (HOT3D's Quest 3 recordings, for one).
    """
    if "calib_json" not in vrs.file_tags:
        raise ValueError(f"{vrs.path}: this VRS carries no device calibration")
    return DeviceCalibration.from_json(vrs.file_tags["calib_json"], str(vrs.path))


def rescale_to_stream(camera: Fisheye624Parameters, width: int, height: int) -> Fisheye624Parameters:
    """Fit a Gen1 factory camera to a stream recorded below sensor resolution, as the SDK's VRS provider does.

    Raises:
        ValueError: The sizes differ in a way ``GEN1_STREAM_RESCALES`` has no entry for.
    """
    if (camera.width, camera.height) == (width, height):
        return camera
    key: tuple[str, tuple[int, int], tuple[int, int]] = (camera.name, (camera.width, camera.height), (width, height))
    if key not in GEN1_STREAM_RESCALES:
        raise ValueError(f"{camera.name}: no rescale from {camera.width}x{camera.height} to the {width}x{height} stream")
    scale, origin_offset = GEN1_STREAM_RESCALES[key]
    return fisheye624.rescale(camera, width=width, height=height, scale=scale, origin_offset=origin_offset)


def project_to_calibration(
    calibration: Fisheye624Parameters,
    world_T_device: Float64[ndarray, "n 4 4"],
    positions: Float32[ndarray, "n 133 3"],
) -> Float64[ndarray, "n 133 2"]:
    """Project dense world joints through the shipped camera model.

    Args:
        calibration: Full lens model in the logged image orientation; ``rig_T_cam`` is ``device_T_camera``.
        world_T_device: Float64[ndarray, "n 4 4"] GT poses, NaN when missing.
        positions: Float32[ndarray, "n 133 3"] world metres.

    Returns:
        Float64[ndarray, "n 133 2"] pixels, with NaN for invalid joints.
    """
    pixels: Float64[ndarray, "n 133 2"] = np.full((*positions.shape[:2], 2), np.nan)
    device_T_camera: Float64[ndarray, "4 4"] = calibration.rig_T_cam.matrix()
    # Loop locals stay unannotated: beartype would rebuild a jaxtyping checker per frame in the dev env.
    for index, pose in enumerate(world_T_device):
        if not np.isfinite(pose).all():
            continue
        cam_T_world = np.linalg.inv(pose @ device_T_camera)
        joints = np.flatnonzero(np.isfinite(positions[index]).all(axis=1))
        # One joint at a time, as a 3-vector: a batched matmul sums in another order.
        points = np.array([cam_T_world[:3, :3] @ positions[index, joint] + cam_T_world[:3, 3] for joint in joints]).reshape(-1, 3)
        in_front = points[:, 2] > 0.0
        pixels[index, joints[in_front]] = fisheye624.project(calibration, points[in_front])
    pixels[~np.isfinite(pixels).all(axis=-1)] = np.nan
    return pixels


def project_frames(
    calibration: Fisheye624Parameters,
    world_T_device: Float64[ndarray, "n 4 4"],
    positions: Float32[ndarray, "n 133 3"],
) -> Float32[ndarray, "n 133 2"]:
    """``project_to_calibration`` for every frame at once, for corpus-scale layers.

    The same pixels to rounding: the batched transform sums in another order, so the last bits differ from the
    per-joint loop, which hot3d and aria_gen2_pilot keep for their byte-identical parity baselines.

    Args:
        calibration: Full lens model in the logged image orientation; ``rig_T_cam`` is ``device_T_camera``.
        world_T_device: Float64[ndarray, "n 4 4"] poses, NaN when missing.
        positions: Float32[ndarray, "n 133 3"] world metres, NaN when missing.

    Returns:
        Float32[ndarray, "n 133 2"] pixels, NaN for missing, rear and out-of-view joints.
    """
    cam_T_world: Float64[ndarray, "n 4 4"] = np.linalg.inv(world_T_device @ calibration.rig_T_cam.matrix())
    local: Float64[ndarray, "p 3"] = (
        np.einsum("nij,nkj->nki", cam_T_world[:, :3, :3], positions.astype(np.float64)) + cam_T_world[:, None, :3, 3]
    ).reshape(-1, 3)
    pixels: Float64[ndarray, "p 2"] = np.full((len(local), 2), np.nan)
    usable: Bool[ndarray, "p"] = np.isfinite(local).all(axis=1) & (local[:, 2] > 0.0)
    pixels[usable] = fisheye624.project(calibration, local[usable])
    return pixels.reshape(*positions.shape[:2], 2).astype(np.float32)


# ── streams ───────────────────────────────────────────────────────────────


def iter_frames(vrs: VrsFile, stream_id: AriaStreamId) -> Iterator[TimedImage]:
    """Yield every frame of one camera stream in VRS record order, decoded from its JPEG record.

    Record order is presentation order (verified strictly increasing on the real
    sequences), which is what an encoder needs: the mp4's Nth sample and
    ``frame_timestamps_ns``'s Nth value describe the same frame.

    Every frame is checked against the first one, so a stream that changes shape
    or dtype partway fails here rather than as a garbled video hundreds of
    frames later.

    Yields:
        ``(capture_timestamp_ns, image)``: a ``uint8`` gray ``h w`` frame for the
        SLAM cameras, a ``uint8`` ``h w 3`` one for camera-rgb.
    """
    # PyTurboJPEG is in the dataforge envs only; other envs import this module through the dataset registry.
    from turbojpeg import TJCS_GRAY, TJPF_GRAY, TJPF_RGB, TurboJPEG

    decoder: TurboJPEG = TurboJPEG()
    first_hw: tuple[int, ...] | None = None
    for index, record in enumerate(VrsImageReader(vrs, stream_id).images()):
        gray: bool = decoder.decode_header(record.image)[3] == TJCS_GRAY
        # ``AriaImage`` on purpose, not an inline jaxtyping subscript: a subscript
        # is re-evaluated on every annotated assignment, and beartype then
        # compiles and caches a fresh checker per frame (see AGENTS.md).
        image: AriaImage = decoder.decode(record.image, pixel_format=TJPF_GRAY)[:, :, 0] if gray else decoder.decode(record.image, pixel_format=TJPF_RGB)
        if first_hw is None:
            first_hw = image.shape[:2]
        if image.dtype != np.uint8 or image.shape[:2] != first_hw:
            raise ValueError(f"{stream_id} frame {index} is {image.shape} {image.dtype}, not a uint8 {first_hw[0]}x{first_hw[1]} frame")
        yield record.capture_timestamp_ns, image


def frame_timestamps_ns(vrs: VrsFile, stream_id: AriaStreamId) -> Int64[ndarray, "n_frames"]:
    """Capture timestamps of one JPEG camera stream, in record order, on Aria's device clock.

    Raises:
        ValueError: If they are not strictly increasing, which would break the
            1:1 mapping of a timestamp onto its video sample.
    """
    times_ns: Int64[ndarray, "n_frames"] = VrsImageReader(vrs, stream_id).capture_timestamps()
    if times_ns.size > 1 and not bool((np.diff(times_ns) > 0).all()):
        raise ValueError(f"{stream_id} timestamps are not strictly increasing; record order is not presentation order")
    return times_ns


def read_imu(vrs: VrsFile, stream_id: AriaStreamId, *, stop_ns: int | None = None) -> ImuSamples:
    """Read one IMU stream whole, at its native rate, without rectifying anything.

    Raw samples on purpose: rectification needs a per-timestamp online
    calibration that a base layer has no business inventing, and a consumer
    that wants it can apply the factory rectification to these.

    A record whose accel or gyro is flagged invalid is dropped from both
    channels, so the two share one timestamp vector (none of the LaMAria
    sequences examined so far contains one).

    Args:
        vrs: The open recording.
        stream_id: IMU stream to read.
        stop_ns: Last capture stamp to keep (a preview); ``None`` keeps every sample.

    Returns:
        The gyro channel in rad/s and the accel channel in m/s^2, in that order.
    """
    records: ImuRecords = vrs.imu(stream_id)
    valid: Bool[ndarray, "n_records"] = records.accel_valid & records.gyro_valid
    if stop_ns is not None:
        valid &= records.capture_timestamp_ns <= stop_ns
    times_ns: Int64[ndarray, "n_samples"] = records.capture_timestamp_ns[valid]
    return (
        ImuChannel(times_ns=times_ns, values_xyz=records.gyro_radsec[valid].astype(np.float64)),
        ImuChannel(times_ns=times_ns, values_xyz=records.accel_msec2[valid].astype(np.float64)),
    )


# ── MPS trajectories ──────────────────────────────────────────────────────

QUATERNION_NORM_TOLERANCE: float = 1e-5
"""MPS prints six decimals: wrist quaternions sit up to 1.2e-6 off unit norm, trajectory ones ~1e-9."""


def numeric_columns(path: Path, columns: list[str]) -> Float64[ndarray, "n c"]:
    """Read selected CSV stream columns; no dataset-owned JSON copy is made."""
    with path.open() as stream:
        header: list[str] = next(csv.reader(stream))
        try:
            selected: list[int] = [header.index(name) for name in columns]
            values: Float64[ndarray, "n c"] = np.loadtxt(stream, delimiter=",", usecols=selected, ndmin=2, dtype=np.float64)
            for index, name in enumerate(columns):
                if name == "tracking_timestamp_us":
                    stamps: Float64[ndarray, "n"] = values[:, index]
                    if not np.isfinite(stamps).all() or np.any(stamps != np.floor(stamps)):
                        raise ValueError("tracking_timestamp_us must contain finite integers")
                elif name.endswith("_tracking_confidence") and not np.isfinite(values[:, index]).all():
                    raise ValueError(f"{name} must be finite")
            return values
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error


def poses_from_columns(values: Float64[ndarray, "n 7"]) -> Float64[ndarray, "n 4 4"]:
    """Translation and xyzw quaternion to SE(3); a non-finite or non-unit row is missing (NaN), never renormalised into a pose."""
    valid: Bool[ndarray, "n"] = np.isfinite(values).all(axis=1)
    valid[valid] = np.abs(np.linalg.norm(values[valid, 3:], axis=1) - 1.0) <= QUATERNION_NORM_TOLERANCE
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
    """World-from-device matrices; rows ``poses_from_columns`` rejected are NaN."""
    quality: Float64[ndarray, "n"]
    """Shipped quality, including 0.0 and 0.5."""

    def __post_init__(self) -> None:
        if len(self.times_ns) != len(self.poses) or len(self.times_ns) != len(self.quality):
            raise ValueError("trajectory columns differ in length")
        if not len(self.times_ns) or np.any(np.diff(self.times_ns) <= 0):
            raise ValueError("trajectory timestamps must be nonempty and strictly increasing")

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
