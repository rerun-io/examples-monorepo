"""Aria Gen1 VRS and LaMAria ground-truth readers, in the rig frame the schema wants.

Everything here answers one question: what does a converter need in order to log
a LaMAria sequence as exoego:v2? Three shapes of answer live together because
they only make sense against each other:

* **The rig.** LaMAria's published calibration uses the **right IMU** as its
  body frame, so the rig frame is imu-right and every sensor pose is
  ``rig_T_sensor = imuR_T_device @ device_T_sensor``. Cameras land in simplecv's
  ``Fisheye62Parameters`` (Aria's FISHEYE624 minus the thin-prism terms, as in
  ``simplecv.data.hot3d_utils``), which is what ``log_pinhole`` consumes.
* **The streams.** Frames and IMU samples in VRS record order, on Aria's DEVICE
  clock in nanoseconds — the clock the pGT and the control points are stamped
  with, so nothing needs shifting anywhere downstream.
* **The ground truth.** The pGT is ``world_T_cam0`` at the slam-left frame
  times; the sparse file holds surveyed control points in LV95/LN02, which are
  6-digit coordinates and are therefore translated by ``CUSTOM_ORIGIN_XYZ``
  exactly as the official tooling does.

Reference: github.com/cvg/lamaria (``lamaria/utils/aria.py`` and
``lamaria/utils/constants.py``); the quaternion order and the transform chain
are cross-checked against a real VRS in ``tests/test_aria_vrs.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from projectaria_tools.core import data_provider
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration, ImuCalibration
from projectaria_tools.core.sensor_data import ImageData, ImageDataRecord, MotionData, TimeDomain
from projectaria_tools.core.stream_id import StreamId
from scipy.spatial.transform import Rotation
from serde import from_dict, serde
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion

from dataforge.logging_toolkit import ImuChannel

# ── streams ───────────────────────────────────────────────────────────────

AriaStreamId: TypeAlias = Literal["1201-1", "1201-2", "214-1", "1202-1", "1202-2"]
"""The five Aria Gen1 streams a LaMAria recording carries; the rest (magnetometer,
barometer, GPS, WiFi, Bluetooth) are not part of the benchmark."""

SLAM_LEFT_STREAM_ID: AriaStreamId = "1201-1"
"""camera-slam-left: 640x480 gray at 20 fps, and the frame the pGT poses."""
SLAM_RIGHT_STREAM_ID: AriaStreamId = "1201-2"
"""camera-slam-right: 640x480 gray at 20 fps."""
RGB_STREAM_ID: AriaStreamId = "214-1"
"""camera-rgb: 1408x1408 RGB at 10 fps, stored JPEG-compressed in the VRS."""
IMU_RIGHT_STREAM_ID: AriaStreamId = "1202-1"
"""imu-right at 1 kHz; its frame is the rig frame (LaMAria's body frame)."""
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

CAMERA_STREAM_IDS: tuple[AriaStreamId, ...] = (SLAM_LEFT_STREAM_ID, SLAM_RIGHT_STREAM_ID, RGB_STREAM_ID)
"""Camera streams in the order they become ``cam_00``, ``cam_01``, ``cam_02``."""

IMU_STREAM_IDS: tuple[AriaStreamId, ...] = (IMU_RIGHT_STREAM_ID, IMU_LEFT_STREAM_ID)
"""IMU streams in the order they become ``imu_00``, ``imu_01``; the first one *is* the rig."""

CUSTOM_ORIGIN_XYZ: Float64[ndarray, "3"] = np.array([2683594.412, 1247727.747, 417.307])
"""LV95/LN02 origin the official tooling subtracts from every surveyed coordinate
(``lamaria.utils.constants.CUSTOM_ORIGIN_COORDINATES``), so metres stay small."""

AriaImage: TypeAlias = UInt8[ndarray, "h w"] | UInt8[ndarray, "h w 3"]
"""One decoded frame: gray for the SLAM cameras, RGB for camera-rgb."""

TimedImage: TypeAlias = tuple[int, AriaImage]
"""A frame with the device-clock capture timestamp, in nanoseconds, that VRS recorded for it."""

ImuSamples: TypeAlias = tuple[ImuChannel, ImuChannel]
"""One IMU's gyro (rad/s) and accel (m/s^2) channels, in that order."""


# ── the device calibration, as a rig ──────────────────────────────────────


def open_vrs(path: Path) -> data_provider.VrsDataProvider:
    """Open one VRS file for reading.

    Args:
        path: The ``.vrs`` file.

    Returns:
        A provider over that single file.

    Raises:
        FileNotFoundError: If the file is missing, truncated or otherwise
            unreadable — the expected failure after an interrupted download.
            ``create_vrs_data_provider`` answers all of those with ``None``,
            which would fail much later as an ``AttributeError``.
    """
    provider: data_provider.VrsDataProvider | None = data_provider.create_vrs_data_provider(str(path))
    if provider is None:
        raise FileNotFoundError(f"no readable VRS at {path}; fetch it first (a resumed download leaves a partial file behind)")
    return provider


def device_calibration(provider: data_provider.VrsDataProvider) -> DeviceCalibration:
    """The factory calibration of an open VRS.

    Args:
        provider: An open provider.

    Returns:
        The device calibration.

    Raises:
        ValueError: If the file carries none; every LaMAria sequence does, so
            this means the recording is not what the caller thinks it is.
    """
    calibration: DeviceCalibration | None = provider.get_device_calibration()
    if calibration is None:
        raise ValueError("this VRS carries no device calibration")
    return calibration


def fisheye62_from_aria(calibration: CameraCalibration, *, rig_T_cam: Float64[ndarray, "4 4"], name: str) -> Fisheye62Parameters:
    """Convert one Aria FISHEYE624 camera into simplecv's Fisheye62 camera.

    Aria's ``get_projection_params()`` is 15 long:
    ``[f, cx, cy, k1..k6, p1, p2, s1..s4]`` — one focal length for both axes. The
    four thin-prism terms have no Fisheye62 slot and are dropped, which
    ``simplecv.data.hot3d_utils`` measured at under a pixel of reprojection error.

    Extrinsics are stored as ``world_R_cam``/``world_t_cam`` holding
    ``rig_T_cam``: ``log_pinhole`` logs ``cam_R_world``/``cam_t_world`` with
    ``from_parent=True``, i.e. the parent-to-child ``cam_T_rig``, which is
    exactly the inverse simplecv computes.

    Args:
        calibration: The camera's entry in the VRS device calibration.
        rig_T_cam: This camera's pose in the rig (imu-right) frame.
        name: Label to carry on the camera, e.g. ``"camera-slam-left"``.

    Returns:
        The same camera in the form ``log_pinhole`` consumes.

    Raises:
        ValueError: If the camera is not a FISHEYE624 (no Gen1 Aria stream is not).
    """
    model: str = calibration.get_model_name().name
    if model != "FISHEYE624":
        raise ValueError(f"{name} is a {model} camera; only Aria's FISHEYE624 maps onto Fisheye62")
    params: Float64[ndarray, "15"] = np.asarray(calibration.get_projection_params(), dtype=np.float64)
    size_wh: Int64[ndarray, "2"] = np.asarray(calibration.get_image_size(), dtype=np.int64)
    return Fisheye62Parameters(
        name=name,
        extrinsics=Extrinsics(world_R_cam=rig_T_cam[:3, :3].copy(), world_t_cam=rig_T_cam[:3, 3].copy()),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=float(params[0]),
            fl_y=float(params[0]),
            cx=float(params[1]),
            cy=float(params[2]),
            width=int(size_wh[0]),
            height=int(size_wh[1]),
        ),
        distortion=KannalaBrandtDistortion(
            k1=float(params[3]),
            k2=float(params[4]),
            k3=float(params[5]),
            k4=float(params[6]),
            k5=float(params[7]),
            k6=float(params[8]),
            p1=float(params[9]),
            p2=float(params[10]),
        ),
    )


@dataclass(frozen=True, slots=True)
class AriaRig:
    """One Aria Gen1's factory calibration, expressed in the rig frame.

    The rig frame is **imu-right**, which is what LaMAria's published
    calibration calls the body frame, so ``rig_T_cam`` here is directly
    comparable with that file's ``T_b_s``.
    """

    cameras: dict[AriaStreamId, Fisheye62Parameters]
    """Every camera stream the VRS carries, keyed by stream id; extrinsics hold ``rig_T_cam``."""
    rig_T_imu: dict[AriaStreamId, Float64[ndarray, "4 4"]]
    """Every IMU stream's pose in the rig frame; imu-right is the identity by construction."""

    @classmethod
    def from_provider(cls, provider: data_provider.VrsDataProvider) -> AriaRig:
        """Read the device calibration out of an open VRS and rebase it on imu-right.

        Args:
            provider: An open provider, e.g. from ``open_vrs``.

        Returns:
            The rig, with one entry per camera and IMU stream of a Gen1 Aria.

        Raises:
            ValueError: The file carries no calibration for one of those streams,
                which every LaMAria sequence does.
        """
        calibration: DeviceCalibration = device_calibration(provider)
        reference: ImuCalibration | None = calibration.get_imu_calib(STREAM_LABELS[IMU_RIGHT_STREAM_ID])
        if reference is None:
            raise ValueError(f"this VRS has no {STREAM_LABELS[IMU_RIGHT_STREAM_ID]} calibration, so it has no rig frame")
        # ``to_matrix()``, never the neighbouring ``to_quat()``: that returns
        # ``[[w, x, y, z]]`` while everything else here reads x, y, z, w.
        device_T_imu_right: Float64[ndarray, "4 4"] = np.asarray(reference.get_transform_device_imu().to_matrix(), dtype=np.float64)
        imu_right_T_device: Float64[ndarray, "4 4"] = np.linalg.inv(device_T_imu_right)

        cameras: dict[AriaStreamId, Fisheye62Parameters] = {}
        for stream_id in CAMERA_STREAM_IDS:
            camera: CameraCalibration | None = calibration.get_camera_calib(STREAM_LABELS[stream_id])
            if camera is None:
                raise ValueError(f"this VRS has no {STREAM_LABELS[stream_id]} calibration, so {stream_id} has no rig pose")
            device_T_cam: Float64[ndarray, "4 4"] = np.asarray(camera.get_transform_device_camera().to_matrix(), dtype=np.float64)
            cameras[stream_id] = fisheye62_from_aria(
                camera,
                rig_T_cam=imu_right_T_device @ device_T_cam,
                name=STREAM_LABELS[stream_id],
            )

        rig_T_imu: dict[AriaStreamId, Float64[ndarray, "4 4"]] = {}
        for stream_id in IMU_STREAM_IDS:
            imu: ImuCalibration | None = calibration.get_imu_calib(STREAM_LABELS[stream_id])
            if imu is None:
                raise ValueError(f"this VRS has no {STREAM_LABELS[stream_id]} calibration, so {stream_id} has no rig pose")
            device_T_imu: Float64[ndarray, "4 4"] = np.asarray(imu.get_transform_device_imu().to_matrix(), dtype=np.float64)
            rig_T_imu[stream_id] = imu_right_T_device @ device_T_imu
        return cls(cameras=cameras, rig_T_imu=rig_T_imu)


# ── streams ───────────────────────────────────────────────────────────────


def iter_frames(provider: data_provider.VrsDataProvider, stream_id: AriaStreamId) -> Iterator[TimedImage]:
    """Yield every frame of one camera stream in VRS record order.

    Record order is presentation order (verified strictly increasing on the real
    sequences), which is what an encoder needs: the mp4's Nth sample and
    ``frame_timestamps_ns``'s Nth value describe the same frame.

    Every frame is checked against the first one, so a stream that changes shape
    or dtype partway fails here rather than as a garbled video hundreds of
    frames later.

    Args:
        provider: An open provider.
        stream_id: A camera stream (``1201-1``, ``1201-2``, ``214-1``).

    Yields:
        ``(capture_timestamp_ns, image)`` with a ``uint8`` gray ``h w`` frame for
        the SLAM cameras and a ``uint8`` ``h w 3`` frame for camera-rgb.
    """
    stream: StreamId = StreamId(stream_id)
    first_hw: tuple[int, ...] | None = None
    for index in range(provider.get_num_data(stream)):
        frame: tuple[ImageData, ImageDataRecord] = provider.get_image_data_by_index(stream, index)
        # ``AriaImage`` on purpose, not an inline jaxtyping subscript: a subscript
        # is re-evaluated on every annotated assignment, and beartype then
        # compiles and caches a fresh checker per frame (see AGENTS.md).
        image: AriaImage = frame[0].to_numpy_array()
        if first_hw is None:
            first_hw = image.shape[:2]
        if image.dtype != np.uint8 or image.shape[:2] != first_hw:
            raise ValueError(f"{stream_id} frame {index} is {image.shape} {image.dtype}, not a uint8 {first_hw[0]}x{first_hw[1]} frame")
        yield int(frame[1].capture_timestamp_ns), image


def frame_timestamps_ns(provider: data_provider.VrsDataProvider, stream_id: AriaStreamId) -> Int64[ndarray, "n_frames"]:
    """Capture timestamps of one stream, in record order, on Aria's device clock.

    Args:
        provider: An open provider.
        stream_id: Any stream in the file.

    Returns:
        One nanosecond timestamp per record, ascending.

    Raises:
        ValueError: If the stream's timestamps are not strictly increasing, which
            would break the 1:1 mapping onto video samples.
    """
    times_ns: Int64[ndarray, "n_frames"] = np.asarray(
        provider.get_timestamps_ns(StreamId(stream_id), TimeDomain.DEVICE_TIME), dtype=np.int64
    )
    if times_ns.size > 1 and not bool((np.diff(times_ns) > 0).all()):
        raise ValueError(f"{stream_id} timestamps are not strictly increasing; record order is not presentation order")
    return times_ns


def read_imu(provider: data_provider.VrsDataProvider, stream_id: AriaStreamId) -> ImuSamples:
    """Read one IMU stream whole, at its native rate, without rectifying anything.

    Raw samples on purpose: rectification needs a per-timestamp online
    calibration that a base layer has no business inventing, and a consumer that
    wants it can apply ``ImuCalibration.raw_to_rectified_*`` to these.

    A record whose accel or gyro is flagged invalid is dropped from both
    channels, so the two share one timestamp vector (none of the LaMAria
    sequences examined so far contains one).

    Args:
        provider: An open provider.
        stream_id: ``1202-1`` (imu-right) or ``1202-2`` (imu-left).

    Returns:
        The gyro channel in rad/s and the accel channel in m/s^2.
    """
    stream: StreamId = StreamId(stream_id)
    times: list[int] = []
    gyro_radsec: list[list[float]] = []
    accel_msec2: list[list[float]] = []
    for index in range(provider.get_num_data(stream)):
        sample: MotionData = provider.get_imu_data_by_index(stream, index)
        if not (sample.accel_valid and sample.gyro_valid):
            continue
        times.append(int(sample.capture_timestamp_ns))
        gyro_radsec.append(list(sample.gyro_radsec))
        accel_msec2.append(list(sample.accel_msec2))
    times_ns: Int64[ndarray, "n_samples"] = np.asarray(times, dtype=np.int64)
    return (
        ImuChannel(times_ns=times_ns, values_xyz=np.asarray(gyro_radsec, dtype=np.float64).reshape(times_ns.size, 3)),
        ImuChannel(times_ns=times_ns, values_xyz=np.asarray(accel_msec2, dtype=np.float64).reshape(times_ns.size, 3)),
    )


# ── pseudo ground truth ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PseudoGt:
    """A sequence's pseudo ground-truth trajectory, as published.

    Rows are kept exactly as the file has them: the corpus repeats a pose
    verbatim while the device is static, and dropping those would silently
    change how a viewer draws (and how the official evaluator scores) a stop.
    """

    times_ns: Int64[ndarray, "n_poses"]
    """Device-clock timestamps, matching the slam-left frame times one for one."""
    world_T_cam0: Float64[ndarray, "n_poses 4 4"]
    """Pose of camera-slam-left in the GT world frame (LV95/LN02-aligned, or MPS's for R_01..R_10)."""


def read_pseudo_gt(path: Path) -> PseudoGt:
    """Read a ``ground_truth/pseudo_dense/<seq>.txt`` trajectory.

    Each line is ``ts_ns tx ty tz qx qy qz qw`` — a translation in metres and a
    quaternion in x, y, z, w order, the same convention the calibration files use.

    Args:
        path: The pGT text file.

    Returns:
        The trajectory, in file order.
    """
    rows: Float64[ndarray, "n_poses 8"] = np.loadtxt(path, dtype=np.float64).reshape(-1, 8)
    world_T_cam0: Float64[ndarray, "n_poses 4 4"] = np.tile(np.eye(4, dtype=np.float64), (rows.shape[0], 1, 1))
    world_T_cam0[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    world_T_cam0[:, :3, 3] = rows[:, 1:4]
    return PseudoGt(times_ns=rows[:, 0].astype(np.int64), world_T_cam0=world_T_cam0)


# ── control points ────────────────────────────────────────────────────────


@serde
@dataclass(frozen=True)
class SurveyedPoint:
    """One ``control_points`` entry as published, before the origin is subtracted.

    The entry's ``tag_id`` and ``image_names`` keys are left unread: the tag ids
    are an artefact of the detection pipeline, and ``images`` already carries the
    point-to-frame mapping the other way round.
    """

    measurement: list[float | None]
    """LV95/LN02 easting, northing, height in metres; a ``None`` height was never levelled."""
    uncertainty: list[float | None]
    """One-sigma survey uncertainty per axis in metres, ``None`` where the axis is unknown."""


@serde
@dataclass(frozen=True)
class DetectedPoint:
    """One ``images`` entry as published: which point was seen in which frame, and where."""

    timestamp: int
    """Device-clock capture time of the frame, in nanoseconds."""
    control_point: str
    """Name of the control point detected."""
    detection: list[float]
    """Pixel coordinates of the detection, ``[u, v]``."""


@dataclass(frozen=True, slots=True)
class ControlPoint:
    """One surveyed control point, in the translated metric world frame."""

    name: str
    """Survey name, e.g. ``"OB1878"``; unique within a sequence."""
    position_xyz_m: Float64[ndarray, "3"]
    """LV95/LN02 minus ``CUSTOM_ORIGIN_XYZ``. An unlevelled point gets ``z = 0.0``,
    i.e. the origin's height, and must be drawn as such rather than as a measurement."""
    has_height: bool
    """False when the survey never levelled this point; its ``z`` is a placeholder."""
    uncertainty_xyz_m: Float64[ndarray, "3"]
    """One-sigma survey uncertainty per axis in metres; ``NaN`` where unknown, never 0."""


@dataclass(frozen=True, slots=True)
class ControlPointDetection:
    """One control point seen in one frame."""

    stream_id: AriaStreamId
    """Camera that saw it, from the image name's prefix."""
    timestamp_ns: int
    """Device-clock capture time of that frame."""
    uv_px: Float64[ndarray, "2"]
    """Detection in pixels, in the camera's native (unrotated) image."""
    control_point: str
    """Name of the control point, keying into ``ControlPointSet.points``."""


@dataclass(frozen=True, slots=True)
class ControlPointSet:
    """A sequence's sparse ground truth: surveyed points and their 2D detections."""

    points: tuple[ControlPoint, ...]
    """Every surveyed point, in the file's order."""
    detections: tuple[ControlPointDetection, ...]
    """Every detection, sorted by stream then time, which is the order a columnar log wants."""


def read_control_points(path: Path) -> ControlPointSet:
    """Read a ``ground_truth/sparse/<seq>.json`` control-point file.

    The published coordinates are LV95/LN02, so they are six-digit numbers whose
    float32 round-off is centimetres; every position is translated by
    ``CUSTOM_ORIGIN_XYZ`` here, once, exactly as the official tooling does.

    Args:
        path: The sparse ground-truth JSON.

    Returns:
        The surveyed points and every detection of them.
    """
    document: dict = json.loads(path.read_text())
    points: list[ControlPoint] = []
    for name, entry in document["control_points"].items():
        surveyed: SurveyedPoint = from_dict(SurveyedPoint, entry)
        easting, northing, height = surveyed.measurement
        if easting is None or northing is None:
            raise ValueError(f"control point {name} has no horizontal measurement: {surveyed.measurement}")
        # Substituting the origin's own height for an unlevelled point makes its
        # translated z exactly 0.0, which is what ``has_height=False`` promises.
        published_xyz_m: Float64[ndarray, "3"] = np.array(
            [easting, northing, CUSTOM_ORIGIN_XYZ[2] if height is None else height], dtype=np.float64
        )
        points.append(
            ControlPoint(
                name=name,
                position_xyz_m=published_xyz_m - CUSTOM_ORIGIN_XYZ,
                has_height=height is not None,
                uncertainty_xyz_m=np.array([np.nan if value is None else value for value in surveyed.uncertainty], dtype=np.float64),
            )
        )

    detections: list[ControlPointDetection] = []
    for image_name, entry in document["images"].items():
        detected: DetectedPoint = from_dict(DetectedPoint, entry)
        detections.append(
            ControlPointDetection(
                stream_id=stream_id_from_image_name(image_name),
                timestamp_ns=detected.timestamp,
                uv_px=np.asarray(detected.detection, dtype=np.float64),
                control_point=detected.control_point,
            )
        )
    detections.sort(key=lambda detection: (detection.stream_id, detection.timestamp_ns))
    return ControlPointSet(points=tuple(points), detections=tuple(detections))


def stream_id_from_image_name(image_name: str) -> AriaStreamId:
    """The stream an extracted frame came from, e.g. ``1201-2-02100-1010.384.jpg`` → ``1201-2``.

    Raises:
        ValueError: If the name does not start with a known Aria stream id.
    """
    for stream_id in CAMERA_STREAM_IDS:
        if image_name.startswith(f"{stream_id}-"):
            return stream_id
    raise ValueError(f"{image_name} does not name an Aria camera stream")
