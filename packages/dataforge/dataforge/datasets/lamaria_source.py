"""LaMAria's own conventions on top of ``dataforge.aria``: the imu-right rig and the published ground truth.

* **The rig.** LaMAria's published calibration uses the **right IMU** as its
  body frame, so the rig frame is imu-right and every sensor pose is
  ``rig_T_sensor = imuR_T_device @ device_T_sensor``. Cameras land in simplecv's
  ``Fisheye62Parameters`` (Aria's FISHEYE624 minus the thin-prism terms), which
  is what ``log_pinhole`` consumes. Gen1 records its cameras sideways, so
  ``AriaRig.from_vrs(rotate_cw90=True)`` turns each of them a quarter turn
  clockwise for a converter that logs upright frames, and ``rotate_uv_cw90``
  turns published pixel coordinates the same way.
* **The ground truth.** The pGT is ``world_T_cam0`` at the slam-left frame
  times, on the DEVICE clock the VRS streams carry, so nothing needs shifting;
  the sparse file holds surveyed control points in LV95/LN02, which are 6-digit
  coordinates and are therefore translated by ``CUSTOM_ORIGIN_XYZ`` exactly as
  the official tooling does.

Reference: github.com/cvg/lamaria (``lamaria/utils/aria.py`` and
``lamaria/utils/constants.py``); the quaternion order and the transform chain
are cross-checked against a real VRS in ``tests/test_aria_vrs.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import field, serde
from simplecv.camera_parameters import Fisheye62Parameters, Fisheye624Parameters
from simplecv.sensors.camera import fisheye624

from dataforge.aria import (
    IMU_RIGHT_STREAM_ID,
    IMU_STREAM_IDS,
    RGB_STREAM_ID,
    SLAM_LEFT_STREAM_ID,
    SLAM_RIGHT_STREAM_ID,
    STREAM_LABELS,
    AriaStreamId,
    DeviceCalibration,
    read_device_calibration,
    rescale_to_stream,
)
from dataforge.records import read_json
from dataforge.vrs import VrsFile

CAMERA_STREAM_IDS: tuple[AriaStreamId, ...] = (SLAM_LEFT_STREAM_ID, SLAM_RIGHT_STREAM_ID, RGB_STREAM_ID)
"""Camera streams in the order they become ``cam_00``, ``cam_01``, ``cam_02``."""

CUSTOM_ORIGIN_XYZ: Float64[ndarray, "3"] = np.array([2683594.412, 1247727.747, 417.307])
"""LV95/LN02 origin the official tooling subtracts from every surveyed coordinate
(``lamaria.utils.constants.CUSTOM_ORIGIN_COORDINATES``), so metres stay small."""


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
    def from_vrs(cls, vrs: VrsFile, *, rotate_cw90: bool) -> AriaRig:
        """Read the device calibration out of a VRS and rebase it on imu-right.

        The result holds one entry per camera and IMU stream of a Gen1 Aria.
        Each camera is first fitted to its stream's resolution (camera-rgb is
        calibrated at 2880x2880 and recorded at 1408x1408).

        Args:
            vrs: An open VRS.
            rotate_cw90: Turn every camera a quarter turn clockwise, which a
                converter that logs upright frames must, since Aria Gen1 records
                its cameras sideways. ``fisheye624.rotate_cw90`` is what keeps the
                pixels and their calibration describing the same rays: the image
                size swaps, the principal point moves to ``(h - 1 - cy, cx)``, the
                tangential and thin-prism terms swap with it, and
                ``device_T_cam`` turns about the optical axis with its translation
                untouched. Left False the calibration is the factory's, which is
                what the published JSON describes and what the published ground
                truth is posed in.

        Raises:
            ValueError: The file carries no factory calibration, or none for one
                of those streams; every LaMAria sequence carries all of them.
        """
        calibration: DeviceCalibration = read_device_calibration(vrs)
        device_T_imu_right: Float64[ndarray, "4 4"] = calibration.imu(STREAM_LABELS[IMU_RIGHT_STREAM_ID]).matrix()
        imu_right_T_device: Float64[ndarray, "4 4"] = np.linalg.inv(device_T_imu_right)

        cameras: dict[AriaStreamId, Fisheye62Parameters] = {}
        for stream_id in CAMERA_STREAM_IDS:
            factory: Fisheye624Parameters = rescale_to_stream(calibration.camera(STREAM_LABELS[stream_id]), *vrs.image_size(stream_id))
            # Rotated before anything is read off it, so the intrinsics and the
            # pose that reach the rig both describe the frames a converter logs.
            camera: Fisheye624Parameters = fisheye624.rotate_cw90(factory) if rotate_cw90 else factory
            cameras[stream_id] = camera.to_fisheye62(rig_T_cam=imu_right_T_device @ camera.rig_T_cam.matrix())

        # imu-right *is* the rig frame, so its pose is the identity by construction:
        # stated exactly rather than as inv(T) @ T, which leaves ~1e-17 of residue.
        rig_T_imu: dict[AriaStreamId, Float64[ndarray, "4 4"]] = {IMU_RIGHT_STREAM_ID: np.eye(4, dtype=np.float64)}
        for stream_id in IMU_STREAM_IDS:
            if stream_id != IMU_RIGHT_STREAM_ID:
                rig_T_imu[stream_id] = imu_right_T_device @ calibration.imu(STREAM_LABELS[stream_id]).matrix()
        return cls(cameras=cameras, rig_T_imu=rig_T_imu)


@serde
@dataclass(frozen=True, slots=True)
class PublishedTransform:
    """A published rigid transform: a quaternion in x, y, z, w order and a translation."""

    qvec: tuple[float, float, float, float]
    """Rotation as ``[x, y, z, w]`` — pycolmap's order, which the official tooling reads it with."""
    tvec: tuple[float, float, float]
    """Translation in metres."""

    def to_matrix(self) -> Float64[ndarray, "4 4"]:
        """The same transform as a 4x4."""
        matrix: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(np.asarray(self.qvec, dtype=np.float64)).as_matrix()
        matrix[:3, 3] = self.tvec
        return matrix


@serde
@dataclass(frozen=True, slots=True)
class _PublishedCameraPose:
    """The one key of a ``cam0``/``cam1`` entry the gt layer reads; the rest is left to the file."""

    rig_T_cam: PublishedTransform = field(rename="T_b_s")
    """Published as ``T_b_s``; the body frame is imu-right, so this *is* ``rig_T_cam``."""


@serde
@dataclass(frozen=True, slots=True)
class _PublishedCalibration:
    """An ``aria_calibrations/<seq>.json``, read only as far as camera-slam-left's pose."""

    cam0: _PublishedCameraPose
    """camera-slam-left, the frame the pseudo ground truth poses."""


def read_rig_T_cam0(path: Path) -> Float64[ndarray, "4 4"]:
    """Read camera-slam-left's rig pose out of a published ``aria_calibrations/<seq>.json``.

    The published body frame is imu-right, which *is* the rig, so that file's
    ``cam0.T_b_s`` is ``rig_T_cam0`` as it stands — the one thing a gt layer
    needs from the file, and the one thing it can be had without the VRS. The
    intrinsics are read out of the VRS device calibration instead, which
    ``tests/test_aria_vrs.py`` cross-checks against this same file.

    Args:
        path: The sequence's published calibration JSON.

    Returns:
        ``rig_T_cam0``, from the entry's x, y, z, w quaternion and its
        translation in metres.

    Raises:
        ValueError: The file is not JSON, or holds no ``cam0.T_b_s`` of that shape.
    """
    published: _PublishedCalibration = read_json(path, _PublishedCalibration)
    return published.cam0.rig_T_cam.to_matrix()


def rotate_uv_cw90(uv_px: Float64[ndarray, "n_points 2"], *, native_height_px: int) -> Float64[ndarray, "n_points 2"]:
    """Turn pixel coordinates a quarter turn clockwise, with the image they name.

    ``np.rot90(image, -1)`` puts the pixel at row ``v``, column ``u`` at row
    ``u``, column ``h - 1 - v``, so this is that same map on ``(u, v)`` pairs.
    It is what keeps a published detection on its tag once the frames are logged
    upright, and it moves a native principal point onto the one
    ``from_vrs(rotate_cw90=True)`` reports.

    Args:
        uv_px: Coordinates in the camera's native image, ``[u, v]`` per row.
        native_height_px: Height of that native image, in pixels.

    Returns:
        The same coordinates in the rotated image, ``[u, v]`` per row.
    """
    return np.stack([native_height_px - 1.0 - uv_px[:, 1], uv_px[:, 0]], axis=1)


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
    quaternion in x, y, z, w order, the same convention the calibration files
    use. Rows stay in file order.
    """
    rows: Float64[ndarray, "n_poses 8"] = np.loadtxt(path, dtype=np.float64).reshape(-1, 8)
    world_T_cam0: Float64[ndarray, "n_poses 4 4"] = np.tile(np.eye(4, dtype=np.float64), (rows.shape[0], 1, 1))
    world_T_cam0[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    world_T_cam0[:, :3, 3] = rows[:, 1:4]
    return PseudoGt(times_ns=rows[:, 0].astype(np.int64), world_T_cam0=world_T_cam0)


# ── control points ────────────────────────────────────────────────────────


@serde
@dataclass(frozen=True, slots=True)
class SurveyedPoint:
    """One ``control_points`` entry as published, before the origin is subtracted.

    The entry's ``tag_id`` and ``image_names`` keys are left unread: the tag ids
    are an artefact of the detection pipeline, and ``images`` already carries the
    point-to-frame mapping the other way round.
    """

    measurement: tuple[float | None, float | None, float | None]
    """LV95/LN02 easting, northing, height in metres; a ``None`` height was never levelled."""
    uncertainty: tuple[float | None, float | None, float | None]
    """One-sigma survey uncertainty per axis in metres, ``None`` where the axis is unknown."""


@serde
@dataclass(frozen=True, slots=True)
class DetectedPoint:
    """One ``images`` entry as published: which point was seen in which frame, and where."""

    timestamp: int
    """Device-clock capture time of the frame, in nanoseconds."""
    control_point: str
    """Name of the control point detected."""
    detection: tuple[float, float]
    """Pixel coordinates of the detection, ``[u, v]``."""


@serde
@dataclass(frozen=True, slots=True)
class _ControlPointFile:
    """A ``ground_truth/sparse/<seq>.json`` as far as dataforge reads it; other keys are the file's own."""

    control_points: dict[str, SurveyedPoint]
    """Surveyed points by name."""
    images: dict[str, DetectedPoint]
    """Detections by the extracted frame's file name, whose prefix names the stream."""


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
    ``CUSTOM_ORIGIN_XYZ`` here, once, exactly as the official tooling does. The
    file is also checked against itself here: a detection that names a point
    the survey never published is an invariant of this one document, so it fails
    at the door rather than in whichever writer reads the set next.

    Raises:
        ValueError: The file is not JSON, does not hold the two maps, a point
            has no horizontal measurement, or a detection names an unpublished point.
    """
    document: _ControlPointFile = read_json(path, _ControlPointFile)
    points: list[ControlPoint] = []
    for name, surveyed in document.control_points.items():
        easting, northing, height = surveyed.measurement
        if easting is None or northing is None:
            raise ValueError(f"{path}: control point {name} has no horizontal measurement: {surveyed.measurement}")
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

    detections: list[ControlPointDetection] = [
        ControlPointDetection(
            stream_id=stream_id_from_image_name(image_name),
            timestamp_ns=detected.timestamp,
            uv_px=np.asarray(detected.detection, dtype=np.float64),
            control_point=detected.control_point,
        )
        for image_name, detected in document.images.items()
    ]
    unknown: set[str] = {detection.control_point for detection in detections} - set(document.control_points)
    if unknown:
        raise ValueError(f"{path}: control point detection(s) name {', '.join(sorted(unknown))}, which the survey does not publish")
    detections.sort(key=lambda detection: (detection.stream_id, detection.timestamp_ns))
    return ControlPointSet(points=tuple(points), detections=tuple(detections))


def stream_id_from_image_name(image_name: str) -> AriaStreamId:
    """The stream an extracted frame came from, e.g. ``1201-2-02100-1010.384.jpg`` → ``1201-2``."""
    for stream_id in CAMERA_STREAM_IDS:
        if image_name.startswith(f"{stream_id}-"):
            return stream_id
    raise ValueError(f"{image_name} does not name an Aria camera stream")
