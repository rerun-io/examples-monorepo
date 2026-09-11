"""One ``dataforge:v1`` segment as calibration, IMU, ground truth and grayscale framesets.

The feed is the Python half of the estimator's data contract: it reads a segment
either from a catalog URL or from a local ``.rrd`` served in process, and hands
the Rust core CPU grayscale images with integer-nanosecond timestamps.

Three decisions are frozen here because each one silently changes the numbers:

* **Pixels.** AV1 samples are muxed without re-encoding and decoded by
  single-threaded dav1d to ``gray8``. The MSD streams are limited-range
  ``yuv420p`` with flat chroma, so the ``gray8`` reformat (limited to full
  expansion) is what recovers the original grayscale; the raw Y plane is off by
  up to 17 LSB. dav1d also pads rows, so the decoded plane's ``line_size``
  exceeds the frame width and the copy honours it.
* **Round trips.** Video, IMU and ground truth for one time window arrive in one
  query each, and long segments are cut into windows on ``video_time`` whose
  edges land on frames that are keyframes in *every* camera, so a window decodes
  standalone. Decimated AV1 decode is a known upstream hazard: every frame is
  decoded and ``frame_stride`` only decides which framesets are yielded.
* **Rig shape.** A rig is not always four hardware-synced cameras fed at their
  stored resolution on one IMU clock. :class:`RigProfile` carries the four things
  that differ and default to what MSD is: which cameras of the rig are fed and in
  what order, the integer downscale applied to frames *and* intrinsics, whether
  the accelerometer has to be interpolated onto the gyroscope's clock, and how
  far apart two cameras' frames may be and still be one frameset.
* **Clocks.** ``video_time`` is the IMU's clock. Frames and ground truth reach it
  by adding ``cam_time_offset_ns``, which is what "added to a camera timestamp to
  reach the IMU clock" means and what basalt's own RoboCap reader does
  (``frameset_t = median(camera_t) + kCameraToImuOffsetNs``, the IMU untouched).
  MSD's offset is zero, so every MSD number is unchanged by this.
* **Geometry.** ``Pinhole:image_from_camera`` is column-major, ``Pinhole:resolution``
  is ``(width, height)`` while the decoded array is ``(height, width)``, and the
  camera ``Transform3D`` is ``ChildFromParent`` — that is ``cam_T_imu``, and it is
  inverted here. msd-g2 stores its images rotated into portrait with the
  calibration rotated to match, so nothing may assume a landscape frame or a
  shared orientation across cameras.
"""

import hashlib
import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from datafusion import col, lit
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_video_codec import CatalogCodecName, catalog_codec_name, wrap_mp4

from slam_rs.reference import ImuParameters, RobocapReference
from slam_rs.trajectory import ASSOCIATION_TOLERANCE_NS, Trajectory, empty_trajectory, shift_clock

RIG_ENTITY: str = "/world/rig_00"
"""Rig node of the ``exoego:v2`` tree; its reference frame is the IMU."""
IMU_ENTITY: str = "/world/rig_00/imu_00"
"""IMU node, whose transform is the identity because the IMU *is* the rig frame."""
TIMELINE: str = "video_time"
"""The one index both datasets carry: nanoseconds since ``property:capture:start_time_ns``."""
CHILD_FROM_PARENT: int = 2
"""``rr.TransformRelation.ChildFromParent``; the only relation the extrinsic inversion is valid for."""
DEFAULT_WINDOW_S: float = 60.0
"""Time window a long segment is cut into: a 7.6 s two-camera segment is 8.5 MB, so a 2,000 s one is not one query."""

CameraModelName: TypeAlias = Literal["kb4", "radtan8"]
"""Projection models V0 supports, named as basalt names them."""

_MODEL_BY_DISTORTION: dict[str, tuple[CameraModelName, int]] = {"kannala_brandt": ("kb4", 4), "brown_conrady": ("radtan8", 8)}
"""``simplecv.components.DistortionModel`` string to the model and the number of coefficients it uses."""


@dataclass(slots=True, frozen=True)
class CameraStatics:
    """One camera node's static components, exactly as the catalog stores them.

    This is the untouched read side: column-major matrices, ``(width, height)``
    resolution, the raw relation code and the full fixed-width coefficient list.
    :func:`camera_calib` is the only place the conversion rules live, which is
    what lets them be tested without a catalog.
    """

    distortion_model: str
    """``simplecv.components.DistortionModel``, e.g. ``kannala_brandt``.

    The projection model comes from here and not from the camera node's own
    ``camera_model`` string, which the RoboCap conversion predates and some
    writers omit.
    """
    distortion_coefficients: Float64[ndarray, " n_slots"]
    """Fixed-width coefficient list; the unused tail is zero."""
    image_from_camera: Float64[ndarray, " 9"]
    """``Pinhole:image_from_camera``, flat and **column-major**."""
    resolution_wh: Float64[ndarray, " 2"]
    """``Pinhole:resolution``, ``(width, height)`` in pixels."""
    transform_mat3x3: Float64[ndarray, " 9"]
    """Camera ``Transform3D:mat3x3``, flat and **column-major**."""
    transform_translation: Float64[ndarray, " 3"]
    """Camera ``Transform3D:translation``."""
    transform_relation: int
    """``Transform3D:relation``; must be :data:`CHILD_FROM_PARENT`."""
    distortion_valid_radius: float | None
    """basalt's ``rpmax``; present on msd-g2 only."""


@dataclass(slots=True, frozen=True)
class CameraCalib:
    """One camera as the estimator wants it: metric intrinsics and ``imu_T_cam``."""

    index: int
    """Camera number the estimator knows this camera by, matching the order of a frameset's images.

    On a rig fed whole that is also the rig index. Where
    :attr:`RigProfile.camera_names` feeds a subset it is the position in that
    list, and :attr:`SegmentFeed.camera_positions` maps it back to the rig — so
    the keypoints, the images and the blueprint views all speak the estimator's
    numbering and only the catalog reads speak the rig's.
    """
    width: int
    """Decoded frame width in pixels."""
    height: int
    """Decoded frame height in pixels."""
    fx: float
    """Focal length along image x, pixels."""
    fy: float
    """Focal length along image y, pixels."""
    cx: float
    """Principal point x, pixels."""
    cy: float
    """Principal point y, pixels."""
    model: CameraModelName
    """Projection model."""
    distortion: Float64[ndarray, " n_coeffs"]
    """Exactly the coefficients the model uses: 4 for kb4, ``k1 k2 p1 p2 k3 k4 k5 k6`` for radtan8."""
    distortion_valid_radius: float | None
    """basalt's ``rpmax``, when the recording carries one."""
    imu_T_cam: Float64[ndarray, "4 4"]
    """Camera pose in the IMU frame: the inverse of the stored ``ChildFromParent`` transform."""


@dataclass(slots=True, frozen=True)
class ImuCalib:
    """The IMU as the estimator wants it: noise model plus the body transform."""

    frequency_hz: float
    """Nominal update rate, from the reference manifest."""
    gyro_noise_std: float
    """Gyroscope noise density."""
    accel_noise_std: float
    """Accelerometer noise density."""
    gyro_bias_std: float
    """Gyroscope bias random walk."""
    accel_bias_std: float
    """Accelerometer bias random walk."""
    cam_time_offset_ns: int
    """Added to a camera timestamp to reach the IMU clock."""
    imu_T_body: Float64[ndarray, "4 4"]
    """Body pose in the IMU frame; the identity whenever the rig reference is the IMU."""


@dataclass(slots=True, frozen=True)
class ImuStream:
    """A segment's paired inertial measurements on one clock."""

    t_ns: Int64[ndarray, " n_samples"]
    """Sample timestamps, strictly increasing."""
    gyro_rad_s: Float64[ndarray, "n_samples 3"]
    """Angular velocity, rad/s."""
    accel_m_s2: Float64[ndarray, "n_samples 3"]
    """Linear acceleration, m/s^2."""

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])

    def between(self, first_ns: int, last_ns: int) -> "ImuStream":
        """The samples with ``first_ns < t <= last_ns``, half-open at the start."""
        keep: Bool[ndarray, " n_samples"] = (self.t_ns > first_ns) & (self.t_ns <= last_ns)
        return ImuStream(t_ns=self.t_ns[keep], gyro_rad_s=self.gyro_rad_s[keep], accel_m_s2=self.accel_m_s2[keep])


@dataclass(slots=True, frozen=True)
class Frameset:
    """One synchronised multi-camera capture, decoded to grayscale."""

    t_ns: int
    """Shared capture timestamp of every image, on the ``video_time`` clock."""
    images: list[UInt8[ndarray, "h w"]]
    """One C-contiguous grayscale image per camera, in rig camera order."""
    imu: ImuStream
    """Inertial samples since the previous frameset, running one sample past :attr:`t_ns`.

    The lead sample matters: a backend that integrates up to the frame time and
    blocks until it can deadlocks on the very first frameset if it is only ever
    given samples at or before that time.
    """
    ground_truth: Float64[ndarray, " 7"] | None
    """Nearest ground-truth pose as ``[tx, ty, tz, qw, qx, qy, qz]``, or None without a ``gt`` layer."""

    def image_digests(self) -> tuple[str, ...]:
        """Digest of each camera's gray8 bytes, in the same order as :attr:`images`.

        This is the unit the basalt C++ reference records in its
        ``frames.sha256`` (one ``t_ns,cam_index,sha256`` line per decoded frame),
        so the two decoders can be compared frame by frame rather than only in
        aggregate. Only the pixel-parity tests and the clip dumper ask for it,
        which is why it is hashed on demand rather than in the feed loop: that
        loop is what a gate's wall time and every fleet row's realtime factor
        measure, and 0.68 ms a frameset of SHA-256 over 2x960x960 is decode plus
        `track` and something else.

        The array's own buffer is hashed rather than a ``tobytes()`` copy of it —
        the same bytes and the same digest, and a non-contiguous frame raises
        here rather than being hashed in a different order.

        Returns:
            One hex digest per camera.
        """
        return tuple(hashlib.sha256(image).hexdigest() for image in self.images)

    def digest(self) -> str:
        """Digest of the timestamp and the per-camera digests: one value per frameset.

        Returns:
            One hex digest for the whole frameset.
        """
        rolled = hashlib.sha256(np.int64(self.t_ns).tobytes())
        for camera_digest in self.image_digests():
            rolled.update(bytes.fromhex(camera_digest))
        return rolled.hexdigest()


@dataclass(slots=True, frozen=True)
class LocalSegment:
    """A segment read from ``.rrd`` files on disk, served in process."""

    base_rrd: Path
    """Base layer: video, IMU and calibration."""
    gt_rrd: Path | None = None
    """Ground-truth layer; omitted when the dataset has none."""


@dataclass(slots=True, frozen=True)
class CatalogSegment:
    """A segment read from a running catalog server."""

    url: str
    """Catalog URL, e.g. ``rerun+http://<host>:9988``."""
    dataset_name: str
    """Dataset entry name on that server."""
    segment_id: str
    """Segment within the dataset."""


SegmentSource: TypeAlias = LocalSegment | CatalogSegment
"""Where a feed gets its data; the rest of the module does not care which."""


@dataclass(slots=True, frozen=True)
class RigProfile:
    """How one rig's recording has to be read, where reading it whole is wrong.

    The default is MSD: every camera the rig declares, at its stored resolution,
    on one hardware-synced clock, with both inertial channels already paired.
    RoboCap is none of those, and each departure is a number a caller must state
    rather than a branch the feed guesses.
    """

    camera_names: tuple[str, ...] | None = None
    """Cameras to feed, by their ``name`` static, in the order the estimator gets them; None feeds every camera in rig order."""
    downscale: int = 1
    """Integer factor applied to the decoded frames and to the intrinsics; 1 feeds the stored resolution."""
    interpolate_accel_onto_gyro: bool = False
    """Interpolate the accelerometer onto the gyroscope's timestamps instead of requiring one shared clock."""
    frameset_tolerance_ns: int = 0
    """How far a camera's frame may sit from the anchor camera's and still join that frameset; 0 demands identical timestamps."""
    video_time_is_absolute: bool = False
    """Whether ``video_time`` already **is** the device clock, so an export adds nothing to it.

    False is MSD, whose ``video_time`` is relative to
    ``property:capture:start_time_ns``; RoboCap records the device clock itself,
    and that is what every basalt CSV beside it carries.
    """

    def __post_init__(self) -> None:
        """Refuse a profile the feed cannot honour, before anything reads a recording with it.

        Raises:
            ValueError: If ``downscale`` is below one or ``frameset_tolerance_ns`` is negative.
        """
        if self.downscale < 1:
            raise ValueError(f"downscale must be at least 1; got {self.downscale}")
        if self.frameset_tolerance_ns < 0:
            raise ValueError(f"frameset_tolerance_ns cannot be negative; got {self.frameset_tolerance_ns}")

    @classmethod
    def from_robocap(cls, reference: RobocapReference) -> "RigProfile":
        """How the RoboCap rig has to be read, from the manifest's record of the C++ lane.

        Args:
            reference: The manifest's ``[robocap]`` table, which is where the five
                departures from MSD are written down.

        Returns:
            The profile the probe and both fleet tools open the rig with.
        """
        return cls(
            camera_names=reference.camera_names,
            downscale=reference.downscale,
            interpolate_accel_onto_gyro=reference.interpolate_accel_onto_gyro,
            frameset_tolerance_ns=reference.frameset_tolerance_ns,
            video_time_is_absolute=reference.video_time_is_absolute,
        )


MSD_RIG: RigProfile = RigProfile()
"""The Monado SLAM Dataset rigs: every camera, native resolution, one clock, paired inertial channels."""


def scale_principal_point(value: float, downscale: int) -> float:
    """One principal-point coordinate at ``1 / downscale`` of its resolution.

    The pixel's own centre is what scales, not its index: a pixel at ``c`` covers
    ``[c, c + 1)`` whose centre is ``c + 0.5``, and the downscaled pixel centre
    ``(c + 0.5) / d`` is at index ``(c + 0.5) / d - 0.5``. This is the convention
    the fork's ``basalt_convert_robocap_calib.py`` writes, and reproducing it from
    the recording's own native statics returns that file's digits exactly
    (``tests/test_catalog_feed.py``).
    """
    return (value + 0.5) / downscale - 0.5


def camera_calib(index: int, statics: CameraStatics, downscale: int = 1) -> CameraCalib:
    """Apply the catalog-to-estimator mapping rules to one camera's statics.

    The rules, each of which has cost someone a wrong trajectory: reshape
    ``image_from_camera`` column-major, read the resolution as ``(width, height)``,
    map the distortion model string and assert the coefficient tail is zero
    rather than truncating it, and invert the ``ChildFromParent`` transform to get
    ``imu_T_cam``.

    A ``downscale`` above one scales the resolution and the intrinsics to the
    frames the feed will actually decode, KB4's resolution-invariant coefficients
    untouched.

    Args:
        index: Camera index on the rig.
        statics: Raw static components of the camera node.
        downscale: Integer factor the frames are decoded at.

    Returns:
        The camera calibration in the estimator's conventions.

    Raises:
        ValueError: If the distortion model is unknown, the coefficient tail is
            non-zero, the transform relation is not ``ChildFromParent``, or
            ``downscale`` is below one.
    """
    if downscale < 1:
        raise ValueError(f"cam_{index:02d}: downscale must be at least 1; got {downscale}")
    width: int = int(statics.resolution_wh[0])
    height: int = int(statics.resolution_wh[1])
    if width // downscale < 1 or height // downscale < 1:
        raise ValueError(f"cam_{index:02d}: downscale {downscale} leaves nothing of the {width}x{height} frame")
    if statics.distortion_model not in _MODEL_BY_DISTORTION:
        raise ValueError(f"cam_{index:02d}: unsupported distortion model {statics.distortion_model!r}, known: {sorted(_MODEL_BY_DISTORTION)}")
    model: CameraModelName = _MODEL_BY_DISTORTION[statics.distortion_model][0]
    n_coeffs: int = _MODEL_BY_DISTORTION[statics.distortion_model][1]
    if statics.distortion_coefficients.shape[0] < n_coeffs:
        raise ValueError(f"cam_{index:02d}: {model} needs {n_coeffs} coefficients, got {statics.distortion_coefficients.shape[0]}")
    tail: Float64[ndarray, " n_tail"] = statics.distortion_coefficients[n_coeffs:]
    # A "kannala_brandt" string does not imply KB4 — Aria's Fisheye624 carries the
    # same string with eight live coefficients. Reject the tail, never truncate it.
    if not np.allclose(tail, 0.0):
        raise ValueError(f"cam_{index:02d}: {model} uses {n_coeffs} coefficients but the tail is non-zero: {tail.tolist()}")
    if statics.transform_relation != CHILD_FROM_PARENT:
        raise ValueError(f"cam_{index:02d}: Transform3D relation {statics.transform_relation} is not ChildFromParent({CHILD_FROM_PARENT})")
    k_matrix: Float64[ndarray, "3 3"] = statics.image_from_camera.reshape(3, 3, order="F")
    cam_R_imu: Float64[ndarray, "3 3"] = statics.transform_mat3x3.reshape(3, 3, order="F")
    imu_T_cam: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    imu_T_cam[:3, :3] = cam_R_imu.T
    imu_T_cam[:3, 3] = -cam_R_imu.T @ statics.transform_translation
    return CameraCalib(
        index=index,
        width=width // downscale,
        height=height // downscale,
        fx=float(k_matrix[0, 0]) / downscale,
        fy=float(k_matrix[1, 1]) / downscale,
        cx=scale_principal_point(float(k_matrix[0, 2]), downscale),
        cy=scale_principal_point(float(k_matrix[1, 2]), downscale),
        model=model,
        distortion=statics.distortion_coefficients[:n_coeffs].copy(),
        distortion_valid_radius=statics.distortion_valid_radius,
        imu_T_cam=imu_T_cam,
    )


def imu_calib(parameters: ImuParameters, imu_T_body: Float64[ndarray, "4 4"]) -> ImuCalib:
    """Combine the manifest's frozen noise model with the recording's IMU transform.

    Args:
        parameters: Frozen IMU parameters from the reference manifest.
        imu_T_body: Body pose in the IMU frame, the identity when the rig reference is the IMU.

    Returns:
        The IMU calibration the estimator is configured with.
    """
    return ImuCalib(
        frequency_hz=parameters.rate_hz,
        gyro_noise_std=parameters.gyro_noise_std,
        accel_noise_std=parameters.accel_noise_std,
        gyro_bias_std=parameters.gyro_bias_std,
        accel_bias_std=parameters.accel_bias_std,
        cam_time_offset_ns=parameters.cam_time_offset_ns,
        imu_T_body=imu_T_body,
    )


def _flat_float(column: pa.Array) -> Float64[ndarray, " n_values"]:
    """Every non-null value of a temporal component column, flat and float64.

    Rerun nests a component's values one list deep and the component's own arity
    another (``list<fixed_size_list<3>>`` for a ``Transform3D:translation``,
    ``list<double>`` for ``Scalars:scalars``), so the unwrapping is a loop rather
    than a fixed number of ``flatten`` calls.
    """
    values: pa.Array = column.drop_null()
    while pa.types.is_list(values.type) or pa.types.is_large_list(values.type) or pa.types.is_fixed_size_list(values.type):
        values = values.flatten()
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)


def _static_cell(statics: pa.Table, column: str) -> pa.Scalar:
    """Row zero of one static component column.

    A static is logged once, so row zero is the value — but a rig node that
    carries no statics at all reached ``statics[column][0]`` as an ``IndexError``
    out of pyarrow with nothing in it that says which component was being read.
    The three readers below differ only in what they make of the cell.

    Args:
        statics: Single-row table from ``filter_contents(...).reader(index=None)``.
        column: Component column name, e.g. ``/world/rig_00:reference``.

    Returns:
        The cell, valid.

    Raises:
        ValueError: If the column is absent, the table has no rows, or the cell is null.
    """
    if column not in statics.column_names:
        raise ValueError(f"static column {column} is missing")
    if statics.num_rows == 0:
        raise ValueError(f"static column {column} has no rows")
    cell: pa.Scalar = statics[column][0]
    if not cell.is_valid:
        raise ValueError(f"static column {column} is null")
    return cell


def _static_scalar(cell: pa.Scalar) -> object:
    """The one Python value in a static cell, whether or not Rerun wrapped it in a list."""
    return cell.values.to_pylist()[0] if isinstance(cell, pa.ListScalar | pa.LargeListScalar) else cell.as_py()


def _static_values(statics: pa.Table, column: str) -> Float64[ndarray, " n"]:
    """One static list component as a flat float64 array."""
    return np.asarray(_static_cell(statics, column).values.to_pylist(), dtype=np.float64).ravel()


def _static_string(statics: pa.Table, column: str) -> str:
    """One static string component, bare or wrapped in a single-element list."""
    value: object = _static_scalar(_static_cell(statics, column))
    if not isinstance(value, str):
        raise ValueError(f"static column {column} is not a string: {value!r}")
    return value


def _static_int(statics: pa.Table, column: str) -> int:
    """One static integer component, read without passing through float64.

    A device clock is around 1e13 ns today, which float64 holds exactly, but the
    margin to 2^53 is only three decades and the whole point of this value is that
    it is added to timestamps that must stay exact.
    """
    value: object = _static_scalar(_static_cell(statics, column))
    if not isinstance(value, int):
        raise ValueError(f"static column {column} is not an integer: {value!r}")
    return value


def read_camera_statics(statics: pa.Table, entity: str) -> CameraStatics:
    """Pull one camera node's static components out of a statics table.

    Args:
        statics: Single-row table from ``filter_contents(...).reader(index=None)``.
        entity: Camera node entity path, e.g. ``/world/rig_00/cam_00``.

    Returns:
        The raw statics, unconverted.
    """
    optional_radius: str = f"{entity}:distortion_valid_radius"
    return CameraStatics(
        distortion_model=_static_string(statics, f"{entity}/pinhole:simplecv.components.DistortionModel"),
        distortion_coefficients=_static_values(statics, f"{entity}/pinhole:simplecv.components.DistortionCoefficients"),
        image_from_camera=_static_values(statics, f"{entity}/pinhole:Pinhole:image_from_camera"),
        resolution_wh=_static_values(statics, f"{entity}/pinhole:Pinhole:resolution"),
        transform_mat3x3=_static_values(statics, f"{entity}:Transform3D:mat3x3"),
        transform_translation=_static_values(statics, f"{entity}:Transform3D:translation"),
        transform_relation=int(_static_values(statics, f"{entity}:Transform3D:relation")[0]),
        distortion_valid_radius=float(_static_values(statics, optional_radius)[0]) if optional_radius in statics.column_names else None,
    )


def decode_gray(mp4_bytes: bytes, downscale: int = 1) -> Iterator[UInt8[ndarray, "h w"]]:
    """Decode an in-memory MP4 to C-contiguous ``gray8`` frames, one decoder thread.

    ``reformat(format="gray8")`` performs the limited-to-full range expansion the
    MSD streams need; reading the raw Y plane instead is a systematic photometric
    shift of up to 17 LSB. dav1d pads each row, so the plane's ``line_size``
    exceeds the frame width and the visible columns are sliced out.

    The plane is read through the buffer protocol and the visible columns are
    copied once (T06): ``bytes(plane)`` used to copy the padded plane first, at
    0.751 ms a frame against 0.677 ms. The copy is explicit rather than
    :func:`numpy.ascontiguousarray`, which would return the slice untouched on a
    frame the decoder did not pad — a view into memory the decoder reuses for the
    next frame.

    A ``downscale`` above one asks the same ``swscale`` call for the smaller frame
    with ``SWS_AREA``: one conversion, colour and size together, which is the
    operation basalt's own RoboCap reader performs
    (``cpu_gray8_swscale_area_downscale3``). Converting first and resampling
    afterwards is a second, different filter and a different trajectory.

    Args:
        mp4_bytes: MP4 produced by :func:`wrap_mp4`.
        downscale: Integer factor to shrink each frame by.

    Yields:
        One grayscale image per frame, in decode order.

    Raises:
        ValueError: If ``downscale`` is below one.
    """
    if downscale < 1:
        raise ValueError(f"downscale must be at least 1; got {downscale}")
    container: av.container.InputContainer = av.open(BytesIO(mp4_bytes), mode="r")
    with container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "NONE"
        for frame in container.decode(stream):
            gray = (
                frame.reformat(format="gray8")
                if downscale == 1
                else frame.reformat(
                    width=max(frame.width // downscale, 1),
                    height=max(frame.height // downscale, 1),
                    format="gray8",
                    interpolation="AREA",
                )
            )
            plane = gray.planes[0]
            padded: UInt8[ndarray, "h stride"] = np.frombuffer(plane, dtype=np.uint8).reshape(gray.height, plane.line_size)
            yield padded[:, : gray.width].copy()


@dataclass(slots=True, frozen=True)
class _VideoIndex:
    """Frameset timing, read without touching a single encoded sample.

    A frameset is one row: its ``video_time``, and the frame each fed camera
    contributes to it. On a hardware-synced rig fed whole that is the identity —
    frameset ``i`` is frame ``i`` of every camera — and on RoboCap it is the
    nearest-match table basalt's reader builds.
    """

    t_ns: Int64[ndarray, " n_framesets"]
    """Frameset timestamps on ``video_time``: the median of the frames in each one."""
    frame_index: Int64[ndarray, "n_framesets n_cameras"]
    """Which frame of each fed camera belongs to each frameset, non-decreasing down each column."""
    camera_t_ns: tuple[Int64[ndarray, " n_frames"], ...]
    """Every fed camera's own frame timestamps, in the order they are fed."""
    keyframe: Bool[ndarray, " n_framesets"]
    """True where each fed camera's frame in this frameset is a keyframe, so a window may start there."""
    codec: CatalogCodecName
    """Codec of the elementary stream."""
    fps: int
    """Nominal frame rate, rounded from the timestamps."""


@dataclass(slots=True, frozen=True)
class SegmentFeed:
    """One segment's calibration, inertial data, ground truth and grayscale framesets."""

    segment_id: str
    """Segment this feed reads."""
    cameras: tuple[CameraCalib, ...]
    """Camera calibrations, in rig order; a frameset's images follow the same order."""
    imu: ImuCalib
    """IMU calibration, noise model included."""
    capture_start_time_ns: int
    """``property:capture:start_time_ns``: add it to a ``video_time`` to reach the absolute device clock."""
    export_offset_ns: int
    """What an exported trajectory adds to its ``video_time`` stamps to land on the clock the C++ CSVs use.

    :attr:`capture_start_time_ns` on a rig whose ``video_time`` is relative to it,
    and zero on one that records the device clock directly
    (:attr:`RigProfile.video_time_is_absolute`). The rule is the recording's, not
    the tool's: two tools reading it from the feed cannot disagree about it.
    """
    frame_t_ns: Int64[ndarray, " n_frames"]
    """Timestamp of every frameset on the inertial clock, before ``frame_stride`` is applied.

    That is ``video_time`` plus :attr:`ImuCalib.cam_time_offset_ns`, which is the
    clock the estimator, the exported trajectory and every basalt CSV are on.
    """
    camera_positions: tuple[int, ...]
    """Rig index of each fed camera, in the order a frameset's images arrive."""
    rig_cameras: int
    """Cameras the rig declares, of which :attr:`camera_positions` are the fed ones."""
    profile: RigProfile
    """How this rig had to be read, exactly as the caller stated it.

    Held rather than copied field by field: a rig knob is one declaration, and a
    feed that carried its own copy of two of them made the profile three
    declarations, of which only one is the caller's.
    """
    frame_stride: int
    """Yield every n-th frameset. Every frame is still decoded: decimated AV1 decode is unreliable."""
    dataset: DatasetEntry
    """Dataset the video samples are fetched from."""
    gt_dataset: DatasetEntry | None
    """Dataset the ground-truth poses come from; None when the source has no ``gt`` layer."""
    index: _VideoIndex
    """Frameset timing, the per-camera frames behind it, and the codec."""
    window_ns: int
    """Longest time window fetched in one round trip."""

    @property
    def has_ground_truth(self) -> bool:
        """Whether a ``gt`` layer is attached."""
        return self.gt_dataset is not None

    def stop_ns_after(self, max_framesets: int | None) -> int | None:
        """When a count of framesets runs out, as a time :meth:`framesets` can stop at.

        A caller counts framesets and the feed reads by time, so the count has to
        become the timestamp of the last frameset that will be yielded. Without
        it the feed is told to run to the end of the segment and fetches a whole
        window nothing below will decode — on the RoboCap rig, thirty seconds of
        four 1080p H.264 streams.

        Args:
            max_framesets: Framesets the caller will consume; None to run on.

        Returns:
            The last yielded frameset's timestamp, or None where the caller set
            no count.
        """
        if max_framesets is None:
            return None
        last_index: int = min(max(max_framesets - 1, 0) * self.frame_stride, len(self.frame_t_ns) - 1)
        return int(self.frame_t_ns[last_index])

    def imu_between(self, first_ns: int, last_ns: int) -> ImuStream:
        """Every inertial sample with ``first_ns <= t <= last_ns``, on the inertial clock."""
        return _read_imu(self.dataset, self.segment_id, self.profile.interpolate_accel_onto_gyro, first_ns, last_ns)

    def ground_truth_between(self, first_ns: int, last_ns: int) -> Trajectory:
        """Ground-truth rig poses over ``[first_ns, last_ns]`` of the inertial clock.

        The layer stores them on ``video_time``, like the frames, so the window is
        asked for in that clock and the answer comes back in the inertial one.
        Empty where there is nothing to give: no ``gt`` layer at all, or a window
        the layer does not cover. Whether the segment *has* a layer is
        :attr:`has_ground_truth`, which is the distinction a caller acts on;
        callers of this test :func:`len`.

        Raises:
            ValueError: If the layer's translation and rotation are not valid on
                the same rows, so no pose can be read off it
                (:func:`_rig_trajectory`).
        """
        if self.gt_dataset is None:
            return empty_trajectory()
        offset_ns: int = self.imu.cam_time_offset_ns
        return shift_clock(_read_ground_truth(self.gt_dataset, self.segment_id, first_ns - offset_ns, last_ns - offset_ns), offset_ns)

    def framesets(self, stop_ns: int | None = None) -> Iterator[Frameset]:
        """Decode the segment and yield one frameset at a time.

        Video, inertial samples and ground truth are all fetched one window at a
        time, so ``window_s`` really does bound memory and startup cost — reading
        a 586 s segment's IMU whole is 500k+ samples before the first frame comes
        out. Window edges land on frames that are keyframes in every camera, so
        each window decodes standalone; inside a window the cameras are decoded in
        lockstep, so only one frame per camera is ever resident.

        Each frameset carries the inertial samples since the previous one, running
        one sample past its own timestamp, and the nearest ground-truth pose.

        A caller that stops early states where, because the fetch is a window
        ahead of what it yields: a consumer breaking out of the loop has already
        paid for a whole window of encoded samples — 30 s x 30 fps x 4 cameras of
        1080p on a RoboCap run — that nothing will read.

        Args:
            stop_ns: Last frameset time worth reading, on the inertial clock; a
                window opening past it is not fetched at all. None reads to the
                end of the segment. Framesets up to the end of the window that
                covers it are still yielded, because a window is the unit that is
                read.

        Yields:
            Framesets in time order, every :attr:`frame_stride`-th one.

        Raises:
            ValueError: If a camera's decoder runs dry before the window ends, a
                decoded frame disagrees with the calibrated resolution, or a
                window's inertial read leaves a gap at the boundary.
        """
        # Each read reaches past both ends of its window. Forward, because a
        # frameset needs one sample beyond its own timestamp. Backward by a whole
        # frame period, because the first frameset of a window owns the samples
        # since the *previous* window's last frameset, which is one frame earlier —
        # a margin of a few IMU periods silently drops 16 ms of every boundary at
        # 54 Hz, which is what the gap check below is here to catch.
        frame_period_ns: int = int(1e9 / max(self.index.fps, 1))
        imu_period_ns: int = int(1e9 / max(self.imu.frequency_hz, 1.0))
        margin_ns: int = max(2 * frame_period_ns + 2 * imu_period_ns, 2_000_000)
        emitted_imu_t_ns: int = -(2**62)
        for start, stop in _window_bounds(self.index, self.window_ns):
            window_first_ns: int = int(self.frame_t_ns[start])
            if stop_ns is not None and window_first_ns > stop_ns:
                break
            window_last_ns: int = int(self.frame_t_ns[stop - 1])
            window_imu: ImuStream = self.imu_between(window_first_ns - margin_ns, window_last_ns + margin_ns)
            if len(window_imu) and emitted_imu_t_ns > -(2**62) and int(window_imu.t_ns[0]) > emitted_imu_t_ns + 1:
                raise ValueError(
                    f"{self.segment_id}: inertial gap at the window starting {window_first_ns} ns — "
                    f"the read begins at {int(window_imu.t_ns[0])} but the last emitted sample was {emitted_imu_t_ns}"
                )
            window_gt: Trajectory = self.ground_truth_between(window_first_ns - margin_ns, window_last_ns + margin_ns)

            decoders: list[Iterator[UInt8[ndarray, "h w"]]] = []
            for position in range(len(self.cameras)):
                samples, keyframes = self._fetch_samples(position, start, stop)
                decoders.append(decode_gray(wrap_mp4(samples, keyframes, fps=self.index.fps, codec=self.index.codec), self.profile.downscale))
            # One frame per camera resident, and one decode cursor per camera: a
            # camera that contributes no frame to this frameset still has its own
            # frames decoded in order, because dropping one breaks the next.
            decoded: list[UInt8[ndarray, "h w"] | None] = [None] * len(self.cameras)
            cursor: list[int] = [int(self.index.frame_index[start, position]) - 1 for position in range(len(self.cameras))]
            for frameset_index in range(start, stop):
                images: list[UInt8[ndarray, "h w"]] = []
                for position, (camera, decoder) in enumerate(zip(self.cameras, decoders, strict=True)):
                    rig_camera: int = self.camera_positions[position]
                    wanted: int = int(self.index.frame_index[frameset_index, position])
                    while cursor[position] < wanted:
                        image: UInt8[ndarray, "h w"] | None = next(decoder, None)
                        if image is None:
                            raise ValueError(f"{self.segment_id}: cam_{rig_camera:02d} ran out of frames at its frame {cursor[position] + 1}")
                        decoded[position] = image
                        cursor[position] += 1
                    current: UInt8[ndarray, "h w"] | None = decoded[position]
                    assert current is not None, f"cam_{rig_camera:02d} has no frame for frameset {frameset_index}"
                    if current.shape != (camera.height, camera.width):
                        raise ValueError(
                            f"{self.segment_id}: cam_{rig_camera:02d} decoded {current.shape}, calibration says {(camera.height, camera.width)}"
                        )
                    images.append(current)
                if frameset_index % self.frame_stride:
                    continue
                t_ns: int = int(self.frame_t_ns[frameset_index])
                lead_index: int = int(np.searchsorted(window_imu.t_ns, t_ns, side="right"))
                lead_ns: int = int(window_imu.t_ns[min(lead_index, len(window_imu) - 1)]) if len(window_imu) else t_ns
                frame_imu: ImuStream = window_imu.between(emitted_imu_t_ns, max(lead_ns, t_ns))
                if len(frame_imu):
                    emitted_imu_t_ns = int(frame_imu.t_ns[-1])
                yield Frameset(
                    t_ns=t_ns,
                    images=images,
                    imu=frame_imu,
                    ground_truth=_nearest_pose(window_gt, t_ns),
                )

    def _fetch_samples(self, position: int, start: int, stop: int) -> tuple[list[UInt8[ndarray, " n_sample_bytes"]], list[bool]]:
        """Encoded samples and keyframe flags of one fed camera, over the framesets ``[start, stop)``.

        The range is contiguous in that camera's own frames, from the frame the
        first frameset takes to the frame the last one does, so the decoder gets
        every frame the ones it must produce depend on.
        """
        camera_index: int = self.camera_positions[position]
        camera_t_ns: Int64[ndarray, " n_frames"] = self.index.camera_t_ns[position]
        first_frame: int = int(self.index.frame_index[start, position])
        last_frame: int = int(self.index.frame_index[stop - 1, position])
        entity: str = f"{RIG_ENTITY}/cam_{camera_index:02d}/pinhole/video"
        first_ns: int = int(camera_t_ns[first_frame])
        last_ns: int = int(camera_t_ns[last_frame])
        table: pa.Table = (
            self.dataset.filter_segments([self.segment_id])
            .filter_contents(entity)
            .reader(index=TIMELINE)
            .select(TIMELINE, f"{entity}:VideoStream:sample", f"{entity}:VideoStream:is_keyframe")
            .filter(col(TIMELINE).cast(pa.int64()).between(lit(first_ns), lit(last_ns)))
            .to_arrow_table()
        )
        times: Int64[ndarray, " n_window"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))
        if not np.array_equal(times, camera_t_ns[first_frame : last_frame + 1]):
            raise ValueError(
                f"{self.segment_id}: cam_{camera_index:02d} returned {len(times)} samples for its frames [{first_frame}, {last_frame}]"
            )
        # Arrow has no list<u8> -> binary cast, so slice the child buffer by the list
        # offsets. large_list keeps 64-bit offsets: one camera of a multi-hour session
        # exceeds the default int32's 2 GiB.
        blobs: pa.LargeListArray = table[1].combine_chunks().cast(pa.list_(pa.large_list(pa.uint8()))).flatten()
        data: UInt8[ndarray, " n_bytes"] = blobs.values.to_numpy(zero_copy_only=True)
        offsets: Int64[ndarray, " n_offsets"] = blobs.offsets.to_numpy(zero_copy_only=True)
        # Views into the column, not copies of it: `av.Packet` takes any buffer
        # and copies into its own, so a `tobytes()` here would be a second copy
        # of every encoded byte. The views keep `data` alive while they live.
        samples: list[UInt8[ndarray, " n_sample_bytes"]] = [data[begin:end] for begin, end in zip(offsets[:-1], offsets[1:], strict=True)]
        # is_keyframe is logged only on keyframes, so its validity is the flag.
        keyframes: list[bool] = table[2].combine_chunks().is_valid().to_pylist()
        return samples, keyframes


def _window_bounds(index: _VideoIndex, window_ns: int) -> list[tuple[int, int]]:
    """Cut the frame index into half-open ranges that each start on a shared keyframe."""
    keyframe_indices: list[int] = [position for position in np.flatnonzero(index.keyframe).tolist() if position > 0]
    bounds: list[tuple[int, int]] = []
    start: int = 0
    while start < len(index.t_ns):
        deadline: int = int(index.t_ns[start]) + window_ns
        stop: int = len(index.t_ns)
        for candidate in keyframe_indices:
            if candidate > start and int(index.t_ns[candidate]) >= deadline:
                stop = candidate
                break
        bounds.append((start, stop))
        start = stop
    return bounds


def _frame_nearest_anchor(times: Int64[ndarray, " n_frames"], cursor: int, anchor_t_ns: int, tolerance_ns: int) -> tuple[int | None, int]:
    """The frame one camera contributes to one anchor, and where its cursor goes if the frameset falls.

    The camera walks forward from ``cursor`` while the next frame is no farther
    from the anchor than the current one — ties take the later frame, which is
    basalt's ``<=`` (``dataset_io_robocap.cpp:422-426``) — and contributes that
    frame if it sits within ``tolerance_ns`` (inclusive, ``:427``).

    When it does not, the cursor moves only if that nearest frame is *earlier*
    than the anchor (``:428``): such a frame is farther from every later anchor
    still, so no anchor can ever take it, while a camera running ahead keeps its
    frame for the next anchor. The returned cursor is therefore where this camera
    stands once the frameset falls; a caller whose frameset stands ignores it and
    moves the cursor past the frame it took (``selected + 1``, ``:439``).

    Args:
        times: The camera's frame timestamps, in time order.
        cursor: The first frame no earlier frameset has consumed.
        anchor_t_ns: Camera 0's frame timestamp.
        tolerance_ns: How far a frame may sit from the anchor's and still join it.

    Returns:
        The frame this camera contributes, or ``None`` if it has none within the
        tolerance, and the cursor this camera stands on if the frameset falls.
    """
    index: int = cursor
    if index >= len(times):
        return None, cursor
    while index + 1 < len(times) and abs(int(times[index + 1]) - anchor_t_ns) <= abs(int(times[index]) - anchor_t_ns):
        index += 1
    if abs(int(times[index]) - anchor_t_ns) > tolerance_ns:
        return None, (index + 1 if int(times[index]) < anchor_t_ns else cursor)
    return index, cursor


def match_framesets(camera_t_ns: Sequence[Int64[ndarray, " n_frames"]], tolerance_ns: int) -> tuple[Int64[ndarray, " n_framesets"], Int64[ndarray, "n_framesets n_cameras"]]:
    """Group frames into framesets the way basalt's multi-camera reader does.

    Camera 0 is the anchor. Every other camera advances to the frame nearest the
    anchor's, and the frameset exists only if every camera has one within
    ``tolerance_ns``. Its timestamp is the median of the frames in it, which for
    an even number of cameras is the lower middle plus half the gap to the upper
    one — the arithmetic is basalt's, and reproducing it exactly is what makes the
    port's frameset times equal the C++'s to the nanosecond.

    A frame joins one frameset only: a complete frameset moves every non-anchor
    cursor past the frame it took (``selected + 1``,
    ``dataset_io_robocap.cpp:439``). An incomplete one moves a cursor only where
    that camera's nearest frame is *earlier* than the anchor and can therefore
    never partner a later one; a camera running ahead keeps its frame for the
    next anchor.

    An incomplete frameset whose anchor lies inside every camera's own span is an
    interior drop, and basalt allows one per thousand interior anchors before it
    calls the run unusable.

    Args:
        camera_t_ns: Each fed camera's frame timestamps, in time order.
        tolerance_ns: How far a frame may sit from the anchor's and still join it.

    Returns:
        The frameset timestamps, and the frame each camera contributes to each.

    Raises:
        ValueError: If no camera was given, a camera has no frames, the frameset
            timestamps do not strictly increase, too many interior framesets are
            incomplete, or no frameset is complete.
    """
    if not camera_t_ns:
        raise ValueError("a frameset needs at least one camera")
    for position, times in enumerate(camera_t_ns):
        if times.size == 0:
            raise ValueError(f"camera {position} has no frames, so it is not part of this recording")
    # The span every camera covers: only an anchor inside it can be expected to
    # have partners, so only a drop inside it counts against the run.
    overlap_start: int = max(int(times[0]) for times in camera_t_ns)
    overlap_end: int = min(int(times[-1]) for times in camera_t_ns)
    cursors: list[int] = [0] * len(camera_t_ns)
    t_ns: list[int] = []
    rows: list[list[int]] = []
    interior_anchors: int = 0
    interior_drops: int = 0
    for anchor_index, anchor_t_ns in enumerate(camera_t_ns[0].tolist()):
        interior: bool = overlap_start <= anchor_t_ns <= overlap_end
        interior_anchors += interior
        row: list[int] = [anchor_index]
        # Where each camera would land: basalt commits these to the cursors only
        # once the whole frameset stands.
        selected: list[int] = list(cursors)
        for position in range(1, len(camera_t_ns)):
            # The cursor this camera stands on if the frameset falls: a camera that
            # fell behind can never catch this anchor again, one running ahead keeps
            # its frame. On the frameset standing, that value is where it already was.
            index, cursors[position] = _frame_nearest_anchor(camera_t_ns[position], cursors[position], anchor_t_ns, tolerance_ns)
            if index is None:
                break
            selected[position] = index
            row.append(index)
        if len(row) != len(camera_t_ns):
            interior_drops += interior
            continue
        for position in range(1, len(camera_t_ns)):
            cursors[position] = selected[position] + 1
        members: list[int] = sorted(int(camera_t_ns[position][frame]) for position, frame in enumerate(row))
        middle: int = len(members) // 2
        frameset_t_ns: int = members[middle] if len(members) % 2 else members[middle - 1] + (members[middle] - members[middle - 1]) // 2
        if t_ns and frameset_t_ns <= t_ns[-1]:
            raise ValueError(f"frameset timestamps are not strictly increasing: {frameset_t_ns} follows {t_ns[-1]}")
        t_ns.append(frameset_t_ns)
        rows.append(row)
    # basalt's own allowance, in its own arithmetic: one in a thousand, at least one.
    allowed_drops: int = max(1, math.ceil(interior_anchors * 0.001))
    if interior_drops > allowed_drops:
        raise ValueError(
            f"{interior_drops} of {interior_anchors} interior framesets are incomplete, more than the {allowed_drops} "
            f"basalt allows: the cameras are not one recording within {tolerance_ns} ns"
        )
    if not rows:
        raise ValueError(f"no frameset has all {len(camera_t_ns)} cameras within {tolerance_ns} ns of camera 0")
    return np.array(t_ns, dtype=np.int64), np.array(rows, dtype=np.int64)


def _video_codec(table: pa.Table, entity: str) -> CatalogCodecName:
    """The codec every sample of one camera's stream is in.

    ``VideoStream:codec`` is logged once, as a static, so the value is row zero
    of the non-null codecs — and a recording that carries neither the samples nor
    the codec has to say which camera it was asked about rather than raise an
    index error out of pyarrow.

    Args:
        table: The three columns ``_read_video_index`` selects: the timeline, ``is_keyframe``, ``codec``.
        entity: The camera's video entity path, for the error.

    Returns:
        The codec name the sample wrapper needs.

    Raises:
        ValueError: If the stream has no samples or carries no codec.
    """
    if table.num_rows == 0:
        raise ValueError(f"{entity}: the recording carries no video samples")
    codecs: pa.Array = table[2].combine_chunks().drop_null().flatten()
    if len(codecs) == 0:
        raise ValueError(f"{entity}: the video stream carries no codec, so its samples cannot be decoded")
    return catalog_codec_name(int(codecs[0].as_py()))


def _shared_codec(per_camera: Sequence[tuple[int, CatalogCodecName]], segment_id: str) -> CatalogCodecName:
    """The one codec every fed camera's stream is in.

    :attr:`_VideoIndex.codec` names the codec :meth:`SegmentFeed._fetch_samples`
    muxes *every* camera's samples under, so a rig whose cameras disagree cannot
    be decoded from one index and has to name the two that differ rather than
    silently decode three of four streams as the fourth.

    Args:
        per_camera: ``(camera_index, codec)`` in feed order, at least one entry.
        segment_id: Segment the cameras belong to, for the error.

    Returns:
        The codec the sample wrapper needs.

    Raises:
        ValueError: If two fed cameras carry different codecs.
    """
    first_camera, codec = per_camera[0]
    for camera_index, other in per_camera[1:]:
        if other != codec:
            raise ValueError(
                f"{segment_id}: cam_{camera_index:02d} is {other} where cam_{first_camera:02d} is {codec}; "
                f"one video index carries one codec for every camera"
            )
    return codec


def _read_video_index(dataset: DatasetEntry, segment_id: str, camera_positions: Sequence[int], tolerance_ns: int) -> _VideoIndex:
    """Frameset timing, per-camera frames, shared keyframes and codec, fetched without any sample bytes."""
    per_camera_times: list[Int64[ndarray, " n_frames"]] = []
    per_camera_keyframe: list[Bool[ndarray, " n_frames"]] = []
    per_camera_codec: list[tuple[int, CatalogCodecName]] = []
    for camera_index in camera_positions:
        entity: str = f"{RIG_ENTITY}/cam_{camera_index:02d}/pinhole/video"
        table: pa.Table = (
            dataset.filter_segments([segment_id])
            .filter_contents(entity)
            .reader(index=TIMELINE)
            .select(TIMELINE, f"{entity}:VideoStream:is_keyframe", f"{entity}:VideoStream:codec")
            .to_arrow_table()
        )
        per_camera_times.append(np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64())))
        per_camera_keyframe.append(np.asarray(table[1].combine_chunks().is_valid().to_numpy(zero_copy_only=False), dtype=bool))
        per_camera_codec.append((camera_index, _video_codec(table, entity)))
    if not per_camera_codec:
        raise ValueError(f"{segment_id}: no camera was selected, so no video columns were read")
    codec: CatalogCodecName = _shared_codec(per_camera_codec, segment_id)
    if tolerance_ns == 0:
        # MSD is hardware-synced: a frameset is "all cameras at the same
        # video_time", and the identity table is what a matcher would return.
        for position, times in enumerate(per_camera_times[1:], start=1):
            if not np.array_equal(times, per_camera_times[0]):
                raise ValueError(
                    f"{segment_id}: cam_{camera_positions[position]:02d} timestamps differ from cam_{camera_positions[0]:02d}; "
                    f"this rig needs RigProfile.frameset_tolerance_ns"
                )
        frameset_t_ns: Int64[ndarray, " n_framesets"] = per_camera_times[0]
        frame_index: Int64[ndarray, "n_framesets n_cameras"] = np.tile(
            np.arange(len(frameset_t_ns), dtype=np.int64).reshape(-1, 1), (1, len(per_camera_times))
        )
    else:
        frameset_t_ns, frame_index = match_framesets(per_camera_times, tolerance_ns)
    if len(frameset_t_ns) < 2:
        raise ValueError(f"{segment_id}: {len(frameset_t_ns)} framesets is not a segment")
    keyframe: Bool[ndarray, " n_framesets"] = np.ones(len(frameset_t_ns), dtype=bool)
    for position, flags in enumerate(per_camera_keyframe):
        keyframe &= flags[frame_index[:, position]]
    fps: int = max(1, round((len(frameset_t_ns) - 1) * 1e9 / float(frameset_t_ns[-1] - frameset_t_ns[0])))
    return _VideoIndex(
        t_ns=frameset_t_ns,
        frame_index=frame_index,
        camera_t_ns=tuple(per_camera_times),
        keyframe=keyframe,
        codec=codec,
        fps=fps,
    )


def _nearest_pose(trajectory: Trajectory, t_ns: int, tolerance_ns: int = ASSOCIATION_TOLERANCE_NS) -> Float64[ndarray, " 7"] | None:
    """The pose closest in time to ``t_ns`` as ``[tx, ty, tz, qw, qx, qy, qz]``, or None if none is close enough.

    The tolerance is the gate's own association tolerance. Without it a frameset
    outside the ground truth's span would silently receive a stale pose: on the
    Index smoke segment the ground truth starts 17.5 ms after ``video_time`` zero,
    so the very first frameset has no truth and must say so rather than borrow one.
    """
    if len(trajectory) == 0:
        return None
    nearest: int = int(np.abs(trajectory.t_ns - t_ns).argmin())
    if abs(int(trajectory.t_ns[nearest]) - t_ns) > tolerance_ns:
        return None
    return np.concatenate([trajectory.position_m[nearest], trajectory.quaternion_wxyz[nearest]])


def _read_imu(dataset: DatasetEntry, segment_id: str, interpolate_accel: bool, first_ns: int, last_ns: int) -> ImuStream:
    """Gyroscope and accelerometer over one time window, on the gyroscope's clock.

    ``video_time`` **is** the inertial clock, so nothing is shifted here; the
    frames come to it (:attr:`SegmentFeed.frame_t_ns`).
    """
    table: pa.Table = (
        dataset.filter_segments([segment_id])
        .filter_contents([f"{IMU_ENTITY}/gyro", f"{IMU_ENTITY}/accel"])
        .reader(index=TIMELINE)
        .filter(col(TIMELINE).cast(pa.int64()).between(lit(first_ns), lit(last_ns)))
        .to_arrow_table()
    )
    row_t_ns: Int64[ndarray, " n_rows"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))
    order: Int64[ndarray, " n_rows"] = np.argsort(row_t_ns, kind="stable")
    row_t_ns = row_t_ns[order]
    streams: dict[str, tuple[Int64[ndarray, " n_samples"], Float64[ndarray, "n_samples 3"]]] = {}
    for sensor in ("gyro", "accel"):
        column: pa.Array = table[f"{IMU_ENTITY}/{sensor}:Scalars:scalars"].combine_chunks().take(pa.array(order))
        valid: Bool[ndarray, " n_rows"] = column.is_valid().to_numpy(zero_copy_only=False)
        values: Float64[ndarray, "n_samples 3"] = _flat_float(column).reshape(-1, 3)
        streams[sensor] = (row_t_ns[valid], values)
    gyro_t_ns: Int64[ndarray, " n_samples"] = streams["gyro"][0]
    accel_t_ns: Int64[ndarray, " n_samples"] = streams["accel"][0]
    if gyro_t_ns.size and not bool(np.all(np.diff(gyro_t_ns) > 0)):
        raise ValueError(f"{segment_id}: IMU timestamps are not strictly increasing")
    # MSD logs both sensors on identical timestamps, so pairing is an assertion.
    # RoboCap's two channels run on their own clocks (10,745 gyro against 10,751
    # accel on session 15), and basalt's reader interpolates the accelerometer
    # onto the gyroscope's timestamps; the core only ever sees the paired form.
    if not interpolate_accel:
        if not np.array_equal(gyro_t_ns, accel_t_ns):
            raise ValueError(
                f"{segment_id}: {len(gyro_t_ns)} gyro and {len(accel_t_ns)} accel samples are not on identical timestamps; pair them before feeding"
            )
        return ImuStream(t_ns=gyro_t_ns, gyro_rad_s=streams["gyro"][1], accel_m_s2=streams["accel"][1])
    return pair_accel_onto_gyro(gyro_t_ns, streams["gyro"][1], accel_t_ns, streams["accel"][1])


def pair_accel_onto_gyro(
    gyro_t_ns: Int64[ndarray, " n_gyro"],
    gyro_rad_s: Float64[ndarray, "n_gyro 3"],
    accel_t_ns: Int64[ndarray, " n_accel"],
    accel_m_s2: Float64[ndarray, "n_accel 3"],
) -> ImuStream:
    """Linearly interpolate the accelerometer onto the gyroscope's timestamps.

    A gyroscope sample outside the accelerometer's own span is dropped rather
    than held at an endpoint: ``numpy.interp`` clamps, which would feed the
    estimator a constant acceleration over a stretch it has no measurement for.
    This is basalt's rule for RoboCap, whose file reader skips a gyroscope sample
    that has no accelerometer sample on both sides of it.

    Args:
        gyro_t_ns: Gyroscope timestamps, strictly increasing.
        gyro_rad_s: Angular velocity, rad/s.
        accel_t_ns: Accelerometer timestamps, non-decreasing; a repeated one keeps its first sample.
        accel_m_s2: Linear acceleration, m/s^2.

    Returns:
        One stream on the gyroscope's clock, covering only the overlap.

    Raises:
        ValueError: If a channel is too short to interpolate with, or the two
            spans do not overlap, so the paired stream would be empty.
    """
    # basalt deduplicates both raw channels before it pairs them
    # (`sort_and_deduplicate`, `dataset_io_robocap.cpp:472`), keeping the first
    # sample of each equal-timestamp run. That is what its `interval == 0` guard
    # reads as alpha 0, and it is the whole difference from `numpy.interp`, which
    # takes the second of a duplicated pair and then interpolates the next
    # gyroscope sample from the wrong end of the gap.
    first_of_run: Bool[ndarray, " n_accel"] = np.ones(accel_t_ns.size, dtype=bool)
    first_of_run[1:] = np.diff(accel_t_ns) != 0
    accel_t_ns = accel_t_ns[first_of_run]
    accel_m_s2 = accel_m_s2[first_of_run]
    if gyro_t_ns.size == 0 or accel_t_ns.size < 2:
        raise ValueError(
            f"pairing needs a gyroscope sample and two accelerometer samples to interpolate between; "
            f"got {gyro_t_ns.size} gyro and {accel_t_ns.size} accel samples"
        )
    inside: Bool[ndarray, " n_gyro"] = (gyro_t_ns >= accel_t_ns[0]) & (gyro_t_ns <= accel_t_ns[-1])
    if not bool(inside.any()):
        # basalt errors instead of handing the estimator an empty inertial stream
        # (`dataset_io_robocap.cpp:496`); a rig with no IMU is not this rig.
        raise ValueError(
            f"the two inertial channels do not overlap, so nothing pairs: the gyroscope spans "
            f"{int(gyro_t_ns[0])}..{int(gyro_t_ns[-1])} ns and the accelerometer {int(accel_t_ns[0])}..{int(accel_t_ns[-1])} ns"
        )
    paired_t_ns: Int64[ndarray, " n_paired"] = gyro_t_ns[inside]
    interpolated: Float64[ndarray, "n_paired 3"] = np.column_stack(
        [np.interp(paired_t_ns, accel_t_ns, accel_m_s2[:, axis]) for axis in range(accel_m_s2.shape[1])]
    )
    return ImuStream(t_ns=paired_t_ns, gyro_rad_s=gyro_rad_s[inside], accel_m_s2=interpolated)


def _read_ground_truth(dataset: DatasetEntry, segment_id: str, first_ns: int, last_ns: int) -> Trajectory:
    """Ground-truth rig poses over one window on ``video_time``, converted from Rerun's XYZW to w-first.

    Empty when the layer carries no rig transform at all and when this window
    holds no pose: the two are the same answer to "what is the truth here", and
    the layer's own absence is :attr:`SegmentFeed.has_ground_truth`.
    """
    table: pa.Table = (
        dataset.filter_segments([segment_id])
        .filter_contents([RIG_ENTITY])
        .reader(index=TIMELINE)
        .filter(col(TIMELINE).cast(pa.int64()).between(lit(first_ns), lit(last_ns)))
        .to_arrow_table()
    )
    return _rig_trajectory(table, segment_id)


def _rig_trajectory(table: pa.Table, segment_id: str) -> Trajectory:
    """One window of the ``gt`` layer's rig transforms as a trajectory, Rerun's XYZW converted to w-first.

    Empty when the layer carries neither component and when the window holds no
    pose at all.

    Args:
        table: The window's rows, one Rerun ``Transform3D`` per row.
        segment_id: Which segment the rows came from, for the refusal below.

    Returns:
        The poses the window carries, on ``video_time``.

    Raises:
        ValueError: If the two components are not valid on the same rows. A pose
            is one row's translation and that same row's rotation, and the two
            components are read by dropping each column's own nulls — so two
            masks with equal counts on different rows flatten to equal lengths
            and every pose silently takes another row's rotation. Rerun stores
            components independently and lets either one be logged or cleared
            alone, so the layout is one a valid recording can hold; a refusal
            naming the row is the only reading of it that cannot be wrong
            (S25 review).
    """
    translation_column: str = f"{RIG_ENTITY}:Transform3D:translation"
    quaternion_column: str = f"{RIG_ENTITY}:Transform3D:quaternion"
    if translation_column not in table.column_names or quaternion_column not in table.column_names:
        return empty_trajectory()
    row_t_ns: Int64[ndarray, " n_rows"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))
    translations: pa.Array = table[translation_column].combine_chunks()
    quaternions: pa.Array = table[quaternion_column].combine_chunks()
    valid: Bool[ndarray, " n_rows"] = translations.is_valid().to_numpy(zero_copy_only=False)
    quaternion_valid: Bool[ndarray, " n_rows"] = quaternions.is_valid().to_numpy(zero_copy_only=False)
    if not np.array_equal(valid, quaternion_valid):
        disagreeing: Int64[ndarray, " n_disagreeing"] = np.flatnonzero(valid != quaternion_valid)
        raise ValueError(
            f"{segment_id}: {disagreeing.size} of {row_t_ns.size} rig rows carry a translation without a quaternion or "
            f"a quaternion without a translation, the first at {int(row_t_ns[disagreeing[0]])} ns; a pose needs both, "
            f"and reading each component off its own rows would give it another row's rotation"
        )
    # One mask for the timestamps and both components: each column's own null
    # removal drops exactly `valid`'s false rows, because the two masks are
    # equal by here.
    position_m: Float64[ndarray, "n_poses 3"] = _flat_float(translations).reshape(-1, 3)
    if position_m.shape[0] == 0:
        return empty_trajectory()
    quaternion_xyzw: Float64[ndarray, "n_poses 4"] = _flat_float(quaternions).reshape(-1, 4)
    return Trajectory(
        t_ns=row_t_ns[valid],
        position_m=position_m,
        quaternion_wxyz=np.column_stack([quaternion_xyzw[:, 3], quaternion_xyzw[:, 0:3]]),
    )


def select_cameras(statics: pa.Table, camera_count: int, camera_names: tuple[str, ...] | None) -> tuple[int, ...]:
    """Rig indices of the named cameras, in the caller's order.

    Names are read from each camera node's ``name`` static and compared with
    hyphens normalised to underscores **on both sides**, because the rig writes
    ``left-front`` where basalt's driver spells it ``left_front`` and a caller
    may reasonably spell it either way.

    Args:
        statics: Single-row table holding every camera node's statics.
        camera_count: Cameras the rig declares.
        camera_names: Names to feed in order, or None for every camera in rig order.

    Returns:
        One rig index per fed camera.

    Raises:
        ValueError: If a name is asked for that the recording does not carry, or
            two cameras answer to the same name.
    """
    if camera_names is None:
        return tuple(range(camera_count))
    by_name: dict[str, int] = {}
    for position in range(camera_count):
        name: str = _static_string(statics, f"{RIG_ENTITY}/cam_{position:02d}:name").replace("-", "_")
        if name in by_name:
            raise ValueError(f"cameras cam_{by_name[name]:02d} and cam_{position:02d} are both named {name!r}")
        by_name[name] = position
    wanted: tuple[str, ...] = tuple(name.replace("-", "_") for name in camera_names)
    missing: list[str] = [name for name, key in zip(camera_names, wanted, strict=True) if key not in by_name]
    if missing:
        raise ValueError(f"the recording has no camera named {missing}; it carries {sorted(by_name)}")
    return tuple(by_name[key] for key in wanted)


def _build_feed(
    sensor_dataset: DatasetEntry,
    gt_dataset: DatasetEntry | None,
    segment_id: str,
    parameters: ImuParameters,
    profile: RigProfile,
    frame_stride: int,
    window_s: float,
) -> SegmentFeed:
    """Read calibration, IMU, ground truth and frameset timing for one segment."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be at least 1; got {frame_stride}")
    rig_statics: pa.Table = sensor_dataset.filter_segments([segment_id]).filter_contents([RIG_ENTITY, IMU_ENTITY]).reader(index=None).to_arrow_table()
    reference: str = _static_string(rig_statics, f"{RIG_ENTITY}:reference")
    if reference != "imu_00":
        raise ValueError(f"{segment_id}: rig reference is {reference!r}, but the feed assumes the IMU is the rig frame")
    camera_count: int = int(_static_values(rig_statics, f"{RIG_ENTITY}:num_cameras")[0])
    rig_entities: list[str] = [f"{RIG_ENTITY}/cam_{position:02d}" for position in range(camera_count)]
    camera_statics: pa.Table = (
        sensor_dataset.filter_segments([segment_id])
        .filter_contents(rig_entities + [f"{entity}/pinhole" for entity in rig_entities])
        .reader(index=None)
        .to_arrow_table()
    )
    camera_positions: tuple[int, ...] = select_cameras(camera_statics, camera_count, profile.camera_names)
    index: _VideoIndex = _read_video_index(sensor_dataset, segment_id, camera_positions, profile.frameset_tolerance_ns)
    cameras: tuple[CameraCalib, ...] = tuple(
        camera_calib(number, read_camera_statics(camera_statics, rig_entities[position]), profile.downscale)
        for number, position in enumerate(camera_positions)
    )
    imu_T_body: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    imu_T_body[:3, :3] = _static_values(rig_statics, f"{IMU_ENTITY}:Transform3D:mat3x3").reshape(3, 3, order="F")
    imu_T_body[:3, 3] = _static_values(rig_statics, f"{IMU_ENTITY}:Transform3D:translation")
    properties: pa.Table = (
        sensor_dataset.filter_segments([segment_id]).filter_contents(["/__properties", "/__properties/**"]).reader(index=None).to_arrow_table()
    )
    capture_start_time_ns: int = _static_int(properties, "property:capture:start_time_ns")
    return SegmentFeed(
        segment_id=segment_id,
        cameras=cameras,
        imu=imu_calib(parameters, imu_T_body),
        capture_start_time_ns=capture_start_time_ns,
        export_offset_ns=0 if profile.video_time_is_absolute else capture_start_time_ns,
        frame_t_ns=index.t_ns + parameters.cam_time_offset_ns,
        camera_positions=camera_positions,
        rig_cameras=camera_count,
        profile=profile,
        frame_stride=frame_stride,
        dataset=sensor_dataset,
        gt_dataset=gt_dataset,
        index=index,
        window_ns=int(window_s * 1e9),
    )


@contextmanager
def open_segment(
    source: SegmentSource,
    parameters: ImuParameters,
    profile: RigProfile = MSD_RIG,
    frame_stride: int = 1,
    window_s: float = DEFAULT_WINDOW_S,
) -> Iterator[SegmentFeed]:
    """Open one segment for reading, from local ``.rrd`` files or from a catalog server.

    A local source is served by an in-process ``rr.server.Server`` on an ephemeral
    port that shuts down when the context exits, so no catalog server is needed.
    The base and ground-truth layers become two datasets there, because one
    in-process dataset takes a single layer per segment.

    Args:
        source: Where the segment lives.
        parameters: Frozen IMU parameters, normally from the reference manifest.
        profile: How this rig has to be read; the default is what MSD is.
        frame_stride: Yield every n-th frameset; every frame is still decoded.
        window_s: Longest time window fetched in one round trip.

    Yields:
        The open feed.

    Raises:
        ValueError: If a local base ``.rrd`` does not hold exactly one segment.
    """
    if isinstance(source, LocalSegment):
        datasets: dict[str, str | PathLike[str] | Sequence[str | PathLike[str]]] = {"base": [str(source.base_rrd)]}
        if source.gt_rrd is not None:
            datasets["gt"] = [str(source.gt_rrd)]
        with rr.server.Server(datasets=datasets) as server:
            client: CatalogClient = server.client()
            base: DatasetEntry = client.get_dataset("base")
            ground_truth: DatasetEntry | None = client.get_dataset("gt") if source.gt_rrd is not None else None
            segment_ids: list[str] = list(base.segment_ids())
            if len(segment_ids) != 1:
                raise ValueError(f"{source.base_rrd} holds {len(segment_ids)} segments; the feed reads one")
            yield _build_feed(base, ground_truth, segment_ids[0], parameters, profile, frame_stride, window_s)
    else:
        dataset: DatasetEntry = CatalogClient(source.url).get_dataset(source.dataset_name)
        layers = dataset.manifest().to_arrow_table()
        has_gt: bool = any(
            row["rerun_segment_id"] == source.segment_id and row["rerun_layer_name"] == "gt"
            for row in layers.select(["rerun_segment_id", "rerun_layer_name"]).to_pylist()
        )
        yield _build_feed(dataset, dataset if has_gt else None, source.segment_id, parameters, profile, frame_stride, window_s)


def read_rig_trajectory(rrd: Path) -> Trajectory:
    """Every rig pose on one ``.rrd`` layer, on the ``video_time`` the layer stores.

    The RoboCap ``slam`` layer is a trajectory and nothing else, on the same
    ``video_time`` as the base layer it sits beside, so it is read the way the
    ground-truth layer is. Moving it onto another clock is the caller's own step
    (:func:`slam_rs.trajectory.shift_clock`) rather than a parameter here: reading
    a layer and moving a clock are two things.

    Args:
        rrd: Layer holding the rig's ``Transform3D`` rows.

    Returns:
        The whole trajectory, oldest pose first.

    Raises:
        ValueError: If the file does not hold exactly one segment, or holds no rig poses.
    """
    with rr.server.Server(datasets={"layer": [str(rrd)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("layer")
        segment_ids: list[str] = list(dataset.segment_ids())
        if len(segment_ids) != 1:
            raise ValueError(f"{rrd} holds {len(segment_ids)} segments; a trajectory layer holds one")
        found: Trajectory = _read_ground_truth(dataset, segment_ids[0], -(2**62), 2**62)
    if len(found) == 0:
        raise ValueError(f"{rrd} carries no {RIG_ENTITY} Transform3D rows")
    return found
