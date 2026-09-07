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
* **Geometry.** ``Pinhole:image_from_camera`` is column-major, ``Pinhole:resolution``
  is ``(width, height)`` while the decoded array is ``(height, width)``, and the
  camera ``Transform3D`` is ``ChildFromParent`` — that is ``cam_T_imu``, and it is
  inverted here. msd-g2 stores its images rotated into portrait with the
  calibration rotated to match, so nothing may assume a landscape frame or a
  shared orientation across cameras.
"""

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
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
from simplecv.catalog_video_codec import CatalogCodecName, catalog_codec_name

from slam_rs.reference import ImuParameters
from slam_rs.trajectory import Trajectory

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

    camera_model: str
    """``camera_model`` string on the camera node, e.g. ``kb4`` or ``pinhole-radtan8``."""
    distortion_model: str
    """``simplecv.components.DistortionModel``, e.g. ``kannala_brandt``."""
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
    image_rotation_cw_deg: int
    """Clockwise rotation the stored images already carry; 0, 90, 180 or 270."""


@dataclass(slots=True, frozen=True)
class CameraCalib:
    """One camera as the estimator wants it: metric intrinsics and ``imu_T_cam``."""

    index: int
    """Camera index on the rig, matching the order of a frameset's images."""
    width: int
    """Decoded frame width in pixels."""
    height: int
    """Decoded frame height in pixels."""
    frequency_hz: float
    """Nominal frame rate; the per-frame timestamps are authoritative."""
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
    image_rotation_cw_deg: int
    """Clockwise rotation already baked into both the images and this calibration."""


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


@dataclass(slots=True, frozen=True)
class Frameset:
    """One synchronised multi-camera capture, decoded to grayscale."""

    t_ns: int
    """Shared capture timestamp of every image, on the ``video_time`` clock."""
    images: list[UInt8[ndarray, "h w"]]
    """One C-contiguous grayscale image per camera, in rig camera order."""
    sha256: str
    """Digest of the timestamp and every image's bytes, so two runs can prove identical pixels."""


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
    """Catalog URL, e.g. ``rerun+http://dgx-spark:9988``."""
    dataset_name: str
    """Dataset entry name on that server."""
    segment_id: str
    """Segment within the dataset."""


SegmentSource: TypeAlias = LocalSegment | CatalogSegment
"""Where a feed gets its data; the rest of the module does not care which."""


def rotate_pinhole_clockwise(
    fx: float, fy: float, cx: float, cy: float, width: int, height: int, rotation_cw_deg: int
) -> tuple[float, float, float, float]:
    """Rotate a landscape pinhole calibration into the frame the images are stored in.

    msd-g2's video is stored rotated into portrait and its catalog calibration is
    rotated to match, so this is the arithmetic that reconciles a raw-MSD
    calibration with a catalog one. Nothing in the feed needs it — the catalog
    already stores the rotated values — but any A/B against a C++ basalt run fed
    from raw MSD does, and pinning it keeps the convention from drifting.

    Args:
        fx: Focal length along x before rotation.
        fy: Focal length along y before rotation.
        cx: Principal point x before rotation.
        cy: Principal point y before rotation.
        width: Image width before rotation.
        height: Image height before rotation.
        rotation_cw_deg: Clockwise rotation applied to the image, 0, 90, 180 or 270.

    Returns:
        ``(fx, fy, cx, cy)`` in the rotated frame.

    Raises:
        ValueError: If the rotation is not a multiple of 90 degrees.
    """
    if rotation_cw_deg == 0:
        return fx, fy, cx, cy
    if rotation_cw_deg == 90:
        return fy, fx, (height - 1) - cy, cx
    if rotation_cw_deg == 180:
        return fx, fy, (width - 1) - cx, (height - 1) - cy
    if rotation_cw_deg == 270:
        return fy, fx, cy, (width - 1) - cx
    raise ValueError(f"image rotation must be 0, 90, 180 or 270 degrees clockwise; got {rotation_cw_deg}")


def camera_calib(index: int, statics: CameraStatics, frequency_hz: float) -> CameraCalib:
    """Apply the catalog-to-estimator mapping rules to one camera's statics.

    The rules, each of which has cost someone a wrong trajectory: reshape
    ``image_from_camera`` column-major, read the resolution as ``(width, height)``,
    map the distortion model string and assert the coefficient tail is zero
    rather than truncating it, and invert the ``ChildFromParent`` transform to get
    ``imu_T_cam``.

    Args:
        index: Camera index on the rig.
        statics: Raw static components of the camera node.
        frequency_hz: Nominal frame rate for this segment.

    Returns:
        The camera calibration in the estimator's conventions.

    Raises:
        ValueError: If the distortion model is unknown, the coefficient tail is
            non-zero, or the transform relation is not ``ChildFromParent``.
    """
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
        width=int(statics.resolution_wh[0]),
        height=int(statics.resolution_wh[1]),
        frequency_hz=frequency_hz,
        fx=float(k_matrix[0, 0]),
        fy=float(k_matrix[1, 1]),
        cx=float(k_matrix[0, 2]),
        cy=float(k_matrix[1, 2]),
        model=model,
        distortion=statics.distortion_coefficients[:n_coeffs].copy(),
        distortion_valid_radius=statics.distortion_valid_radius,
        imu_T_cam=imu_T_cam,
        image_rotation_cw_deg=statics.image_rotation_cw_deg,
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


def _static_values(statics: pa.Table, column: str) -> Float64[ndarray, " n"]:
    """One static list component as a flat float64 array."""
    if column not in statics.column_names:
        raise ValueError(f"static column {column} is missing")
    cell: pa.Scalar = statics[column][0]
    if not cell.is_valid:
        raise ValueError(f"static column {column} is null")
    return np.asarray(cell.values.to_pylist(), dtype=np.float64).ravel()


def _static_string(statics: pa.Table, column: str) -> str:
    """One static string component, bare or wrapped in a single-element list."""
    if column not in statics.column_names:
        raise ValueError(f"static column {column} is missing")
    cell: pa.Scalar = statics[column][0]
    if not cell.is_valid:
        raise ValueError(f"static column {column} is null")
    value: object = cell.values.to_pylist()[0] if isinstance(cell, pa.ListScalar | pa.LargeListScalar) else cell.as_py()
    if not isinstance(value, str):
        raise ValueError(f"static column {column} is not a string: {value!r}")
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
    optional_rotation: str = f"{entity}:image_rotation_cw_deg"
    return CameraStatics(
        camera_model=_static_string(statics, f"{entity}:camera_model"),
        distortion_model=_static_string(statics, f"{entity}/pinhole:simplecv.components.DistortionModel"),
        distortion_coefficients=_static_values(statics, f"{entity}/pinhole:simplecv.components.DistortionCoefficients"),
        image_from_camera=_static_values(statics, f"{entity}/pinhole:Pinhole:image_from_camera"),
        resolution_wh=_static_values(statics, f"{entity}/pinhole:Pinhole:resolution"),
        transform_mat3x3=_static_values(statics, f"{entity}:Transform3D:mat3x3"),
        transform_translation=_static_values(statics, f"{entity}:Transform3D:translation"),
        transform_relation=int(_static_values(statics, f"{entity}:Transform3D:relation")[0]),
        distortion_valid_radius=float(_static_values(statics, optional_radius)[0]) if optional_radius in statics.column_names else None,
        image_rotation_cw_deg=int(_static_values(statics, optional_rotation)[0]) if optional_rotation in statics.column_names else 0,
    )


def wrap_mp4(samples: list[bytes], keyframes: list[bool], fps: int, codec: CatalogCodecName) -> bytes:
    """Mux pre-encoded samples into an in-memory MP4 with positional pts, no re-encode.

    ``simplecv.rerun_dataloader`` has the same function, but importing it drags in
    torchcodec and torchvision, which this CPU lane deliberately does not install.

    Args:
        samples: Encoded video samples in decode order; the first must be a keyframe.
        keyframes: Keyframe flag per sample.
        fps: Frame rate written into the muxed track and its time base.
        codec: Codec of the pre-encoded samples.

    Returns:
        The complete MP4 file as bytes.
    """
    buffer: BytesIO = BytesIO()
    # Pin the track timescale to fps: the muxer otherwise picks 15360 without
    # rescaling our positional pts, and the track then claims a ~0.1 s duration.
    with av.open(buffer, "w", format="mp4", options={"video_track_timescale": str(fps)}) as container:
        stream = container.add_mux_stream(codec, rate=fps, width=16, height=16)
        stream.time_base = Fraction(1, fps)
        for sample_index, (sample, is_keyframe) in enumerate(zip(samples, keyframes, strict=True)):
            packet: av.Packet = av.Packet(sample)
            packet.pts = packet.dts = sample_index
            packet.duration = 1
            packet.time_base = stream.time_base
            packet.stream = stream
            packet.is_keyframe = is_keyframe
            container.mux(packet)
    return buffer.getvalue()


def decode_gray(mp4_bytes: bytes) -> Iterator[UInt8[ndarray, "h w"]]:
    """Decode an in-memory MP4 to C-contiguous ``gray8`` frames, one decoder thread.

    ``reformat(format="gray8")`` performs the limited-to-full range expansion the
    MSD streams need; reading the raw Y plane instead is a systematic photometric
    shift of up to 17 LSB. dav1d pads each row, so the plane's ``line_size``
    exceeds the frame width and the visible columns are sliced out.

    Args:
        mp4_bytes: MP4 produced by :func:`wrap_mp4`.

    Yields:
        One grayscale image per frame, in decode order.
    """
    container: av.container.InputContainer = av.open(BytesIO(mp4_bytes), mode="r")
    with container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "NONE"
        for frame in container.decode(stream):
            gray = frame.reformat(format="gray8")
            plane = gray.planes[0]
            padded: UInt8[ndarray, "h stride"] = np.frombuffer(bytes(plane), dtype=np.uint8).reshape(gray.height, plane.line_size)
            yield np.ascontiguousarray(padded[:, : gray.width])


@dataclass(slots=True, frozen=True)
class _VideoIndex:
    """Per-camera frame timing, read without touching a single encoded sample."""

    t_ns: Int64[ndarray, " n_frames"]
    """Frame timestamps on ``video_time``, shared by every camera on a hardware-synced rig."""
    keyframe: Bool[ndarray, " n_frames"]
    """True where the frame is a keyframe in every camera, so a window may start there."""
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
    imu_stream: ImuStream
    """Every inertial sample of the segment, on the ``video_time`` clock."""
    ground_truth: Trajectory | None
    """Ground-truth rig poses on ``video_time``, or None when the source has no ``gt`` layer."""
    frame_t_ns: Int64[ndarray, " n_frames"]
    """Timestamp of every frameset in the segment, before ``frame_stride`` is applied."""
    frame_stride: int
    """Yield every n-th frameset. Every frame is still decoded: decimated AV1 decode is unreliable."""
    dataset: DatasetEntry
    """Dataset the video samples are fetched from."""
    index: _VideoIndex
    """Frame timing and codec, shared by all cameras."""
    window_ns: int
    """Longest time window fetched in one round trip."""

    def framesets(self) -> Iterator[Frameset]:
        """Decode the segment and yield one frameset at a time.

        Window edges land on frames that are keyframes in every camera, so each
        window decodes standalone; inside a window the cameras are decoded in
        lockstep, so only one frame per camera is ever resident.

        Yields:
            Framesets in time order, every :attr:`frame_stride`-th one.

        Raises:
            ValueError: If a camera's decoder runs dry before the window ends, or
                a decoded frame disagrees with the calibrated resolution.
        """
        for start, stop in _window_bounds(self.index, self.window_ns):
            muxed: list[bytes] = [
                wrap_mp4(*self._fetch_samples(camera.index, start, stop), fps=self.index.fps, codec=self.index.codec) for camera in self.cameras
            ]
            decoders: list[Iterator[UInt8[ndarray, "h w"]]] = [decode_gray(window) for window in muxed]
            for frame_index in range(start, stop):
                images: list[UInt8[ndarray, "h w"]] = []
                for camera, decoder in zip(self.cameras, decoders, strict=True):
                    image: UInt8[ndarray, "h w"] | None = next(decoder, None)
                    if image is None:
                        raise ValueError(f"{self.segment_id}: cam_{camera.index:02d} ran out of frames at index {frame_index}")
                    if image.shape != (camera.height, camera.width):
                        raise ValueError(
                            f"{self.segment_id}: cam_{camera.index:02d} decoded {image.shape}, calibration says {(camera.height, camera.width)}"
                        )
                    images.append(image)
                if frame_index % self.frame_stride:
                    continue
                t_ns: int = int(self.frame_t_ns[frame_index])
                digest = hashlib.sha256(np.int64(t_ns).tobytes())
                for image in images:
                    digest.update(image.tobytes())
                yield Frameset(t_ns=t_ns, images=images, sha256=digest.hexdigest())

    def _fetch_samples(self, camera_index: int, start: int, stop: int) -> tuple[list[bytes], list[bool]]:
        """Encoded samples and keyframe flags of one camera over the frame range ``[start, stop)``."""
        entity: str = f"{RIG_ENTITY}/cam_{camera_index:02d}/pinhole/video"
        first_ns: int = int(self.index.t_ns[start])
        last_ns: int = int(self.index.t_ns[stop - 1])
        table: pa.Table = (
            self.dataset.filter_segments([self.segment_id])
            .filter_contents(entity)
            .reader(index=TIMELINE)
            .select(TIMELINE, f"{entity}:VideoStream:sample", f"{entity}:VideoStream:is_keyframe")
            .filter(col(TIMELINE).cast(pa.int64()).between(lit(first_ns), lit(last_ns)))
            .to_arrow_table()
        )
        times: Int64[ndarray, " n_window"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))
        if not np.array_equal(times, self.index.t_ns[start:stop]):
            raise ValueError(f"{self.segment_id}: cam_{camera_index:02d} returned {len(times)} samples for frames [{start}, {stop})")
        # Arrow has no list<u8> -> binary cast, so slice the child buffer by the list
        # offsets. large_list keeps 64-bit offsets: one camera of a multi-hour session
        # exceeds the default int32's 2 GiB.
        blobs: pa.LargeListArray = table[1].combine_chunks().cast(pa.list_(pa.large_list(pa.uint8()))).flatten()
        data: UInt8[ndarray, " n_bytes"] = blobs.values.to_numpy(zero_copy_only=True)
        offsets: Int64[ndarray, " n_offsets"] = blobs.offsets.to_numpy(zero_copy_only=True)
        samples: list[bytes] = [data[begin:end].tobytes() for begin, end in zip(offsets[:-1], offsets[1:], strict=True)]
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


def _read_video_index(dataset: DatasetEntry, segment_id: str, camera_count: int) -> _VideoIndex:
    """Frame timestamps, shared keyframes and codec, fetched without any sample bytes."""
    per_camera_times: list[Int64[ndarray, " n_frames"]] = []
    shared_keyframe: Bool[ndarray, " n_frames"] | None = None
    codec_fourcc: int | None = None
    for camera_index in range(camera_count):
        entity: str = f"{RIG_ENTITY}/cam_{camera_index:02d}/pinhole/video"
        table: pa.Table = (
            dataset.filter_segments([segment_id])
            .filter_contents(entity)
            .reader(index=TIMELINE)
            .select(TIMELINE, f"{entity}:VideoStream:is_keyframe", f"{entity}:VideoStream:codec")
            .to_arrow_table()
        )
        per_camera_times.append(np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64())))
        keyframe: Bool[ndarray, " n_frames"] = np.asarray(table[1].combine_chunks().is_valid().to_numpy(zero_copy_only=False), dtype=bool)
        shared_keyframe = keyframe if shared_keyframe is None else (shared_keyframe & keyframe)
        codec_fourcc = int(table[2].combine_chunks().drop_null().flatten()[0].as_py())
    if shared_keyframe is None or codec_fourcc is None:
        raise ValueError(f"{segment_id}: the rig reports {camera_count} cameras, so no video columns were read")
    for camera_index, times in enumerate(per_camera_times[1:], start=1):
        # MSD is hardware-synced: a frameset is "all cameras at the same video_time".
        # Keep the assertion rather than a nearest-match synchroniser.
        if not np.array_equal(times, per_camera_times[0]):
            raise ValueError(f"{segment_id}: cam_{camera_index:02d} timestamps differ from cam_00; this rig needs a frameset matcher")
    frame_t_ns: Int64[ndarray, " n_frames"] = per_camera_times[0]
    if len(frame_t_ns) < 2:
        raise ValueError(f"{segment_id}: {len(frame_t_ns)} frames is not a segment")
    fps: int = max(1, round((len(frame_t_ns) - 1) * 1e9 / float(frame_t_ns[-1] - frame_t_ns[0])))
    return _VideoIndex(t_ns=frame_t_ns, keyframe=shared_keyframe, codec=catalog_codec_name(codec_fourcc), fps=fps)


def _read_imu(dataset: DatasetEntry, segment_id: str, cam_time_offset_ns: int) -> ImuStream:
    """Gyroscope and accelerometer of one segment, asserted paired on one clock."""
    table: pa.Table = (
        dataset.filter_segments([segment_id]).filter_contents([f"{IMU_ENTITY}/gyro", f"{IMU_ENTITY}/accel"]).reader(index=TIMELINE).to_arrow_table()
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
    # MSD logs both sensors on identical timestamps. A rig that does not (RoboCap)
    # needs the accelerometer interpolated onto the gyro clock before it gets here,
    # so the core only ever sees the paired form.
    if not np.array_equal(gyro_t_ns, accel_t_ns):
        raise ValueError(
            f"{segment_id}: {len(gyro_t_ns)} gyro and {len(accel_t_ns)} accel samples are not on identical timestamps; pair them before feeding"
        )
    if gyro_t_ns.size and not bool(np.all(np.diff(gyro_t_ns) > 0)):
        raise ValueError(f"{segment_id}: IMU timestamps are not strictly increasing")
    return ImuStream(t_ns=gyro_t_ns + cam_time_offset_ns, gyro_rad_s=streams["gyro"][1], accel_m_s2=streams["accel"][1])


def _read_ground_truth(dataset: DatasetEntry, segment_id: str) -> Trajectory | None:
    """Ground-truth rig poses on ``video_time``, converted from Rerun's XYZW to w-first."""
    table: pa.Table = dataset.filter_segments([segment_id]).filter_contents([RIG_ENTITY]).reader(index=TIMELINE).to_arrow_table()
    translation_column: str = f"{RIG_ENTITY}:Transform3D:translation"
    quaternion_column: str = f"{RIG_ENTITY}:Transform3D:quaternion"
    if translation_column not in table.column_names or quaternion_column not in table.column_names:
        return None
    row_t_ns: Int64[ndarray, " n_rows"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))
    translations: pa.Array = table[translation_column].combine_chunks()
    valid: Bool[ndarray, " n_rows"] = translations.is_valid().to_numpy(zero_copy_only=False)
    position_m: Float64[ndarray, "n_poses 3"] = _flat_float(translations).reshape(-1, 3)
    if position_m.shape[0] == 0:
        return None
    quaternion_xyzw: Float64[ndarray, "n_poses 4"] = _flat_float(table[quaternion_column].combine_chunks()).reshape(-1, 4)
    return Trajectory(
        t_ns=row_t_ns[valid],
        position_m=position_m,
        quaternion_wxyz=np.column_stack([quaternion_xyzw[:, 3], quaternion_xyzw[:, 0:3]]),
    )


def _build_feed(
    sensor_dataset: DatasetEntry,
    gt_dataset: DatasetEntry | None,
    segment_id: str,
    parameters: ImuParameters,
    frame_stride: int,
    window_s: float,
) -> SegmentFeed:
    """Read calibration, IMU, ground truth and frame timing for one segment."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be at least 1; got {frame_stride}")
    rig_statics: pa.Table = sensor_dataset.filter_segments([segment_id]).filter_contents([RIG_ENTITY, IMU_ENTITY]).reader(index=None).to_arrow_table()
    reference: str = _static_string(rig_statics, f"{RIG_ENTITY}:reference")
    if reference != "imu_00":
        raise ValueError(f"{segment_id}: rig reference is {reference!r}, but the feed assumes the IMU is the rig frame")
    camera_count: int = int(_static_values(rig_statics, f"{RIG_ENTITY}:num_cameras")[0])
    camera_entities: list[str] = [f"{RIG_ENTITY}/cam_{position:02d}" for position in range(camera_count)]
    camera_statics: pa.Table = (
        sensor_dataset.filter_segments([segment_id])
        .filter_contents(camera_entities + [f"{entity}/pinhole" for entity in camera_entities])
        .reader(index=None)
        .to_arrow_table()
    )
    index: _VideoIndex = _read_video_index(sensor_dataset, segment_id, camera_count)
    cameras: tuple[CameraCalib, ...] = tuple(
        camera_calib(position, read_camera_statics(camera_statics, entity), float(index.fps)) for position, entity in enumerate(camera_entities)
    )
    imu_T_body: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    imu_T_body[:3, :3] = _static_values(rig_statics, f"{IMU_ENTITY}:Transform3D:mat3x3").reshape(3, 3, order="F")
    imu_T_body[:3, 3] = _static_values(rig_statics, f"{IMU_ENTITY}:Transform3D:translation")
    return SegmentFeed(
        segment_id=segment_id,
        cameras=cameras,
        imu=imu_calib(parameters, imu_T_body),
        imu_stream=_read_imu(sensor_dataset, segment_id, parameters.cam_time_offset_ns),
        ground_truth=_read_ground_truth(gt_dataset, segment_id) if gt_dataset is not None else None,
        frame_t_ns=index.t_ns,
        frame_stride=frame_stride,
        dataset=sensor_dataset,
        index=index,
        window_ns=int(window_s * 1e9),
    )


@contextmanager
def open_segment(
    source: SegmentSource,
    parameters: ImuParameters,
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
            yield _build_feed(base, ground_truth, segment_ids[0], parameters, frame_stride, window_s)
    else:
        dataset: DatasetEntry = CatalogClient(source.url).get_dataset(source.dataset_name)
        yield _build_feed(dataset, dataset, source.segment_id, parameters, frame_stride, window_s)
