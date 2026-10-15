"""One ``dataforge:v1`` segment as calibration, IMU, ground truth and grayscale framesets.

This Python half of the estimator contract reads a catalog URL or local ``.rrd`` served in process and gives the Rust core
CPU grayscale images with integer-nanosecond timestamps. Five decisions affect the numbers:

1. **Pixels.** CPU muxes without re-encoding and decodes AV1 with single-threaded dav1d to ``gray8``. MSD's limited-range
   ``yuv420p`` has flat chroma: limited-to-full expansion restores original grayscale; raw Y differs by up to 17 LSB. Copies honor
   dav1d's padded ``line_size`` (> width). Optional CUDA uses SimpleCV NVDEC and GPU RGB-to-gray with different rounding.
2. **Round trips.** Each camera, IMU and truth is read per window. Long segments split on ``video_time`` at keyframes shared
   by every camera, so windows decode alone. Upstream decimated AV1 decode is unsafe: decode all frames; ``frame_stride``
   selects yielded framesets. Offline ``cache_video`` fetches selected compressed streams once, reuses packets for the frame
   index and keeps decoders alive for the segment.
3. **Rig shape.** Rigs need not have four hardware-synced cameras at stored resolution on one IMU clock. :class:`RigProfile`
   defaults to MSD and sets camera selection/order, integer downscale for frames and intrinsics, accel interpolation onto
   the gyro clock, and the allowed time gap between cameras in a frameset.
4. **Clocks.** DataForge frames, IMU and truth share ``video_time``. Add ``cam_time_offset_ns`` (from the catalog's signed
   ``applied_time_shift_ns``) to reach the estimator's inertial clock; invert for query bounds. Never reapply factory
   per-camera offsets. MSD's zero offset leaves all its numbers unchanged.
5. **Geometry.** ``Pinhole:image_from_camera`` is column-major; ``Pinhole:resolution`` is ``(width, height)``, decoded arrays
   are ``(height, width)``. Camera ``Transform3D`` is ``ChildFromParent`` (``cam_T_imu``), inverted here. msd-g2 images and
   calibration are rotated to portrait: assume neither landscape frames nor a shared camera orientation.
"""

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import ParseResult, unquote, urlparse

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from datafusion import col, lit
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_video import CatalogVideo, catalog_codec, catalog_keyframes, packet_views, read_catalog_videos
from simplecv.catalog_video_codec import CatalogCodecName, wrap_mp4
from simplecv.imu_calibration import ImuCalibration

from slam_rs import _core
from slam_rs.catalog_calibration import (
    CHILD_FROM_PARENT as CHILD_FROM_PARENT,
)
from slam_rs.catalog_calibration import (
    CameraCalib as CameraCalib,
)
from slam_rs.catalog_calibration import (
    CameraModelName as CameraModelName,
)
from slam_rs.catalog_calibration import (
    CameraStatics as CameraStatics,
)
from slam_rs.catalog_calibration import (
    ImuCalib as ImuCalib,
)
from slam_rs.catalog_calibration import (
    camera_calib as camera_calib,
)
from slam_rs.catalog_calibration import (
    imu_calib as imu_calib,
)
from slam_rs.catalog_calibration import (
    scale_principal_point as scale_principal_point,
)
from slam_rs.catalog_timing import (
    ImuStream as ImuStream,
)
from slam_rs.catalog_timing import _frame_nearest_anchor as _frame_nearest_anchor
from slam_rs.catalog_timing import (
    match_framesets as match_framesets,
)
from slam_rs.catalog_timing import (
    pair_accel_onto_gyro as pair_accel_onto_gyro,
)
from slam_rs.rig import IMU_ENTITY as IMU_ENTITY
from slam_rs.rig import RIG_ENTITY as RIG_ENTITY
from slam_rs.rig import TIMELINE as TIMELINE
from slam_rs.trajectory import ASSOCIATION_TOLERANCE_NS, Trajectory, empty_trajectory, shift_clock

DEFAULT_WINDOW_S: float = 60.0
"""Time window a long segment is cut into: a 7.6 s two-camera segment is 8.5 MB, so a 2,000 s one is not one query."""

@dataclass(slots=True, frozen=True)
class Frameset:
    """One synchronised multi-camera capture, decoded to grayscale."""

    t_ns: int
    """Shared capture timestamp of every image, on the estimator's inertial clock."""
    images: list[UInt8[ndarray, "h w"]]
    """One C-contiguous grayscale image per camera, in rig camera order."""
    imu: ImuStream
    """Inertial samples since the previous frameset, including one past :attr:`t_ns`.
    Without that lead, a backend blocking until it can integrate to frame time deadlocks on the first frameset.
    """
    ground_truth: Float64[ndarray, " 7"] | None
    """Nearest ground-truth pose as ``[tx, ty, tz, qw, qx, qy, qz]``, or None without a ``gt`` layer."""

    def image_digests(self) -> tuple[str, ...]:
        """Return a hex digest per contiguous gray8 camera buffer, in image order, for pixel comparisons and clip dumps.
        Hash on demand to avoid feed-loop cost and a temporary bytes copy.
        """
        return tuple(hashlib.sha256(image).hexdigest() for image in self.images)

    def digest(self) -> str:
        """Return one frameset hex digest combining its timestamp and per-camera digests."""
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
    dataset: DatasetEntry | None = None
    """Resolved dataset handle; absent until availability has been checked."""
    ground_truth_uri: str | None = None
    """Registered ground-truth RRD, resolved from the catalog and readable by this worker."""

    @property
    def has_ground_truth(self) -> bool:
        """Whether the resolved segment has a ground-truth layer."""
        return self.ground_truth_uri is not None


def resolve_catalog_segments(sources: Sequence[CatalogSegment], require_ground_truth: bool = False) -> tuple[CatalogSegment, ...]:
    """Resolve availability with one manifest query per catalog dataset; ``open_segment`` reuses handles without lookup.
    The caller's scoring policy sets ground-truth requirements.
    """
    datasets: dict[tuple[str, str], DatasetEntry] = {}
    layers: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    resolved: list[CatalogSegment] = []
    for source in sources:
        key: tuple[str, str] = (source.url, source.dataset_name)
        if key not in datasets:
            dataset: DatasetEntry = CatalogClient(source.url).get_dataset(source.dataset_name)
            datasets[key] = dataset
            layers[key] = {}
            # One row per segment; its layer names and storage URLs are parallel lists.
            table: pa.Table = dataset.segment_table().select("rerun_segment_id", "rerun_layer_names", "rerun_storage_urls").to_arrow_table()
            for row in table.to_pylist():
                layers[key][row["rerun_segment_id"]] = dict(zip(row["rerun_layer_names"], row["rerun_storage_urls"], strict=True))
        if source.segment_id not in layers[key]:
            raise ValueError(f"{source.segment_id}: absent from catalog")
        has_gt: bool = "gt" in layers[key][source.segment_id]
        if require_ground_truth and not has_gt:
            raise ValueError(f"{source.segment_id}: ground-truth layer absent")
        resolved.append(replace(source, dataset=datasets[key], ground_truth_uri=layers[key][source.segment_id].get("gt")))
    return tuple(resolved)


SegmentSource: TypeAlias = LocalSegment | CatalogSegment
"""Where a feed gets its data; the rest of the module does not care which."""


@dataclass(slots=True, frozen=True)
class RigProfile:
    """Rig read rules where reading the whole recording is wrong; callers state deviations rather than the feed guessing.
    Defaults are MSD: all declared cameras, stored resolution, one hardware-synced clock, paired inertial channels.
    RoboCap differs on all four.
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
    """True if ``video_time`` is already the device clock used by trajectory CSV exports, so exports add nothing (RoboCap).
    False for MSD: ``video_time`` is relative to ``property:capture:start_time_ns``.
    """

    def __post_init__(self) -> None:
        """Before reading, raise ValueError for unsupported ``downscale < 1`` or ``frameset_tolerance_ns < 0``."""
        if self.downscale < 1:
            raise ValueError(f"downscale must be at least 1; got {self.downscale}")
        if self.frameset_tolerance_ns < 0:
            raise ValueError(f"frameset_tolerance_ns cannot be negative; got {self.frameset_tolerance_ns}")

MSD_RIG: RigProfile = RigProfile()
"""The Monado SLAM Dataset rigs: every camera, native resolution, one clock, paired inertial channels."""

ROBOCAP_RIG: RigProfile = _core.catalog_rig_profile("robocap")
"""RoboCap selection, downscale and clock rules from the Rust core."""


def _flat_float(column: pa.Array) -> Float64[ndarray, " n_values"]:
    """Flatten non-null temporal component values to float64. Rerun nests values one list deep, plus the component's arity:
    ``Transform3D:translation`` is ``list<fixed_size_list<3>>``, ``Scalars:scalars`` is ``list<double>``; unwrap in a loop.
    """
    values: pa.Array = column.drop_null()
    while pa.types.is_list(values.type) or pa.types.is_large_list(values.type) or pa.types.is_fixed_size_list(values.type):
        values = values.flatten()
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)


def _static_cell(statics: pa.Table, column: str) -> pa.Scalar:
    """Return the valid row-zero cell for ``column`` (e.g. ``/world/rig_00:reference``), since statics are logged once.
    ``statics`` is a single-row ``filter_contents(...).reader(index=None)`` table. Raise ValueError for a missing column,
    no rows or a null cell: empty rig statics formerly caused a pyarrow IndexError that did not name the component.
    The three readers below differ only in how they interpret the cell.
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
    """Read a static integer without float64 to keep timestamp sums exact: today's device clock (~1e13 ns) fits float64,
    but has only three decades of margin to 2^53.
    """
    value: object = _static_scalar(_static_cell(statics, column))
    if not isinstance(value, int):
        raise ValueError(f"static column {column} is not an integer: {value!r}")
    return value


def read_camera_statics(statics: pa.Table, entity: str) -> CameraStatics:
    """Return raw, unconverted camera statics for ``entity`` (e.g. ``/world/rig_00/cam_00``).
    ``statics`` is a single-row table from ``filter_contents(...).reader(index=None)``.
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
    """Yield C-contiguous ``gray8`` frames in decode order from in-memory ``mp4_bytes`` produced by :func:`wrap_mp4`.
    Use one decoder thread. ``reformat(format="gray8")`` expands MSD's limited range to full; raw Y shifts pixels by up to
    17 LSB. Slice visible columns from dav1d's padded rows (``line_size`` > width). Read via the buffer protocol and copy
    once (T06: 0.677 ms/frame vs 0.751 ms with the old padded-plane ``bytes(plane)`` copy). Copy explicitly:
    :func:`numpy.ascontiguousarray` would leave unpadded frames as views into memory the decoder reuses for the next frame.
    Integer ``downscale > 1`` shrinks frames in the same ``swscale`` call with ``SWS_AREA``, combining color conversion
    and area resampling; converting then resizing uses a second, different filter and changes the trajectory.
    Raise ValueError if ``downscale < 1``.
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
    """Frameset timing without reading encoded samples: each row holds ``video_time`` and each fed camera's frame.
    For a whole hardware-synced rig, frameset ``i`` uses every camera's frame ``i``; RoboCap uses this feed's nearest matches.
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
    """Add to exported ``video_time`` stamps for the absolute device clock: :attr:`capture_start_time_ns` for relative rigs,
    zero if :attr:`RigProfile.video_time_is_absolute`. This recording-owned rule keeps tools reading the feed consistent.
    """
    frame_t_ns: Int64[ndarray, " n_frames"]
    """All frameset times before ``frame_stride``: ``video_time`` + :attr:`ImuCalib.cam_time_offset_ns`, on the inertial clock
    used by the estimator, exported trajectory and ground-truth CSV exports.
    """
    camera_positions: tuple[int, ...]
    """Rig index of each fed camera, in the order a frameset's images arrive."""
    rig_cameras: int
    """Cameras the rig declares, of which :attr:`camera_positions` are the fed ones."""
    profile: RigProfile
    """The caller's exact rig read rules, held whole so each setting has one declaration.
    Copying two fields into the feed previously made three profile declarations, only one of them the caller's.
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
    cached_payloads: tuple[bytes, ...] | None = None
    """Muxed streams for an offline run; packet buffers are released after indexing and muxing."""
    decode_device: Literal["cpu", "cuda"] = "cpu"
    """Pixel decoder; CUDA uses the shared TorchCodec reader with GPU resize."""

    @property
    def has_ground_truth(self) -> bool:
        """Whether a ``gt`` layer is attached."""
        return self.gt_dataset is not None

    def stop_ns_after(self, max_framesets: int | None) -> int | None:
        """Return the last yielded timestamp for ``max_framesets`` consumed framesets; None means no count limit.
        :meth:`framesets` reads by time: this bound avoids fetching an unused window en route to segment end
        (30 s of four 1080p H.264 streams on RoboCap).
        """
        if max_framesets is None:
            return None
        last_index: int = min(max(max_framesets - 1, 0) * self.frame_stride, len(self.frame_t_ns) - 1)
        return int(self.frame_t_ns[last_index])

    def imu_between(self, first_ns: int, last_ns: int) -> ImuStream:
        """Every inertial sample with ``first_ns <= t <= last_ns``, on the inertial clock."""
        offset_ns: int = self.imu.cam_time_offset_ns
        samples: ImuStream = _read_imu(
            self.dataset, self.segment_id, self.profile.interpolate_accel_onto_gyro, first_ns - offset_ns, last_ns - offset_ns,
        )
        return replace(samples, t_ns=samples.t_ns + offset_ns)

    def ground_truth_between(self, first_ns: int, last_ns: int) -> Trajectory:
        """Return rig truth over inertial ``[first_ns, last_ns]``; query its stored ``video_time`` clock (also used by frames)
        and shift back. Empty for no ``gt`` layer or no window coverage: test :func:`len` for poses, :attr:`has_ground_truth`
        for layer presence. Raise ValueError if translation/rotation validity differs by row, preventing pose reads
        (:func:`rig_trajectory`).
        """
        if self.gt_dataset is None:
            return empty_trajectory()
        offset_ns: int = self.imu.cam_time_offset_ns
        return shift_clock(read_ground_truth(self.gt_dataset, self.segment_id, first_ns - offset_ns, last_ns - offset_ns), offset_ns)

    def framesets(self, stop_ns: int | None = None) -> Iterator[Frameset]:
        """Decode the segment; yield every :attr:`frame_stride`-th frameset, one at a time in time order.
        Each carries IMU samples since the previous frameset through one past its timestamp, plus the nearest truth pose.
        Default video/IMU/truth reads use windows to bound memory/startup cost; edges are keyframes in every camera so each
        window decodes alone. Cached video uses one whole-segment window, including IMU/truth. CPU holds one image per camera;
        CUDA holds a batch. Fetching runs a window ahead: state early stops to avoid an unread encoded window
        (RoboCap: 30 s x 30 fps x 4 cameras at 1080p).
        ``stop_ns`` is the last useful inertial-clock time; skip windows opening later. None reads to segment end.
        The covering window still yields through its end because reads are per window.
        Raise ValueError for a decoder ending before the window, a frame/calibration resolution mismatch or an IMU boundary gap.
        """
        # Read beyond both ends: one sample after frame time, a whole frame period before for samples since the prior window's
        # last frameset. A few IMU periods would drop 16 ms at every 54 Hz boundary; the gap check below catches this.
        frame_period_ns: int = int(1e9 / max(self.index.fps, 1))
        imu_period_ns: int = int(1e9 / max(self.imu.frequency_hz, 1.0))
        margin_ns: int = max(2 * frame_period_ns + 2 * imu_period_ns, 2_000_000)
        emitted_imu_t_ns: int = -(2**62)
        bounds: list[tuple[int, int]] = [(0, len(self.frame_t_ns))] if self.cached_payloads is not None else _window_bounds(self.index, self.window_ns)
        for start, stop in bounds:
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
            cursor: list[int] = []
            for position, camera in enumerate(self.cameras):
                window: tuple[bytes, int] = self._window_payload(position, start, stop)
                decoders.append(self._open_decoder(window[0], camera))
                cursor.append(window[1] - 1)
            # Keep one selected frame and decode cursor per camera; decode even unused frames in order or the next breaks.
            decoded: list[UInt8[ndarray, "h w"] | None] = [None] * len(self.cameras)
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

    def _open_decoder(self, payload: bytes, camera: CameraCalib) -> Iterator[UInt8[ndarray, "h w"]]:
        """Keep the reference pixel conversion distinct from NVDEC."""
        if self.decode_device == "cuda":
            from slam_rs.cuda_decode import decode_gray_cuda

            return decode_gray_cuda(payload, (camera.height, camera.width), self.profile.downscale)
        return decode_gray(payload, self.profile.downscale)

    def _window_payload(self, position: int, start: int, stop: int) -> tuple[bytes, int]:
        """Muxed window and its first source-frame index, before frameset matching."""
        if self.cached_payloads is not None:
            return self.cached_payloads[position], 0
        packets: tuple[list[UInt8[ndarray, " n_sample_bytes"]], list[bool]] = self._fetch_samples(position, start, stop)
        return wrap_mp4(packets[0], packets[1], fps=self.index.fps, codec=self.index.codec), int(self.index.frame_index[start, position])

    def _fetch_samples(self, position: int, start: int, stop: int) -> tuple[list[UInt8[ndarray, " n_sample_bytes"]], list[bool]]:
        """Return one fed camera's encoded samples/keyframe flags for framesets ``[start, stop)``.
        Include every camera frame from the first frameset's choice to the last's, so decoding has all dependencies.
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
        return packet_views(table[1]), catalog_keyframes(table[2])


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


def _video_codec(table: pa.Table, entity: str) -> CatalogCodecName:
    """Return the sample wrapper's codec name for every sample in one camera stream.
    ``table`` has ``_read_video_index``'s timeline, ``is_keyframe``, ``codec`` columns; ``entity`` names the video path in errors.
    Static ``VideoStream:codec`` is logged once: take row zero of non-null codecs. Raise ValueError if samples or codec are
    absent, identifying the camera rather than exposing a pyarrow index error.
    """
    if table.num_rows == 0:
        raise ValueError(f"{entity}: the recording carries no video samples")
    return catalog_codec(table[2], entity)


def _shared_codec(per_camera: Sequence[tuple[int, CatalogCodecName]], segment_id: str) -> CatalogCodecName:
    """Return the sample wrapper's shared codec from at least one ``per_camera`` (camera_index, codec) pair in feed order.
    ``segment_id`` identifies the cameras' segment in errors. :attr:`_VideoIndex.codec` is what
    :meth:`SegmentFeed._fetch_samples` muxes every camera under: differing codecs cannot decode from one index.
    Raise ValueError naming the two mismatched cameras, rather than silently decoding three of four streams as the fourth.
    """
    first_camera, codec = per_camera[0]
    for camera_index, other in per_camera[1:]:
        if other != codec:
            raise ValueError(
                f"{segment_id}: cam_{camera_index:02d} is {other} where cam_{first_camera:02d} is {codec}; "
                f"one video index carries one codec for every camera"
            )
    return codec


def _read_video_index(
    dataset: DatasetEntry, segment_id: str, camera_positions: Sequence[int], tolerance_ns: int,
    cached_videos: tuple[CatalogVideo, ...] | None = None,
) -> _VideoIndex:
    """Frameset timing, per-camera frames, shared keyframes and codec, fetched without any sample bytes."""
    per_camera_times: list[Int64[ndarray, " n_frames"]] = []
    per_camera_keyframe: list[Bool[ndarray, " n_frames"]] = []
    per_camera_codec: list[tuple[int, CatalogCodecName]] = []
    for position, camera_index in enumerate(camera_positions):
        if cached_videos is not None:
            video: CatalogVideo = cached_videos[position]
            per_camera_times.append(video.t_ns)
            per_camera_keyframe.append(np.asarray(video.keyframes, dtype=bool))
            per_camera_codec.append((camera_index, video.codec))
            continue
        entity: str = f"{RIG_ENTITY}/cam_{camera_index:02d}/pinhole/video"
        table: pa.Table = (
            dataset.filter_segments([segment_id])
            .filter_contents(entity)
            .reader(index=TIMELINE)
            .select(TIMELINE, f"{entity}:VideoStream:is_keyframe", f"{entity}:VideoStream:codec")
            .to_arrow_table()
        )
        per_camera_times.append(np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64())))
        per_camera_keyframe.append(np.asarray(catalog_keyframes(table[1]), dtype=bool))
        per_camera_codec.append((camera_index, _video_codec(table, entity)))
    if not per_camera_codec:
        raise ValueError(f"{segment_id}: no camera was selected, so no video columns were read")
    codec: CatalogCodecName = _shared_codec(per_camera_codec, segment_id)
    if tolerance_ns == 0:
    # MSD hardware sync gives all cameras the same video_time per frameset; matching would return the identity table.
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
    """Return the nearest pose to ``t_ns`` as ``[tx, ty, tz, qw, qx, qy, qz]``, or None outside the gate's association tolerance.
    This prevents stale truth outside its span: Index smoke truth starts 17.5 ms after ``video_time`` zero, so its first frameset has none.
    """
    if len(trajectory) == 0:
        return None
    nearest: int = int(np.abs(trajectory.t_ns - t_ns).argmin())
    if abs(int(trajectory.t_ns[nearest]) - t_ns) > tolerance_ns:
        return None
    return np.concatenate([trajectory.position_m[nearest], trajectory.quaternion_wxyz[nearest]])


def _read_imu(dataset: DatasetEntry, segment_id: str, interpolate_accel: bool, first_ns: int, last_ns: int) -> ImuStream:
    """Gyroscope and accelerometer over one window on the catalog's ``video_time`` clock."""
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
    try:
        return _core.catalog_pair_imu(gyro_t_ns, streams["gyro"][1], accel_t_ns, streams["accel"][1], interpolate_accel)
    except ValueError as error:
        raise ValueError(f"{segment_id}: {error}") from error


def read_ground_truth(dataset: DatasetEntry, segment_id: str, first_ns: int, last_ns: int) -> Trajectory:
    """Return one window's rig truth on ``video_time``, converting Rerun XYZW to w-first. Empty for no rig transform or no
    poses in this window; both mean no truth here. :attr:`SegmentFeed.has_ground_truth` distinguishes layer absence.
    """
    table: pa.Table = (
        dataset.filter_segments([segment_id])
        .filter_contents([RIG_ENTITY])
        .reader(index=TIMELINE)
        .filter(col(TIMELINE).cast(pa.int64()).between(lit(first_ns), lit(last_ns)))
        .to_arrow_table()
    )
    return rig_trajectory(table, segment_id)


def rig_trajectory(table: pa.Table, segment_id: str, entity: str = RIG_ENTITY, timeline: str = TIMELINE) -> Trajectory:
    """Return a ``gt`` window's rig trajectory on ``video_time``, converting Rerun XYZW to w-first.
    ``table`` holds one ``entity`` Transform3D per row, indexed by ``timeline``; ``segment_id`` identifies errors.
    Empty if neither component exists or the window has no pose. Raise ValueError naming the row if translation and rotation
    validity differs: a pose needs both from the same row. Dropping column nulls separately can give equal lengths but pair
    different rows. Rerun stores components independently and permits logging/clearing either alone, so valid recordings can
    have this layout; refusal is the only safe reading (S25 review).
    """
    translation_column: str = f"{entity}:Transform3D:translation"
    quaternion_column: str = f"{entity}:Transform3D:quaternion"
    if translation_column not in table.column_names or quaternion_column not in table.column_names:
        return empty_trajectory()
    row_t_ns: Int64[ndarray, " n_rows"] = np.asarray(table[timeline].combine_chunks().cast(pa.int64()))
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
    # Equal masks: each component's null removal drops exactly `valid`'s false rows, matching the timestamps' mask.
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
    """Return one rig index per ``camera_names`` entry in caller order; None selects all ``camera_count`` declared cameras
    in rig order. ``statics`` is a single-row table of every camera's statics. Normalize hyphens to underscores in requested
    and recorded names (left-front == left_front). Raise ValueError for a missing name or two cameras sharing a name.
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
    profile: RigProfile,
    frame_stride: int,
    window_s: float,
    cache_video: bool = False,
    decode_device: Literal["cpu", "cuda"] = "cpu",
) -> SegmentFeed:
    """Read calibration, IMU, ground truth and frameset timing for one segment."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be at least 1; got {frame_stride}")
    rig_statics: pa.Table = sensor_dataset.filter_segments([segment_id]).filter_contents([RIG_ENTITY, IMU_ENTITY]).reader(index=None).to_arrow_table()
    reference: str = _static_string(rig_statics, f"{RIG_ENTITY}:reference")
    if reference != "imu_00":
        raise ValueError(f"{segment_id}: rig reference is {reference!r}, but the feed assumes the IMU is the rig frame")
    # Validate calibration before fetching or decoding the large camera payloads.
    calibration: ImuCalibration = ImuCalibration.from_catalog(rig_statics, IMU_ENTITY)
    applied_shift_ns: int = _static_int(rig_statics, f"{IMU_ENTITY}:applied_time_shift_ns")
    try:
        imu: ImuCalib = imu_calib(calibration, np.eye(4, dtype=np.float64), applied_shift_ns)
    except ValueError as error:
        raise ValueError(f"{segment_id}: {error}") from error
    camera_count: int = int(_static_values(rig_statics, f"{RIG_ENTITY}:num_cameras")[0])
    rig_entities: list[str] = [f"{RIG_ENTITY}/cam_{position:02d}" for position in range(camera_count)]
    camera_statics: pa.Table = (
        sensor_dataset.filter_segments([segment_id])
        .filter_contents(rig_entities + [f"{entity}/pinhole" for entity in rig_entities])
        .reader(index=None)
        .to_arrow_table()
    )
    camera_positions: tuple[int, ...] = select_cameras(camera_statics, camera_count, profile.camera_names)
    cached_videos: tuple[CatalogVideo, ...] | None = (
        read_catalog_videos(sensor_dataset, segment_id, [f"{rig_entities[position]}/pinhole/video" for position in camera_positions], TIMELINE)
        if cache_video else None
    )
    index: _VideoIndex = _read_video_index(sensor_dataset, segment_id, camera_positions, profile.frameset_tolerance_ns, cached_videos)
    cached_payloads: tuple[bytes, ...] | None = None
    if cached_videos is not None:
        # Mux one camera at a time and drop its packet views as soon as its MP4 exists,
        # so the peak holds the packets plus one payload rather than every payload.
        pending: list[CatalogVideo] = list(cached_videos)
        del cached_videos
        payloads: list[bytes] = []
        while pending:
            video: CatalogVideo = pending.pop(0)
            payloads.append(wrap_mp4(video.samples, video.keyframes, fps=index.fps, codec=video.codec))
            del video
        cached_payloads = tuple(payloads)
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
        imu=replace(imu, imu_T_body=imu_T_body),
        capture_start_time_ns=capture_start_time_ns,
        export_offset_ns=0 if profile.video_time_is_absolute else capture_start_time_ns,
        frame_t_ns=index.t_ns + imu.cam_time_offset_ns,
        camera_positions=camera_positions,
        rig_cameras=camera_count,
        profile=profile,
        frame_stride=frame_stride,
        dataset=sensor_dataset,
        gt_dataset=gt_dataset,
        index=index,
        window_ns=int(window_s * 1e9),
        cached_payloads=cached_payloads,
        decode_device=decode_device,
    )


@contextmanager
def open_segment(
    source: SegmentSource,
    profile: RigProfile = MSD_RIG,
    frame_stride: int = 1,
    window_s: float = DEFAULT_WINDOW_S,
    *,
    cache_video: bool = False,
    decode_device: Literal["cpu", "cuda"] = "cpu",
    include_ground_truth: bool = True,
) -> Iterator[SegmentFeed]:
    """Yield an open feed for ``source`` (local ``.rrd`` files or a catalog server), with ``profile`` rig read rules (MSD default).
    Local files need no catalog server: an in-process ``rr.server.Server`` uses an ephemeral port and stops on context exit.
    Base and truth use separate datasets because each in-process dataset accepts one layer per segment.
    ``frame_stride`` yields every n-th frameset while decoding all frames; ``window_s`` bounds each fetch's time window.
    ``cache_video`` fetches compressed streams together once; RAM scales with encoded segment size. ``decode_device`` selects
    CPU PyAV or CUDA TorchCodec (requires slam-rs-cuda). ``include_ground_truth`` reads isolated GT for scoring; False reads
    only estimator inputs. Raise ValueError if the local base ``.rrd`` does not contain exactly one segment.
    """
    if isinstance(source, LocalSegment):
        datasets: dict[str, str | PathLike[str] | Sequence[str | PathLike[str]]] = {"base": [str(source.base_rrd)]}
        if include_ground_truth and source.gt_rrd is not None:
            datasets["gt"] = [str(source.gt_rrd)]
        with rr.server.Server(datasets=datasets) as server:
            client: CatalogClient = server.client()
            base: DatasetEntry = client.get_dataset("base")
            ground_truth: DatasetEntry | None = client.get_dataset("gt") if "gt" in datasets else None
            segment_ids: list[str] = list(base.segment_ids())
            if len(segment_ids) != 1:
                raise ValueError(f"{source.base_rrd} holds {len(segment_ids)} segments; the feed reads one")
            yield _build_feed(base, ground_truth, segment_ids[0], profile, frame_stride, window_s, cache_video, decode_device)
    else:
        resolved: CatalogSegment = source if source.dataset is not None else resolve_catalog_segments((source,))[0]
        assert resolved.dataset is not None
        with ExitStack() as stack:
            ground_truth: DatasetEntry | None = None
            if include_ground_truth and resolved.ground_truth_uri is not None:
                uri: ParseResult = urlparse(resolved.ground_truth_uri)
                if uri.scheme == "file" and (uri.netloc not in ("", "localhost") or not Path(unquote(uri.path)).is_file()):
                    raise ValueError(
                        f"{resolved.segment_id}: registered ground truth {resolved.ground_truth_uri} is not accessible on this worker; "
                        "mount the catalog's input storage at the same path to isolate ground truth from estimated poses"
                    )
                # The public catalog API merges layers. A GT-only local catalog
                # keeps estimates on the animated rig out of every scoring read.
                gt_server: rr.server.Server = stack.enter_context(rr.server.Server())
                ground_truth = gt_server.client().create_dataset("gt")
                ground_truth.register([resolved.ground_truth_uri]).wait()
            yield _build_feed(
                resolved.dataset, ground_truth,
                resolved.segment_id, profile, frame_stride, window_s, cache_video, decode_device,
            )
