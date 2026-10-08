"""One robocap catalog segment as the hands core's input: the rig, the six cameras' framesets of raw luma, and the SLAM poses.

Everything comes from the Rerun catalog and nothing is written to disk on the way: the six encoded camera streams are
fetched in one bulk query (simplecv's ``read_catalog_videos``, compressed, a few MB per second of video), decoded on the CPU
in lockstep and handed to ``robocap_live._core`` one frameset at a time.

Decisions that change the numbers. They are the ones a ``robocap-live-dump/1`` dump of the segment carries (``frame.rs``), so the
layer equals a ``robocap-live --source replay <dump> --slam reference`` run on such a dump:

- **Luma is the decoder's raw Y plane** (libavcodec, no range or colour conversion). On the cap the hand pipeline reads the
  NV12 Y plane the recorder then encodes, so the raw decoded Y is that signal, up to the H.264 loss. The RoboCap streams carry
  full-range luma with the range left unspecified (s10 cam_00: Y 44..255, p99 = 255), so a ``gray8`` reformat would treat it
  as limited range: a 255/219 stretch that moves values by up to 20 LSB and clips everything above 235. (slam_rs's catalog
  feed does the ``gray8`` reformat.)
- **A frameset is what the live adapter groups**: it opens at the earliest frame not yet taken and holds, per camera, that
  camera's next frame if it lies within ``group_tolerance_ns`` (3 ms live) of the opening frame; its time is the opening
  frame's. A camera without a frame there is absent from the frameset.
- **Poses** are the segment's ``/world/rig_00`` transforms on ``video_time`` (the ``slam_rs`` layer): each frameset takes the
  nearest within ``pose_tolerance_ns`` (omitted otherwise), the dump's reference-pose table. The core's pipeline looks each
  frameset's pose up there and tracks a frameset without one on the newest earlier pose, as the runtime's ``--slam reference``
  replay does.
- **Times** stay on the catalog's ``video_time``, so the layer lines up with the ``base`` and ``slam_rs`` layers.
"""

import queue
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from io import BytesIO

import av
import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rrd_query_utils import first_valid_value

from robocap_live import _core

CAMERA_NAMES: tuple[str, ...] = tuple(_core.CAMERA_NAMES)
"""Camera names in index order (``frame.rs::CAMERA_NAMES``; catalog ``/world/rig_00/cam_00..cam_05``)."""
RIG_ENTITY: str = "/world/rig_00"
CAMERA_ENTITIES: tuple[str, ...] = tuple(f"{RIG_ENTITY}/cam_{camera:02d}" for camera in range(len(CAMERA_NAMES)))
TIMELINE: str = "video_time"
CHILD_FROM_PARENT: int = rr.components.TransformRelation.ChildFromParent.value
"""``Transform3D:relation`` of a camera node: the extrinsic is camera-from-rig."""


# --- framesets and poses -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FramesetIndex:
    """Which frame of each camera belongs to which frameset."""

    t_ns: Int64[ndarray, " f"]
    """The opening (earliest) frame's ``video_time``."""
    frame: Int64[ndarray, "f c"]
    """Frame number within its camera, -1 where the camera has no frame in the frameset."""
    cam_t_ns: Int64[ndarray, "f c"]
    """Each camera's frame time, 0 where absent."""

    def head(self, count: int) -> "FramesetIndex":
        """The first ``count`` framesets."""
        return FramesetIndex(self.t_ns[:count], self.frame[:count], self.cam_t_ns[:count])


def group_framesets(camera_t_ns: list[Int64[ndarray, " n"]], tolerance_ns: int) -> FramesetIndex:
    """Group the cameras' frames as the live adapter does (see the module docstring).

    Raises:
        ValueError: If the tolerance is negative or a camera's frame times are not strictly increasing.
    """
    if tolerance_ns < 0:
        raise ValueError(f"group tolerance {tolerance_ns} ns is negative")
    for camera, times in enumerate(camera_t_ns):
        if times.size and not bool(np.all(np.diff(times) > 0)):
            raise ValueError(f"camera {camera}: frame times are not strictly increasing")
    cameras: int = len(camera_t_ns)
    cursor: list[int] = [0] * cameras
    t_rows: list[int] = []
    frame_rows: list[list[int]] = []
    time_rows: list[list[int]] = []
    while True:
        heads: list[int] = [int(times[cursor[c]]) for c, times in enumerate(camera_t_ns) if cursor[c] < len(times)]
        if not heads:
            break
        opening: int = min(heads)
        frames: list[int] = [-1] * cameras
        stamps: list[int] = [0] * cameras
        for c, times in enumerate(camera_t_ns):
            if cursor[c] < len(times) and int(times[cursor[c]]) - opening <= tolerance_ns:
                frames[c] = cursor[c]
                stamps[c] = int(times[cursor[c]])
                cursor[c] += 1
        t_rows.append(opening)
        frame_rows.append(frames)
        time_rows.append(stamps)
    return FramesetIndex(
        t_ns=np.asarray(t_rows, dtype=np.int64),
        frame=np.asarray(frame_rows, dtype=np.int64).reshape(-1, cameras),
        cam_t_ns=np.asarray(time_rows, dtype=np.int64).reshape(-1, cameras),
    )


@dataclass(frozen=True, slots=True)
class CatalogPoses:
    """The rig's SLAM poses, time-sorted, finite rows only."""

    t_ns: Int64[ndarray, " p"]
    world_from_rig: Float64[ndarray, "p 4 4"]


def nearest_poses(frame_t_ns: Int64[ndarray, " f"], poses: CatalogPoses, tolerance_ns: int) -> CatalogPoses:
    """Matched poses within ``tolerance_ns``, keyed by the matched frameset times."""
    if not len(poses.t_ns):
        return CatalogPoses(np.empty(0, dtype=np.int64), np.empty((0, 4, 4), dtype=np.float64))
    right: Int64[ndarray, " f"] = np.clip(np.searchsorted(poses.t_ns, frame_t_ns), 0, len(poses.t_ns) - 1)
    left: Int64[ndarray, " f"] = np.clip(right - 1, 0, len(poses.t_ns) - 1)
    nearest: Int64[ndarray, " f"] = np.where(np.abs(poses.t_ns[left] - frame_t_ns) <= np.abs(poses.t_ns[right] - frame_t_ns), left, right)
    close: Bool[ndarray, " f"] = np.abs(poses.t_ns[nearest] - frame_t_ns) <= tolerance_ns
    return CatalogPoses(frame_t_ns[close], poses.world_from_rig[nearest[close]])


# --- the catalog -------------------------------------------------------------------------------------------------------


def _floats(statics: pa.Table, column: str) -> Float64[ndarray, " n"]:
    return np.asarray(first_valid_value(statics[column], component_name=column), dtype=np.float64).ravel()


def _text(statics: pa.Table, column: str) -> str:
    value: object = first_valid_value(statics[column], component_name=column)
    if isinstance(value, list):
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"static {column} is not text: {value!r}")
    return value


def read_rig(dataset: DatasetEntry, segment_id: str, source: str) -> _core.Rig:
    """The six cameras' extrinsics, intrinsics and Fisheye62 coefficients from the segment's statics (the ``base`` layer).

    Rerun stores ``mat3x3`` and the Pinhole K column-major as float32; widening them to float64 is exact. The rig's
    ``device`` is the segment's device id (the serial in the segment id); the hands do not use it.

    Raises:
        ValueError: If a camera is out of order, its extrinsic is not camera-from-rig, or its lens is not Kannala-Brandt.
    """
    entities: list[str] = [*CAMERA_ENTITIES, *(f"{camera}/pinhole" for camera in CAMERA_ENTITIES)]
    statics: pa.Table = dataset.filter_segments(segment_id).filter_contents(entities).reader(index=None).to_arrow_table()
    where: str = f"{source} {segment_id}"
    try:
        cam_from_rig: Float64[ndarray, "c 4 4"] = np.tile(np.eye(4), (len(CAMERA_NAMES), 1, 1))
        resolution: Float64[ndarray, "c 2"] = np.zeros((len(CAMERA_NAMES), 2))
        focal: Float64[ndarray, "c 2"] = np.zeros((len(CAMERA_NAMES), 2))
        principal: Float64[ndarray, "c 2"] = np.zeros((len(CAMERA_NAMES), 2))
        fisheye62: Float64[ndarray, "c 8"] = np.zeros((len(CAMERA_NAMES), 8))
        names: list[str] = []
        for camera, entity in enumerate(CAMERA_ENTITIES):
            names.append(_text(statics, f"{entity}:name"))
            if int(_floats(statics, f"{entity}:Transform3D:relation")[0]) != CHILD_FROM_PARENT:
                raise ValueError(f"{where}: {entity}'s extrinsic is not ChildFromParent (camera from rig)")
            cam_from_rig[camera, :3, :3] = _floats(statics, f"{entity}:Transform3D:mat3x3").reshape(3, 3, order="F")
            cam_from_rig[camera, :3, 3] = _floats(statics, f"{entity}:Transform3D:translation")
            intrinsics: Float64[ndarray, "3 3"] = _floats(statics, f"{entity}/pinhole:Pinhole:image_from_camera").reshape(3, 3, order="F")
            focal[camera] = (intrinsics[0, 0], intrinsics[1, 1])
            principal[camera] = (intrinsics[0, 2], intrinsics[1, 2])
            resolution[camera] = _floats(statics, f"{entity}/pinhole:Pinhole:resolution")
            model: str = _text(statics, f"{entity}/pinhole:simplecv.components.DistortionModel")
            coefficients: Float64[ndarray, " k"] = _floats(statics, f"{entity}/pinhole:simplecv.components.DistortionCoefficients")
            if model != "kannala_brandt" or len(coefficients) != 8:
                raise ValueError(f"{where}: {entity} has {model} with {len(coefficients)} coefficients; expected Fisheye62 [k1..k6, p1, p2]")
            fisheye62[camera] = coefficients
        parts: list[str] = segment_id.split("__")
        device: str = parts[1] if len(parts) > 2 else segment_id
        return _core.Rig(names=names, resolution_wh=resolution, cam_from_rig=cam_from_rig, focal=focal, principal=principal,
                         fisheye62=fisheye62, source=f"{source} {segment_id}", device=device)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{where}: {error}") from error


def read_poses(dataset: DatasetEntry, segment_id: str) -> CatalogPoses:
    """``/world/rig_00``'s temporal Transform3D on ``video_time`` (the ``slam_rs`` layer), sorted, finite rows only."""
    quaternion_column: str = f"{RIG_ENTITY}:Transform3D:quaternion"
    translation_column: str = f"{RIG_ENTITY}:Transform3D:translation"
    table: pa.Table = (dataset.filter_segments(segment_id).filter_contents([RIG_ENTITY]).reader(index=TIMELINE)
                       .select(TIMELINE, quaternion_column, translation_column).to_arrow_table())
    t_ns: Int64[ndarray, " r"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))
    quaternion: Float64[ndarray, "r 4"] = _list_rows(table[quaternion_column], 4)
    translation: Float64[ndarray, "r 3"] = _list_rows(table[translation_column], 3)
    valid: Bool[ndarray, " r"] = np.isfinite(quaternion).all(axis=1) & np.isfinite(translation).all(axis=1) & (np.linalg.norm(quaternion, axis=1) > 0)
    order: Int64[ndarray, " p"] = np.flatnonzero(valid)[np.argsort(t_ns[valid], kind="stable")]
    world_from_rig: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(order), 1, 1))
    world_from_rig[:, :3, :3] = Rotation.from_quat(quaternion[order]).as_matrix()  # xyzw, normalised
    world_from_rig[:, :3, 3] = translation[order]
    return CatalogPoses(t_ns[order], world_from_rig)


def _list_rows(column: pa.ChunkedArray, width: int) -> Float64[ndarray, "r w"]:
    """A ``list<fixed_size_list<width>>`` component column as one row per table row; NaN where the row has no value."""
    out: Float64[ndarray, "r w"] = np.full((len(column), width), np.nan)
    for row, cell in enumerate(column.to_pylist()):
        if cell:
            out[row] = np.asarray(cell[0] if isinstance(cell[0], list) else cell, dtype=np.float64)[:width]
    return out


# --- decoding ----------------------------------------------------------------------------------------------------------


def y_planes(video: CatalogVideo, fps: int, threads: int) -> Iterator[UInt8[ndarray, "h w"]]:
    """The camera's frames in order as the decoder's Y plane (libavcodec on the CPU, no range conversion).

    Each plane is a view of the decoded frame's memory, which the view keeps alive: no copy here and, for a plane without
    row padding (RoboCap's 1920-wide H.264 has none), none in the core either.

    Raises:
        ValueError: If a frame decodes to a pixel format without a plain luma plane.
    """
    container: av.container.InputContainer = av.open(BytesIO(wrap_mp4(video.samples, video.keyframes, fps, codec=video.codec)), mode="r")
    with container:
        stream: av.video.stream.VideoStream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = threads
        for frame in container.decode(stream):
            if frame.format.name not in ("yuv420p", "yuvj420p", "nv12", "gray"):
                raise ValueError(f"decoded pixel format {frame.format.name}: no plain Y plane")
            plane: av.video.plane.VideoPlane = frame.planes[0]
            padded: UInt8[ndarray, "h stride"] = np.frombuffer(plane, dtype=np.uint8).reshape(frame.height, plane.line_size)
            yield padded[:, : frame.width]


@dataclass(frozen=True, slots=True)
class Frameset:
    """One frameset as the core takes it."""

    index: int
    t_ns: int
    """The catalog ``video_time`` of its earliest frame."""
    cam_t_ns: list[int]
    """Each camera's frame time (the frameset's time where the camera is absent)."""
    luma: list[UInt8[ndarray, "h w"] | None]
    """Per camera, the 1920x1080 raw Y plane; None where absent."""


@dataclass(slots=True)
class DecodeStats:
    """What the decode thread spent, filled in while it runs."""

    framesets: int = 0
    busy_s: float = 0.0
    """Time spent decoding (and waiting on the decoders), excluding time blocked on a full queue."""
    blocked_s: float = 0.0
    """Time the decode thread waited for the consumer."""


@dataclass(slots=True)
class CatalogSegment:
    """One segment read from the catalog, ready to decode: the rig, the encoded videos, the frameset index and the poses."""

    segment_id: str
    rig: _core.Rig
    videos: tuple[CatalogVideo, ...]
    index: FramesetIndex
    poses: CatalogPoses
    """Nearest SLAM poses within the tolerance, keyed by matched frameset time."""
    fps: int
    """Nominal frame rate for the MP4 wrapper, from the frameset times."""
    read_s: float
    """Wall time of the catalog queries."""
    decode: DecodeStats = field(default_factory=DecodeStats)

    def framesets(self, decode_threads: int = 2, prefetch: int = 4) -> Generator[Frameset, None, None]:
        """Decode every camera in lockstep on a background thread and yield the framesets in order.

        The thread runs ``prefetch`` framesets ahead, so decoding overlaps the core's pipeline (a push releases the GIL).

        Raises:
            ValueError: If a camera decodes fewer or more frames than it has packets.
        """
        frames: queue.Queue[Frameset | BaseException | None] = queue.Queue(maxsize=max(1, prefetch))
        stop: threading.Event = threading.Event()

        def offer(item: Frameset | BaseException | None) -> bool:
            """Queue ``item`` unless the consumer has stopped; False once it has."""
            while not stop.is_set():
                try:
                    frames.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            try:
                started: float = time.perf_counter()
                for frameset in self._decode(decode_threads):
                    ready: float = time.perf_counter()
                    self.decode.busy_s += ready - started
                    if not offer(frameset):
                        return
                    started = time.perf_counter()
                    self.decode.blocked_s += started - ready
                    self.decode.framesets += 1
                offer(None)
            except BaseException as error:  # handed to the consumer, which re-raises it
                offer(error)

        thread: threading.Thread = threading.Thread(target=produce, name="robocap-decode", daemon=True)
        thread.start()
        try:
            while True:
                item: Frameset | BaseException | None = frames.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            thread.join()

    def _decode(self, threads: int) -> Iterator[Frameset]:
        decoders: list[Iterator[UInt8[ndarray, "h w"]]] = [y_planes(video, self.fps, threads) for video in self.videos]
        decoded: list[int] = [0] * len(self.videos)
        for row in range(len(self.index.t_ns)):
            luma: list[UInt8[ndarray, "h w"] | None] = []
            for camera, decoder in enumerate(decoders):
                wanted: int = int(self.index.frame[row, camera])
                if wanted < 0:
                    luma.append(None)
                    continue
                if wanted != decoded[camera]:
                    raise ValueError(f"camera {camera}: frameset {row} wants frame {wanted}, the decoder is at {decoded[camera]}")
                image: UInt8[ndarray, "h w"] | None = next(decoder, None)
                if image is None:
                    raise ValueError(f"camera {camera}: the decoder ended after {decoded[camera]} of {len(self.videos[camera].samples)} frames")
                decoded[camera] += 1
                luma.append(image)
            t_ns: int = int(self.index.t_ns[row])
            yield Frameset(
                index=row,
                t_ns=t_ns,
                cam_t_ns=[int(t) if frame >= 0 else t_ns for t, frame in zip(self.index.cam_t_ns[row], self.index.frame[row], strict=True)],
                luma=luma,
            )
        for camera, decoder in enumerate(decoders):
            if decoded[camera] == len(self.videos[camera].samples) and next(decoder, None) is not None:
                raise ValueError(f"camera {camera}: more frames decoded than its {len(self.videos[camera].samples)} packets")


def open_segment(dataset: DatasetEntry, segment_id: str, *, source: str, group_tolerance_ns: int = 3_000_000,
                 pose_tolerance_ns: int = 5_000_000, max_framesets: int | None = None) -> CatalogSegment:
    """Read one segment's rig, encoded videos and poses from the catalog and group its framesets.

    Raises:
        ValueError: If a camera stream is missing or out of order, or the segment has fewer than two framesets.
    """
    started: float = time.perf_counter()
    rig: _core.Rig = read_rig(dataset, segment_id, source)
    videos: tuple[CatalogVideo, ...] = read_catalog_videos(dataset, segment_id, [f"{camera}/pinhole/video" for camera in CAMERA_ENTITIES], TIMELINE)
    poses: CatalogPoses = read_poses(dataset, segment_id)
    read_s: float = time.perf_counter() - started
    index: FramesetIndex = group_framesets([video.t_ns for video in videos], group_tolerance_ns)
    if max_framesets is not None:
        index = index.head(max_framesets)
    if len(index.t_ns) < 2:
        raise ValueError(f"{segment_id}: {len(index.t_ns)} framesets is not a segment")
    fps: int = max(1, round((len(index.t_ns) - 1) * 1e9 / float(index.t_ns[-1] - index.t_ns[0])))
    return CatalogSegment(segment_id=segment_id, rig=rig, videos=videos, index=index,
                          poses=nearest_poses(index.t_ns, poses, pose_tolerance_ns), fps=fps, read_s=read_s)
