"""Build a ``preview`` layer for segment-table cards: the middle window of one segment on its own timeline.

Everything a card shows lives under /preview on the timeline ``preview_time`` (same values as video_time, but only the
window exists there), so a card set to ``preview_time`` loops over just this window:
  /preview/body/skeleton   copy of /world/skeleton (Points3D + its AnnotationContext), window rows only
  /preview/body/soma_xlo   copy of /world/pred/soma_x/mesh_xlo (the soma_xlo layer), window rows only
  /preview/center          static Transform3D to the window's mean skeleton position: the card's 3D view origin, so a
                           fixed card camera fits around the body (world = root frame; /world has no transform)
  /world/rig_NN            the moving rigs' poses (the Quest) copied onto preview_time at their own paths, so their
                           device meshes (dataset assets) and frustums follow in the card. Each pose chunk also carries an
                           empty Points3D:positions column: Rerun 0.41 previews prefetch transforms only on the timeline
                           with the most rows (video_time, logic.rs:1001, RR-4257), so a pose chunk on preview_time loads
                           only when a visualizer query pulls it in. The empty points draw nothing.
  /preview/exo_cam         the exo phone's static camera pose (copy of /world/<exo>/cam_00), and under it
  /preview/exo_cam/pinhole its pinhole scaled to 480x270, so the card's 2D view projects the 3D skeleton and xlo mesh
                           onto the exo video exactly like the main view's exo panels (static poses load in previews)
    .../pinhole/video      the exo phone video, re-encoded to 480x270 AV1 (low delay, keyframe every 30 frames)
    .../pinhole/ego        2D offset below the exo image, so both videos stack in one view:
    .../pinhole/ego/video      the OAK RGB video, same encoding
    .../pinhole/ego/keypoints  the OAK's own 2D skeleton (pinhole/skeleton), scaled to 480x270, as in the main view
                           (the main view hides the body mesh on the OAK camera too)

Videos and body keep the source frame rate unless ``fps`` is given. Copies get fresh row and chunk ids. Video frames keep
their source timestamps (frame i of the re-encode = sample i), and only the samples from the keyframe before the window to
its end are fetched, found through ``VideoStream:is_keyframe``. The source resolutions are checked against the pinholes,
and a segment with a SOMA mesh but no soma_xlo rows in the window fails. The OAK has a static mount transform with
parent_frame on the Quest rig, so only the Quest pose moves. The file holds no segment properties: a second
``/__properties`` chunk would collide with the base's when the catalog merges layers.
"""

import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
from dataforge.schema import TIMELINE, cam_path, pinhole_path, video_path
from dataforge.selfcap_schema import (
    PREVIEW_CENTER,
    PREVIEW_EXO_CAM,
    PREVIEW_EXO_PINHOLE,
    PREVIEW_SKELETON,
    PREVIEW_SOMA_XLO,
    PREVIEW_TIMELINE,
    PREVIEW_VIDEO_GAP_PX,
    PREVIEW_VIDEO_SIZE,
    SKELETON,
    SOMA_MESH,
    SOMA_XLO,
)
from dataforge.video_encoding import resolve_ffmpeg
from dataforge.writing import atomic_write
from datafusion import col, lit
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry, Schema
from rerun.chunk import Chunk, LazyChunkStream, LazyStore, OptimizationProfile
from simplecv.catalog_rig_layout import CatalogRigLayout, catalog_components
from simplecv.catalog_video import CatalogVideo, catalog_codec, catalog_keyframes, packet_views
from simplecv.catalog_video_codec import CatalogCodecName, wrap_mp4

from selfcap_curate.catalog_io import rig_layout, select_layout

EXO_SIZE: tuple[int, int] = (1920, 1080)
"""Width and height the phone pinholes and videos must have."""
EGO_SIZE: tuple[int, int] = (1280, 720)
"""Width and height the OAK pinhole and video must have."""
APPLICATION_ID: str = "selfcap-preview"
"""Application id of a preview file; the catalog stacks layers by recording id."""
EXO_PREFERENCE: tuple[int, ...] = (2, 1, 0, 3)
"""Phone rigs in the order the card prefers them."""
MUX_FPS: int = 30
"""Nominal rate of the source mp4 handed to ffmpeg; its frames are selected by position, so the value does not matter."""


RigChoice: TypeAlias = int | None
"""A rig index, or None to choose from what the segment holds."""


@dataclass(frozen=True, slots=True)
class PreviewOptions:
    """What one preview layer selects from its segment."""

    seconds: float = 30.0
    """Length of the middle window."""
    fps: float | None = None
    """Thin videos and body to about this rate; None keeps the source rate."""
    ego: RigChoice = None
    """OAK rig index; None reads it from the rig ``kind`` metadata."""
    exo: RigChoice = None
    """Exo phone rig index; None takes the first posed phone, rig 2 first."""


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """What one build wrote."""

    segment: str
    """The segment, also the recording id of the file."""
    target: Path
    """The finished layer file."""
    lo_ns: int
    """Window start on the source timeline, inclusive."""
    hi_ns: int
    """Window end on the source timeline, exclusive."""
    ego_rig: int
    """OAK rig the ego video and keypoints came from."""
    exo_rig: int
    """Phone rig the exo video came from."""
    exo_camera_model: bool
    """Whether the phone is posed, so the card projects 3D content onto its video."""
    moving_rigs: tuple[str, ...]
    """Entities whose poses were copied onto the preview timeline."""
    exo_frames: int
    """Frames of the re-encoded exo video."""
    ego_frames: int
    """Frames of the re-encoded ego video."""
    file_bytes: int
    """Size of the finished file."""


@dataclass(frozen=True, slots=True)
class CameraModel:
    """The static chunks that place and project one camera."""

    pose: list[Chunk]
    """Static Transform3D chunks of the camera node."""
    pinhole: list[Chunk]
    """Static Pinhole chunks of the pinhole node."""

    @property
    def is_posed(self) -> bool:
        """Whether both a pose and a pinhole exist."""
        return bool(self.pose and self.pinhole)


@dataclass(frozen=True, slots=True)
class VideoIndex:
    """Timing of one video, read without its samples."""

    entity: str
    """The VideoStream entity."""
    times: Int64[ndarray, "n"]
    """Sample times on the source timeline, strictly increasing, in nanoseconds."""
    keyframes: Bool[ndarray, "n"]
    """Whether each sample is a keyframe."""


def has_component(chunk: Chunk, prefix: str) -> bool:
    """Whether a chunk holds a column whose name starts with ``prefix``."""
    return any(name.startswith(prefix) for name in chunk.to_record_batch().schema.names)


def read_camera_models(store: LazyStore, rigs: list[int]) -> dict[int, CameraModel]:
    """The static pose and pinhole chunks of each rig's ``cam_00``, from one scan."""
    entities: list[str] = [entity for rig in rigs for entity in (cam_path(rig, 0), pinhole_path(rig, 0))]
    found: dict[str, list[Chunk]] = {entity: [] for entity in entities}
    for chunk in store.stream().filter(content=entities, is_static=True).to_chunks():
        found[str(chunk.entity_path)].append(chunk)
    return {
        rig: CameraModel(
            [chunk for chunk in found[cam_path(rig, 0)] if has_component(chunk, "Transform3D:")],
            [chunk for chunk in found[pinhole_path(rig, 0)] if has_component(chunk, "Pinhole:")],
        )
        for rig in rigs
    }


def exo_candidates(entry: DatasetEntry, segment: str, options: PreviewOptions) -> tuple[int, list[int]]:
    """The OAK rig and the phones the exo may come from, best first; a named rig needs no layout.

    Returns:
        The OAK rig and the phone candidates.

    Raises:
        ValueError: If the segment does not have a SelfCap rig layout and a rig is left to choose.
    """
    if options.ego is not None and options.exo is not None:
        return options.ego, [options.exo]
    layout: CatalogRigLayout = rig_layout(entry, segment)
    ego_rig: int = select_layout(layout, ()).oak_rig if options.ego is None else options.ego
    if options.exo is not None:
        return ego_rig, [options.exo]
    phones: set[int] = {int(camera.rig.removeprefix("rig_")) for camera in layout.cameras if camera.rig_kind == "exo"}
    return ego_rig, [rig for rig in EXO_PREFERENCE if rig in phones]


def scale_points_2d(batch: pa.RecordBatch, scale: float) -> pa.RecordBatch:
    """Scale Points2D positions (and radii given in pixels, i.e. positive) from source to preview resolution."""
    columns: list[pa.Array] = []
    for name, column in zip(batch.schema.names, batch.columns, strict=True):
        if name == "Points2D:positions":
            xy: pa.Array = column.flatten()
            scaled_xy: Float32[ndarray, "m"] = (xy.flatten().to_numpy(zero_copy_only=False).astype(np.float64) * scale).astype(np.float32)
            column = pa.ListArray.from_arrays(column.offsets, pa.FixedSizeListArray.from_arrays(pa.array(scaled_xy), type=xy.type), type=column.type, mask=column.is_null())
        elif name == "Points2D:radii":
            radii: Float64[ndarray, "m"] = column.flatten().to_numpy(zero_copy_only=False).astype(np.float64)
            scaled_radii: Float32[ndarray, "m"] = np.where(radii > 0, radii * scale, radii).astype(np.float32)
            column = pa.ListArray.from_arrays(column.offsets, pa.array(scaled_radii), type=column.type, mask=column.is_null())
        columns.append(column)
    return pa.RecordBatch.from_arrays(columns, schema=batch.schema)


def scale_pinhole(batch: pa.RecordBatch, scale: float) -> pa.RecordBatch:
    """Scale Pinhole intrinsics (fx, fy, skew, cx, cy) and resolution to a smaller image of the same camera."""
    columns: list[pa.Array] = []
    for name, column in zip(batch.schema.names, batch.columns, strict=True):
        if name in ("Pinhole:image_from_camera", "Pinhole:resolution"):
            inner: pa.Array = column.flatten()
            values: Float32[ndarray, "n k"] = inner.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(inner), -1)
            if name == "Pinhole:resolution":
                values = values * scale
            else:  # Mat3x3, column-major: rows 0 and 1 of every column scale, the bottom row (0, 0, 1) stays
                if not np.allclose(values[:, [2, 5, 8]], [0.0, 0.0, 1.0]):
                    raise ValueError(f"unexpected intrinsics layout {values[0]}")
                values[:, [0, 1, 3, 4, 6, 7]] *= scale
            scaled: pa.Array = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), pa.float32()), type=inner.type)
            column = pa.ListArray.from_arrays(column.offsets, scaled, type=column.type, mask=column.is_null())
        columns.append(column)
    return pa.RecordBatch.from_arrays(columns, schema=batch.schema)


def check_resolution(model: CameraModel, segment: str, entity: str, size: tuple[int, int]) -> None:
    """Refuse a camera whose pinhole does not have the source size the card's scale factors assume."""
    resolutions: list[list[int]] = [
        np.rint(batch.column("Pinhole:resolution").flatten().flatten().to_numpy(zero_copy_only=False)[:2]).astype(int).tolist()
        for batch in (chunk.to_record_batch() for chunk in model.pinhole)
        if "Pinhole:resolution" in batch.schema.names
    ]
    if not resolutions or resolutions[0] != list(size):
        raise ValueError(f"{segment}: {entity} resolution {resolutions[:1]} != {size}")


def copy_window(
    chunk: Chunk,
    path: str,
    lo: int,
    hi: int,
    *,
    fps: float | None = None,
    points_scale: float | None = None,
    query_anchor: bool = False,
    pinhole_scale: float | None = None,
) -> list[Chunk]:
    """Re-path one chunk; temporal rows are cut to [lo, hi) and moved from the source timeline to preview_time."""
    batch: pa.RecordBatch = chunk.to_record_batch()
    if points_scale is not None:
        batch = scale_points_2d(batch, points_scale)
    if pinhole_scale is not None:
        batch = scale_pinhole(batch, pinhole_scale)
    data: list[int] = [i for i, field in enumerate(batch.schema) if (field.metadata or {}).get(b"rerun:kind") == b"data"]
    meta: dict[bytes, bytes] = {k: v for k, v in (batch.schema.metadata or {}).items() if k not in (b"rerun:id", b"rerun:entity_path")}
    meta[b"rerun:entity_path"] = path.encode()
    if chunk.is_static:
        out: pa.RecordBatch = pa.RecordBatch.from_arrays([batch.column(i) for i in data], schema=pa.schema([batch.schema.field(i) for i in data], meta))
        return Chunk.from_record_batch(out, index=None, entity_path=path)
    if TIMELINE not in chunk.timeline_names:
        return []
    stamps: Int64[ndarray, "r"] = batch.column(TIMELINE).cast(pa.int64()).to_numpy()
    batch = batch.filter(pa.array((stamps >= lo) & (stamps < hi)))
    if batch.num_rows == 0:
        return []
    if fps:
        slot: Int64[ndarray, "n"] = (batch.column(TIMELINE).cast(pa.int64()).to_numpy() - lo) // int(1e9 / fps)
        batch = batch.filter(pa.array(np.r_[True, slot[1:] != slot[:-1]]))
    time_field: pa.Field = pa.field(PREVIEW_TIMELINE, pa.duration("ns"), metadata={"rerun:kind": "index", "rerun:index_name": PREVIEW_TIMELINE})
    fields: list[pa.Field] = [time_field, *(batch.schema.field(i) for i in data)]
    columns: list[pa.Array] = [batch.column(TIMELINE), *(batch.column(i) for i in data)]
    if query_anchor:
        anchor: tuple[pa.Field, pa.Array] = empty_points_column(batch.num_rows)
        fields.append(anchor[0])
        columns.append(anchor[1])
    out = pa.RecordBatch.from_arrays(columns, schema=pa.schema(fields, meta))
    return Chunk.from_record_batch(out, index=PREVIEW_TIMELINE, entity_path=path)


def empty_points_column(row_count: int) -> tuple[pa.Field, pa.Array]:
    """A Points3D:positions column with no points in each row (makes a pose chunk visible to previews)."""
    reference: Chunk = Chunk.from_columns(
        "/reference",
        indexes=[rr.TimeColumn(PREVIEW_TIMELINE, duration=np.zeros(row_count, "timedelta64[ns]"))],
        columns=rr.Points3D.columns(positions=np.zeros((0, 3), np.float32)).partition([0] * row_count),
    )
    batch: pa.RecordBatch = reference.to_record_batch()
    return batch.schema.field("Points3D:positions"), batch.column("Points3D:positions")


def body_center(chunks: list[Chunk]) -> Float32[ndarray, "3"]:
    """Mean of all finite skeleton positions in the copied window."""
    points: list[Float32[ndarray, "n 3"]] = []
    for chunk in chunks:
        batch: pa.RecordBatch = chunk.to_record_batch()
        if not chunk.is_static and "Points3D:positions" in batch.schema.names:
            values: pa.Array = batch.column("Points3D:positions").flatten().flatten()
            points.append(values.to_numpy(zero_copy_only=False).reshape(-1, 3))
    xyz: Float32[ndarray, "n 3"] = np.concatenate(points) if points else np.zeros((0, 3), np.float32)
    xyz = xyz[np.isfinite(xyz).all(1)]
    return xyz.mean(axis=0) if len(xyz) else np.zeros(3, np.float32)


def read_video_index(entry: DatasetEntry, segment: str, entity: str) -> VideoIndex:
    """Every sample time and keyframe flag of a video; the sparse flag column leaves the samples on the server."""
    flags: str = f"{entity}:VideoStream:is_keyframe"
    table: pa.Table = entry.filter_segments(segment).filter_contents([entity]).reader(index=TIMELINE).select(TIMELINE, flags).to_arrow_table()
    times: Int64[ndarray, "n"] = table[TIMELINE].cast(pa.int64()).to_numpy()
    if not len(times):
        raise ValueError(f"{entity} has no video samples on {TIMELINE}")
    if not np.all(np.diff(times) > 0):
        raise ValueError(f"{entity}: expected strictly increasing sample times")
    return VideoIndex(entity, times, np.asarray(catalog_keyframes(table[flags]), dtype=bool))


def read_video_samples(entry: DatasetEntry, segment: str, index: VideoIndex, first: int, last: int) -> CatalogVideo:
    """Samples ``first`` to ``last`` (exclusive) of a video, as views into the Arrow column.

    Raises:
        RuntimeError: If the catalog returns other samples than the index promised.
    """
    sample_column: str = f"{index.entity}:VideoStream:sample"
    codec_column: str = f"{index.entity}:VideoStream:codec"
    start: pa.Scalar = pa.scalar(int(index.times[first]), pa.duration("ns"))
    stop: pa.Scalar = pa.scalar(int(index.times[last - 1]), pa.duration("ns"))
    table: pa.Table = (
        entry.filter_segments(segment)
        .filter_contents([index.entity])
        .reader(index=TIMELINE)
        .filter((col(TIMELINE) >= lit(start)) & (col(TIMELINE) <= lit(stop)))
        .select(TIMELINE, sample_column, codec_column)
        .to_arrow_table()
    )
    times: Int64[ndarray, "k"] = index.times[first:last]
    if not np.array_equal(table[TIMELINE].cast(pa.int64()).to_numpy(), times):
        raise RuntimeError(f"{index.entity}: the catalog returned {table.num_rows} samples for {len(times)} indexed ones")
    codec: CatalogCodecName = catalog_codec(table[codec_column], f"{segment} {index.entity}")
    return CatalogVideo(times, packet_views(table[sample_column]), index.keyframes[first:last].tolist(), codec)


def read_ivf(path: Path) -> list[bytes]:
    """The AV1 temporal units of an IVF file."""
    data: bytes = path.read_bytes()
    position: int = struct.unpack_from("<H", data, 6)[0]
    frames: list[bytes] = []
    while position < len(data):
        size: int = struct.unpack_from("<I", data, position)[0]
        frames.append(data[position + 12 : position + 12 + size])
        position += 12 + size
    return frames


def reencode(
    entry: DatasetEntry, segment: str, index: VideoIndex, lo: int, hi: int, work_dir: Path, name: str, fps: float | None
) -> tuple[Int64[ndarray, "k"], list[bytes]]:
    """Low-res AV1 of the [lo, hi) window at about ``fps`` (every k-th source frame); decoding starts at the last keyframe."""
    first: int = int(np.searchsorted(index.times, lo))
    last: int = int(np.searchsorted(index.times, hi))
    if first == last:
        raise ValueError(f"{name}: no samples in the window")
    step: int = max(1, round(1e9 / float(np.median(np.diff(index.times[first:last]))) / fps)) if fps and last - first > 1 else 1
    keyframes: Int64[ndarray, "j"] = np.flatnonzero(index.keyframes[: first + 1])
    if not len(keyframes):
        raise ValueError(f"{name}: no keyframe at or before sample {first}")
    key: int = int(keyframes[-1])
    video: CatalogVideo = read_video_samples(entry, segment, index, key, last)
    if video.codec != "av1":
        raise ValueError(f"{name}: expected AV1 samples, found {video.codec}")
    source: Path = work_dir / f"{name}-src.mp4"
    low: Path = work_dir / f"{name}-low.ivf"
    source.write_bytes(wrap_mp4(video.samples, video.keyframes, MUX_FPS, video.codec))
    skip: int = first - key
    encode: subprocess.CompletedProcess[str] = subprocess.run(
        [str(resolve_ffmpeg()), "-hide_banner", "-loglevel", "error", "-y", "-c:v", "libdav1d", "-i", str(source),
         "-vf", f"select=gte(n\\,{skip})*not(mod(n-{skip}\\,{step})),scale={PREVIEW_VIDEO_SIZE[0]}:{PREVIEW_VIDEO_SIZE[1]}:flags=area", "-fps_mode", "passthrough",
         "-c:v", "libsvtav1", "-preset", "8", "-crf", "40", "-g", "30", "-pix_fmt", "yuv420p",
         "-svtav1-params", "pred-struct=1:enable-overlays=0:scd=0", "-f", "ivf", str(low)],
        capture_output=True,
        text=True,
        check=False,
    )
    if encode.returncode != 0:
        raise RuntimeError(f"{name}: ffmpeg exited with {encode.returncode}: {encode.stderr.strip()[-400:]}")
    frames: list[bytes] = read_ivf(low)
    kept: Int64[ndarray, "k"] = index.times[first:last][::step]
    if len(frames) != len(kept):
        raise RuntimeError(f"{name}: re-encode gave {len(frames)} frames for {len(kept)} kept source samples")
    return kept, frames


def video_chunks(path: str, times: Int64[ndarray, "n"], frames: list[bytes]) -> list[Chunk]:
    """The codec and the samples of one re-encoded video."""
    return [
        Chunk.from_columns(path, indexes=[], columns=rr.VideoStream.columns(codec=[rr.components.VideoCodec.AV1])),
        Chunk.from_columns(
            path,
            indexes=[rr.TimeColumn(PREVIEW_TIMELINE, duration=times.astype("timedelta64[ns]"))],
            columns=rr.VideoStream.columns(sample=[np.frombuffer(frame, np.uint8) for frame in frames]),
        ),
    ]


def build_segment_preview(entry: DatasetEntry, segment: str, target: Path, options: PreviewOptions) -> PreviewResult:
    """Write the preview layer of one segment to the new file ``target``.

    The file appears complete or not at all, and an existing target is refused, also one that appears during the build.

    Args:
        entry: Dataset that holds the segment; only read.
        segment: The segment; it is also the recording id of the layer file.
        target: New layer file.
        options: Window and rig selection.

    Returns:
        What was written.

    Raises:
        FileExistsError: If ``target`` exists.
    """
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to write {target}: it exists")
    store: LazyStore = entry.segment_store(segment)
    schema: Schema = store.schema()
    choice: tuple[int, list[int]] = exo_candidates(entry, segment, options)
    ego_rig: int = choice[0]
    phones: list[int] = choice[1]
    models: dict[int, CameraModel] = read_camera_models(store, list(dict.fromkeys([ego_rig, *phones])))
    exo_rig: int = next((rig for rig in phones if models[rig].is_posed), phones[0])
    exo_model: CameraModel = models[exo_rig]
    check_resolution(exo_model, segment, pinhole_path(exo_rig, 0), EXO_SIZE)
    check_resolution(models[ego_rig], segment, pinhole_path(ego_rig, 0), EGO_SIZE)
    exo_index: VideoIndex = read_video_index(entry, segment, video_path(exo_rig, 0))
    ego_index: VideoIndex = read_video_index(entry, segment, video_path(ego_rig, 0))
    middle: int = (int(exo_index.times[0]) + int(exo_index.times[-1])) // 2
    half: int = int(options.seconds * 1e9 / 2)
    lo: int = max(int(exo_index.times[0]), middle - half)
    hi: int = min(int(exo_index.times[-1]) + 1, middle + half)
    body: list[Chunk] = []
    for source, destination in ((SKELETON, PREVIEW_SKELETON), (SOMA_XLO, PREVIEW_SOMA_XLO)):
        for chunk in store.stream().filter(content=source).to_chunks():
            body += copy_window(chunk, destination, lo, hi, fps=options.fps)
    if not any(str(chunk.entity_path) == PREVIEW_SOMA_XLO for chunk in body) and SOMA_MESH in schema.entity_paths():
        raise RuntimeError(f"{segment}: SOMA mesh present but no soma_xlo rows in the window (register soma_xlo first)")
    center: Float32[ndarray, "3"] = body_center([chunk for chunk in body if str(chunk.entity_path) == PREVIEW_SKELETON])
    body.append(Chunk.from_columns(PREVIEW_CENTER, indexes=[], columns=rr.Transform3D.columns(translation=[center.tolist()])))
    moving: list[str] = sorted(
        {
            item.entity_path
            for item in catalog_components(schema)
            if item.archetype == "Transform3D" and not item.is_static and item.entity_path.startswith("/world/rig_") and "/pinhole" not in item.entity_path
        }
    )
    if moving:
        for chunk in store.stream().filter(content=moving, is_static=False).to_chunks():
            if has_component(chunk, "Transform3D:"):
                body += copy_window(chunk, str(chunk.entity_path), lo, hi, fps=options.fps, query_anchor=True)
    # Exo camera model: static pose + pinhole scaled to the preview video, so 3D content projects onto it.
    if exo_model.is_posed:  # an unposed exo phone gets no camera model: its video stays a plain image
        for chunk in exo_model.pose:
            body += copy_window(chunk, PREVIEW_EXO_CAM, lo, hi)
        for chunk in exo_model.pinhole:
            body += copy_window(chunk, PREVIEW_EXO_PINHOLE, lo, hi, pinhole_scale=PREVIEW_VIDEO_SIZE[0] / EXO_SIZE[0])
    for chunk in store.stream().filter(content=f"{pinhole_path(ego_rig, 0)}/skeleton").to_chunks():
        body += copy_window(chunk, f"{PREVIEW_EXO_PINHOLE}/ego/keypoints", lo, hi, fps=options.fps, points_scale=PREVIEW_VIDEO_SIZE[0] / EGO_SIZE[0])
    # A hidden sibling directory: the finished file reaches its name by one link, which refuses an existing target.
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{segment}.", dir=target.parent) as work_dir:
        exo_kept: tuple[Int64[ndarray, "k"], list[bytes]] = reencode(entry, segment, exo_index, lo, hi, Path(work_dir), "exo", options.fps)
        ego_kept: tuple[Int64[ndarray, "k"], list[bytes]] = reencode(entry, segment, ego_index, lo, hi, Path(work_dir), "ego", options.fps)
        chunks: list[Chunk] = [
            *body,
            *video_chunks(f"{PREVIEW_EXO_PINHOLE}/video", *exo_kept),
            *video_chunks(f"{PREVIEW_EXO_PINHOLE}/ego/video", *ego_kept),
            Chunk.from_columns(f"{PREVIEW_EXO_PINHOLE}/ego", indexes=[], columns=rr.Transform3D.columns(translation=[[0.0, PREVIEW_VIDEO_SIZE[1] + PREVIEW_VIDEO_GAP_PX, 0.0]])),
        ]
        with atomic_write(target, exclusive=True) as staged:
            LazyChunkStream.from_iter(chunks).collect(optimize=OptimizationProfile.OBJECT_STORE).write_rrd(staged, application_id=APPLICATION_ID, recording_id=segment)
    return PreviewResult(
        segment=segment,
        target=target,
        lo_ns=lo,
        hi_ns=hi,
        ego_rig=ego_rig,
        exo_rig=exo_rig,
        exo_camera_model=exo_model.is_posed,
        moving_rigs=tuple(sorted({str(chunk.entity_path) for chunk in body if str(chunk.entity_path).startswith("/world/rig_")})),
        exo_frames=len(exo_kept[1]),
        ego_frames=len(ego_kept[1]),
        file_bytes=target.stat().st_size,
    )
