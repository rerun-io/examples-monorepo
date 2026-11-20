"""FullCircle operator masks for a dual-fisheye base recording: a ``people_mask`` layer on both lenses.

Reads the lens videos, their Mei calibrations and the accelerometer of a dataforge Insta360 base recording from its
catalog segment (only the packets of the run window), runs
:class:`~posekit.models.fullcircle.FullCircleMasker` in FullCircle's two passes (detect the operator direction on
every tracked frameset, fill the gaps from the nearest frameset, then track and map back) and writes the mask layer
contract (track 3, v2): per lens ``<camera>/pinhole/people_mask``, a ``SegmentationImage`` (uint8, 1 = operator)
at the resolution it is computed at, with one static ``Transform3D(scale=[W_img / W_mask, H_img / H_mask, 1])``,
stamped with that lens's frame times, every ``stride``-th frameset; the dilation is already applied.

The rig frame of the masker is lens 0's camera frame (+Y down when the camera is held upright, so the cube faces and
the synthetic view are upright); lens 1 enters through its rotation relative to lens 0 (the baseline is ignored:
scene at infinity, as in a stitched panorama).
"""

import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from datafusion import col, lit
from einops import rearrange
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_calibration import CatalogCameraCalibration, read_camera_models
from simplecv.catalog_video import CatalogVideo, catalog_keyframes, catalog_video_from_table
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.mask_layer import MASK_ANNOTATION, MASK_NAME, MASKED_CLASS_ID
from posekit.models.fullcircle import (
    FullCircleConfig,
    FullCircleMasker,
    OperatorEvidence,
    OperatorMasks,
    SphereSource,
    fill_nearest,
    level_rotation,
    view_up,
)
from posekit.ops.sphere import MeiCamera


@dataclass
class FullCircleMasksConfig:
    """Write a FullCircle operator-mask layer for the two lenses of an Insta360 base recording."""

    rr_config: RerunTyroConfig
    """Output sink: ``--rr-config.save <layer.rrd> --rr-config.headless --rr-config.recording-id <segment>
    --rr-config.application-id dataforge``."""
    dataset: str | None = None
    """Catalog dataset of the base recording (dataforge ``insta360``), read with ``segment`` from ``catalog_url``."""
    segment: str | None = None
    """The base recording's segment in ``dataset``."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog holding ``dataset``."""
    rrd: Path | None = None
    """A base recording not on a catalog yet, instead of ``dataset``/``segment``: served by an in-process catalog."""
    cameras: tuple[str, str] = ("/world/rig_00/cam_00", "/world/rig_00/cam_01")
    """The two lens camera nodes (each has ``pinhole`` and ``pinhole/video``)."""
    stride: int = 5
    """Framesets between two tracked framesets (5 at 30 fps = 6 Hz, near FullCircle's 5 fps captures)."""
    start: int = 0
    """First frameset index."""
    end: int | None = None
    """Frameset index after the last (default: the end of the shorter lens stream)."""
    mask_downscale: int = 4
    """Mask size = lens size / this (480 px for 1920 px lenses, about the synthetic view's angular resolution)."""
    masker: FullCircleConfig = field(default_factory=FullCircleConfig)
    """FullCircle options (detector, tracker, view sizes, dilation in lens pixels)."""
    timeline: str = "video_time"
    """Timeline of the lens videos and of the masks."""
    accel: str = "/world/rig_00/imu_00/accel"
    """Accelerometer series (rig frame) that levels the faces and the synthetic view."""
    gravity_window_s: float = 0.5
    """Half-width of the window over which the specific force is averaged into gravity, seconds."""
    debug_rrd: Path | None = None
    """Optional evidence recording: faces with detections and the synthetic view with its mask, every ``debug_every``-th tracked frameset."""
    debug_every: int = 20
    """Tracked framesets between two evidence frames."""
    decode_chunk: int = 16
    """Tracked framesets decoded per NVDEC call, and detected per detector pass."""
    fps: int = 30
    """Nominal frame rate of the wrapping MP4 track (decoding only)."""


def read_lens_sources(
    dataset: DatasetEntry, segment: str, cameras: tuple[str, str], mask_downscale: int
) -> tuple[list[SphereSource], Float64[ndarray, "3 3"]]:
    """The two lens cameras of a base recording in lens 0's frame, and the rotation of rig (IMU) vectors into that frame.

    Raises:
        ValueError: If a lens lacks its calibration statics or is not a Mei camera.
    """
    models: dict[str, CatalogCameraCalibration] = read_camera_models(dataset, segment, cameras)
    # The lens Transform3D is ChildFromParent (lens from rig).
    cam_R_rigs: list[Float64[ndarray, "3 3"]] = [np.asarray(models[camera].pinhole.extrinsics.cam_R_world, dtype=np.float64) for camera in cameras]
    sources: list[SphereSource] = []
    for camera, cam_R_rig in zip(cameras, cam_R_rigs, strict=True):
        model: CatalogCameraCalibration = models[camera]
        if model.distortion is None:
            raise ValueError(f"{camera}: no distortion model, so not a Mei camera")
        k: Float64[ndarray, "3 3"] = np.asarray(model.pinhole.intrinsics.k_matrix, dtype=np.float64)
        width, height = int(model.pinhole.intrinsics.width), int(model.pinhole.intrinsics.height)
        lens: MeiCamera = MeiCamera.from_calibration(
            model.distortion.model, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), model.distortion.coefficients.tolist()
        )
        sources.append(SphereSource(lens, cam_R_rigs[0] @ cam_R_rig.T, (height, width), (height // mask_downscale, width // mask_downscale)))
    return sources, cam_R_rigs[0]


def read_lens_index(dataset: DatasetEntry, segment: str, entity: str, timeline: str) -> tuple[ndarray, list[bool]]:
    """One lens's frame stamps and keyframe flags, without its packets.

    Raises:
        ValueError: If the entity has no frames with strictly increasing stamps.
    """
    sample: str = f"{entity}:VideoStream:sample"
    keyframe: str = f"{entity}:VideoStream:is_keyframe"
    view = dataset.filter_segments(segment).filter_contents([entity]).reader(index=timeline)
    table: pa.Table = view.filter(col(sample).is_not_null()).select(timeline, keyframe).to_arrow_table()
    times: ndarray = table[timeline].combine_chunks().to_numpy(zero_copy_only=False)
    if not len(times) or not np.all(np.diff(times.view(np.int64)) > 0):
        raise ValueError(f"{segment} {entity}: expected frames with strictly increasing timestamps")
    return times, catalog_keyframes(table[keyframe])


def read_lens_window(dataset: DatasetEntry, segment: str, entity: str, timeline: str, times: ndarray) -> CatalogVideo:
    """One lens's packets at ``times``, a run of its index read with one range query.

    Raises:
        ValueError: If the catalog returns other frames than ``times``.
    """
    view = dataset.filter_segments(segment).filter_contents([entity]).reader(index=timeline)
    in_window = (col(timeline) >= lit(pa.scalar(times[0]))) & (col(timeline) <= lit(pa.scalar(times[-1])))
    columns: list[str] = [f"{entity}:VideoStream:{component}" for component in ("sample", "is_keyframe", "codec")]
    video: CatalogVideo = catalog_video_from_table(
        view.filter(in_window).select(timeline, *columns).to_arrow_table(), entity, timeline, f"{segment} {entity}"
    )
    if not np.array_equal(video.times, times):
        raise ValueError(f"{segment} {entity}: the catalog returned {len(video.times)} frames for {len(times)} indexed ones")
    return video


def read_level_rotations(
    dataset: DatasetEntry,
    segment: str,
    accel: str,
    timeline: str,
    stamps_ns: Int64[ndarray, " n"],
    cam0_R_rig: Float64[ndarray, "3 3"],
    window_s: float,
) -> list[Float64[ndarray, "3 3"]]:
    """Per stamp, ``rig_R_level`` in lens 0's frame: gravity is minus the mean specific force within ``window_s`` of it.

    Raises:
        ValueError: If the recording has no accelerometer series, or a window holds no sample.
    """
    scalars: str = f"{accel}:Scalars:scalars"
    view = dataset.filter_segments(segment).filter_contents([accel]).reader(index=timeline)
    table: pa.Table = view.filter(col(scalars).is_not_null()).select(timeline, scalars).to_arrow_table()
    if not table.num_rows:
        raise ValueError(f"{accel}: no accelerometer series to level the views with")
    times: Int64[ndarray, " m"] = np.asarray(table[timeline].combine_chunks().to_numpy(zero_copy_only=False)).view(np.int64)
    forces: Float64[ndarray, "m 3"] = np.asarray(table[scalars].to_pylist(), dtype=np.float64)
    order: Int64[ndarray, " m"] = np.argsort(times)
    times, forces = times[order], forces[order]
    cumulative: Float64[ndarray, "m1 3"] = np.concatenate([np.zeros((1, 3)), np.cumsum(forces, axis=0)])
    half: int = int(window_s * 1e9)
    lo: Int64[ndarray, " n"] = np.searchsorted(times, stamps_ns - half)
    hi: Int64[ndarray, " n"] = np.searchsorted(times, stamps_ns + half)
    if np.any(hi <= lo):
        raise ValueError(f"{accel}: a frame has no accelerometer sample within {window_s} s")
    down_rig: Float64[ndarray, "n 3"] = -(cumulative[hi] - cumulative[lo])
    return [level_rotation(cam0_R_rig @ down) for down in down_rig]


def main(config: FullCircleMasksConfig) -> None:
    """Write the operator-mask layer from the catalog, or from ``rrd`` served in-process.

    Raises:
        ValueError: Unless exactly one of ``rrd`` and ``dataset`` + ``segment`` names the base.
    """
    if (config.rrd is None) == (config.dataset is None or config.segment is None):
        raise ValueError("name the base recording with --dataset and --segment, or with --rrd")
    if config.rrd is None:
        assert config.dataset is not None and config.segment is not None
        write_masks(config, CatalogClient(config.catalog_url).get_dataset(name=config.dataset), config.segment)
        return
    with rr.server.Server(datasets={"base": [str(config.rrd)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("base")
        segments: list[str] = list(dataset.segment_ids())
        if len(segments) != 1:
            raise ValueError(f"{config.rrd} holds {len(segments)} recordings; expected one")
        write_masks(config, dataset, segments[0])


def write_masks(config: FullCircleMasksConfig, dataset: DatasetEntry, segment_id: str) -> None:
    """Write the operator-mask layer of one segment and print per-step timings."""
    process_start: float = time.perf_counter()
    if config.rr_config.recording_id != segment_id:
        raise ValueError(f"pass --rr-config.recording-id {segment_id} so the mask layer joins the base recording")
    sources, cam0_R_rig = read_lens_sources(dataset, segment_id, config.cameras, config.mask_downscale)
    print(", ".join(f"lens {s.image_hw} -> mask {s.mask_hw}" for s in sources))
    device: torch.device = torch.device(config.masker.device)
    entities: list[str] = [f"{camera}/pinhole/video" for camera in config.cameras]
    indexes: list[tuple[ndarray, list[bool]]] = [read_lens_index(dataset, segment_id, entity, config.timeline) for entity in entities]
    times: list[ndarray] = [index[0] for index in indexes]
    n_framesets: int = min(len(t) for t in times)
    end: int = n_framesets if config.end is None else min(config.end, n_framesets)
    tracked: list[int] = list(range(config.start, end, config.stride))
    # Each decoder holds the packets from the keyframe at or before ``start`` to ``end``.
    firsts: list[int] = [max(i for i in range(config.start + 1) if keyframes[i]) for _, keyframes in indexes]
    decoders: list[VideoDecoder] = []
    for entity, lens_times, first in zip(entities, times, firsts, strict=True):
        window: CatalogVideo = read_lens_window(dataset, segment_id, entity, config.timeline, lens_times[first:end])
        decoders.append(
            VideoDecoder(
                wrap_mp4(window.samples, window.keyframes, config.fps, codec=window.codec),
                device=str(device),
                seek_mode="exact",
                num_ffmpeg_threads=0,
            )
        )
    read_s: float = time.perf_counter() - process_start
    print(
        f"segment {segment_id}: {n_framesets} framesets, tracking {len(tracked)} (stride {config.stride}) over {config.start}..{end - 1}; read {read_s:.1f} s"
    )

    levels: list[Float64[ndarray, "3 3"]] = read_level_rotations(
        dataset,
        segment_id,
        config.accel,
        config.timeline,
        np.asarray([times[0][i] for i in tracked]).view(np.int64),
        cam0_R_rig,
        config.gravity_window_s,
    )
    load_start: float = time.perf_counter()
    masker: FullCircleMasker = FullCircleMasker(config.masker, sources)
    print(f"model load + static tables {time.perf_counter() - load_start:.1f} s")
    debug: rr.RecordingStream | None = None
    if config.debug_rrd is not None:
        debug = rr.RecordingStream("fullcircle_debug", recording_id=f"{segment_id}-fullcircle-debug")
        debug.save(config.debug_rrd)

    decode_wait_s: float = 0.0

    # One thread per lens: an NVDEC decoder fails on a thread other than the one it first decoded on, and the two lenses
    # decode in parallel (1.7x the frames per second of one thread for both on the 5090).
    decode_threads: list[ThreadPoolExecutor] = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"decode-{lens}") for lens in range(len(decoders))]

    def decode(lens: int, chunk: list[int]) -> UInt8[Tensor, "n h w 3"]:
        frames: UInt8[Tensor, "n h w 3"] = rearrange(decoders[lens].get_frames_at([i - firsts[lens] for i in chunk]).data, "n c h w -> n h w c")
        torch.cuda.synchronize(device)
        return frames

    def decoded_chunks() -> Iterator[tuple[int, list[int], list[list[UInt8[Tensor, "h w 3"]]]]]:
        """The tracked framesets ``decode_chunk`` at a time: the position of the first, their indices, and per frameset the
        lens frames. The next chunk decodes on the decode threads while the caller computes on this one."""
        nonlocal decode_wait_s
        chunks: list[list[int]] = [tracked[begin : begin + config.decode_chunk] for begin in range(0, len(tracked), config.decode_chunk)]

        def submit(chunk: list[int]) -> list[Future[Tensor]]:
            return [thread.submit(decode, lens, chunk) for lens, thread in enumerate(decode_threads)]

        pending: list[Future[Tensor]] = submit(chunks[0]) if chunks else []
        for position, chunk in enumerate(chunks):
            tick: float = time.perf_counter()
            batches: list[Tensor] = [future.result() for future in pending]
            decode_wait_s += time.perf_counter() - tick
            pending = submit(chunks[position + 1]) if position + 1 < len(chunks) else []
            frames: list[list[Tensor]] = [[batch[column] for batch in batches] for column in range(len(chunk))]
            yield position * config.decode_chunk, chunk, frames

    detect_start: float = time.perf_counter()
    directions: list[Float64[ndarray, "3"] | None] = []
    for begin, chunk, frames in decoded_chunks():
        batch: list[OperatorEvidence] = masker.detect(frames, levels[begin : begin + len(chunk)])
        for position, (index, evidence) in enumerate(zip(chunk, batch, strict=True), start=begin):
            directions.append(evidence.direction)
            if debug is not None and position % config.debug_every == 0:
                debug.set_time("frameset", sequence=index)
                for face in range(6):
                    debug.log(f"faces/face_{face}", rr.Image(evidence.face_images[face].cpu().numpy()).compress(90))
                    keep: ndarray = evidence.faces == face
                    debug.log(
                        f"faces/face_{face}/persons", rr.Boxes2D(array=evidence.boxes[keep], array_format=rr.Box2DFormat.XYXY, colors=(255, 200, 0))
                    )
    found: int = sum(d is not None for d in directions)
    detect_wall: float = time.perf_counter() - detect_start
    print(f"pass 1 (detect): {found} of {len(tracked)} framesets with a person, {detect_wall:.1f} s")
    filled: list[Float64[ndarray, "3"]] = fill_nearest(directions)
    up_level: Float64[ndarray, "3"] = view_up([level.T @ d for level, d in zip(levels, filled, strict=True)])
    print(f"view up hint (level frame) {up_level.tolist()}")

    targets: list[str] = [f"{camera}/pinhole/{MASK_NAME}" for camera in config.cameras]
    for target, source in zip(targets, sources, strict=True):
        rr.log(target, MASK_ANNOTATION, static=True)
        rr.log(target, rr.Transform3D(scale=[*source.mask_scale, 1.0]), static=True)
    if debug is not None:
        debug.log("synthetic/mask", rr.AnnotationContext([(0, "keep", (0, 0, 0, 0)), (MASKED_CLASS_ID, "operator", (255, 0, 0, 160))]), static=True)
    track_start: float = time.perf_counter()
    for begin, chunk, frames in decoded_chunks():
        chunk_masks: list[list[Tensor]] = [[] for _ in targets]
        for position, (index, lens_frames) in enumerate(zip(chunk, frames, strict=True), start=begin):
            result: OperatorMasks = masker.track(lens_frames, filled[position], levels[position] @ up_level)
            for lens_masks, mask in zip(chunk_masks, result.source_masks, strict=True):
                lens_masks.append(mask)
            if debug is not None and position % config.debug_every == 0:
                debug.set_time("frameset", sequence=index)
                debug.log("synthetic", rr.Image(result.synthetic_rgb.cpu().numpy()).compress(90))
                debug.log("synthetic/mask", rr.SegmentationImage(result.synthetic_mask.to(torch.uint8).cpu().numpy()))
                debug.log("operator_direction", rr.Arrows3D(vectors=[filled[position]], colors=(255, 0, 0)))
        # One host copy per lens and chunk.
        for lens, (target, lens_masks) in enumerate(zip(targets, chunk_masks, strict=True)):
            host: UInt8[ndarray, "n mh mw"] = (torch.stack(lens_masks).to(torch.uint8) * MASKED_CLASS_ID).cpu().numpy()
            for index, mask in zip(chunk, host, strict=True):
                rr.set_time(config.timeline, duration=np.timedelta64(int(times[lens][index].view(np.int64)), "ns"))
                rr.log(target, rr.SegmentationImage(mask))
        elapsed: float = time.perf_counter() - track_start
        if begin % (config.decode_chunk * 25) == 0:
            done: int = begin + len(chunk)
            print(f"frameset {chunk[-1]}: {done} tracked, {done / elapsed:.1f} framesets/s")
    for thread in decode_threads:
        thread.shutdown()
    recording: rr.RecordingStream | None = rr.get_data_recording()
    assert recording is not None
    recording.flush()
    if debug is not None:
        debug.flush()
    track_wall: float = time.perf_counter() - track_start
    total: float = time.perf_counter() - process_start
    steps: str = "".join(f"{k} {v:.1f} s, " for k, v in masker.seconds.items()) if config.masker.step_timing else "(--masker.step-timing), "
    print(
        f"pass 2 (track + map back): {len(tracked)} framesets in {track_wall:.1f} s; waiting for decode (both passes) {decode_wait_s:.1f} s; steps: {steps}"
        f"{masker.prompts} tracker prompts; peak GPU allocated {torch.cuda.max_memory_allocated() / 1e9:.1f} GB; total {total:.1f} s for {len(tracked) * 2} lens masks"
    )
