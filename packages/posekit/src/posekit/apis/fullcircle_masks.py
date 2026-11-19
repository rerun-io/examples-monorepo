"""FullCircle operator masks for a dual-fisheye base recording: a ``people_mask`` layer on both lenses.

Reads the lens videos and their Mei calibrations from a dataforge Insta360 base recording, runs
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.chunk as rrc
import torch
from einops import rearrange
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.catalog_video import CatalogVideo, catalog_codec, catalog_keyframes, packet_views
from simplecv.catalog_video_codec import CatalogCodecName, wrap_mp4
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from posekit.models.fullcircle import FullCircleConfig, FullCircleMasker, OperatorEvidence, OperatorMasks, SphereSource, fill_nearest, view_up
from posekit.ops.sphere import MeiCamera

OPERATOR_CLASS_ID: int = 1
"""Mask value of the operator (nonzero = drop features there)."""


@dataclass
class FullCircleMasksConfig:
    """Write a FullCircle operator-mask layer for the two lenses of an Insta360 base recording."""

    rrd: Path
    """Base recording (dataforge ``insta360``) with ``/world/rig_00/cam_00`` and ``cam_01``."""
    rr_config: RerunTyroConfig
    """Output sink: ``--rr-config.save <layer.rrd> --rr-config.headless --rr-config.recording-id <segment>
    --rr-config.application-id dataforge``."""
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
    debug_rrd: Path | None = None
    """Optional evidence recording: faces with detections and the synthetic view with its mask, every ``debug_every``-th tracked frameset."""
    debug_every: int = 20
    """Tracked framesets between two evidence frames."""
    decode_chunk: int = 16
    """Tracked framesets decoded per NVDEC call."""
    fps: int = 30
    """Nominal frame rate of the wrapping MP4 track (decoding only)."""


def read_lens_sources(store: rrc.LazyStore, cameras: tuple[str, str], mask_downscale: int) -> list[SphereSource]:
    """The two lens cameras of a base recording, in lens 0's frame.

    Raises:
        ValueError: If a lens lacks its calibration statics or is not a Mei camera.
    """
    wanted: dict[str, list[str]] = {camera: [f"{camera}:Transform3D:mat3x3"] for camera in cameras} | {
        f"{camera}/pinhole": [
            f"{camera}/pinhole:simplecv.components.DistortionModel",
            f"{camera}/pinhole:simplecv.components.DistortionCoefficients",
            f"{camera}/pinhole:Pinhole:image_from_camera",
            f"{camera}/pinhole:Pinhole:resolution",
        ]
        for camera in cameras
    }
    needed: set[str] = {key for keys in wanted.values() for key in keys}
    statics: dict[str, list] = {}
    for chunk in store.stream().filter(content=list(wanted)).to_chunks():
        if chunk.is_static:
            batch = chunk.to_record_batch()
            for name in batch.schema.names:
                if f"{chunk.entity_path}:{name}" in needed:
                    statics[f"{chunk.entity_path}:{name}"] = batch.column(name).to_pylist()[0]
    missing: list[str] = sorted(needed - statics.keys())
    if missing:
        raise ValueError(f"missing lens statics {missing}")
    cam_R_rig: list[Float64[ndarray, "3 3"]] = []
    lenses: list[tuple[MeiCamera, tuple[int, int]]] = []
    for camera in cameras:
        pinhole: str = f"{camera}/pinhole"
        # Rerun stores 3x3 matrices column-major; the lens Transform3D is ChildFromParent (lens from rig).
        cam_R_rig.append(np.asarray(statics[f"{camera}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T)
        k: Float64[ndarray, "3 3"] = np.asarray(statics[f"{pinhole}:Pinhole:image_from_camera"][0], dtype=np.float64).reshape(3, 3).T
        width, height = (int(v) for v in statics[f"{pinhole}:Pinhole:resolution"][0])
        model: str = str(statics[f"{pinhole}:simplecv.components.DistortionModel"][0])
        coefficients: list[float] = [float(c) for c in statics[f"{pinhole}:simplecv.components.DistortionCoefficients"][0]]
        lenses.append(
            (MeiCamera.from_calibration(model, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), coefficients), (height, width))
        )
    sources: list[SphereSource] = []
    for index, (lens, image_hw) in enumerate(lenses):
        cam0_R_cam: Float64[ndarray, "3 3"] = cam_R_rig[0] @ cam_R_rig[index].T
        sources.append(SphereSource(lens, cam0_R_cam, image_hw, (image_hw[0] // mask_downscale, image_hw[1] // mask_downscale)))
    return sources


def read_lens_video(store: rrc.LazyStore, entity: str, timeline: str) -> CatalogVideo:
    """One lens's encoded packets from the recording file, reading only that entity's chunks.

    Raises:
        ValueError: If the entity has no frames with strictly increasing stamps.
    """
    chunks: list = store.stream().filter(content=[entity]).to_chunks()
    if not chunks:
        raise ValueError(f"{entity}: no video in the recording")
    table: pa.Table = rrc.ChunkStore.from_chunks(chunks).reader(index=timeline).to_arrow_table()
    sample: str = f"{entity}:VideoStream:sample"
    if table[sample].null_count:
        table = table.filter(table[sample].is_valid())
    times: ndarray = table[timeline].combine_chunks().to_numpy(zero_copy_only=False)
    if not len(times) or not np.all(np.diff(times.view(np.int64)) > 0):
        raise ValueError(f"{entity}: expected frames with strictly increasing timestamps")
    codec: CatalogCodecName = catalog_codec(table[f"{entity}:VideoStream:codec"], entity)
    return CatalogVideo(times, packet_views(table[sample]), catalog_keyframes(table[f"{entity}:VideoStream:is_keyframe"]), codec)


def main(config: FullCircleMasksConfig) -> None:
    """Write the operator-mask layer and print per-step timings."""
    process_start: float = time.perf_counter()
    reader: rrc.RrdReader = rrc.RrdReader(config.rrd)
    recording_ids: list[str] = [entry.recording_id for entry in reader.recordings()]
    if len(recording_ids) != 1:
        raise ValueError(f"{config.rrd} holds {len(recording_ids)} recordings; expected one")
    segment_id: str = recording_ids[0]
    if config.rr_config.recording_id != segment_id:
        raise ValueError(f"pass --rr-config.recording-id {segment_id} so the mask layer joins the base recording")
    store: rrc.LazyStore = reader.store()
    sources: list[SphereSource] = read_lens_sources(store, config.cameras, config.mask_downscale)
    print(", ".join(f"lens {s.image_hw} -> mask {s.mask_hw}" for s in sources))
    device: torch.device = torch.device(config.masker.device)
    videos: list[CatalogVideo] = [read_lens_video(store, f"{camera}/pinhole/video", config.timeline) for camera in config.cameras]
    decoders: list[VideoDecoder] = [
        VideoDecoder(wrap_mp4(v.samples, v.keyframes, config.fps, codec=v.codec), device=str(device), seek_mode="exact", num_ffmpeg_threads=0)
        for v in videos
    ]
    times: list[ndarray] = [v.times for v in videos]
    n_framesets: int = min(len(t) for t in times)
    end: int = n_framesets if config.end is None else min(config.end, n_framesets)
    tracked: list[int] = list(range(config.start, end, config.stride))
    read_s: float = time.perf_counter() - process_start
    print(
        f"segment {segment_id}: {n_framesets} framesets, tracking {len(tracked)} (stride {config.stride}) over {config.start}..{end - 1}; read {read_s:.1f} s"
    )

    load_start: float = time.perf_counter()
    masker: FullCircleMasker = FullCircleMasker(config.masker, sources)
    print(f"model load + static tables {time.perf_counter() - load_start:.1f} s")
    debug: rr.RecordingStream | None = None
    if config.debug_rrd is not None:
        debug = rr.RecordingStream("fullcircle_debug", recording_id=f"{segment_id}-fullcircle-debug")
        debug.save(config.debug_rrd)

    def framesets(indices: list[int]) -> list[list[UInt8[Tensor, "h w 3"]]]:
        batches: list[Tensor] = [rearrange(decoder.get_frames_at(indices).data, "n c h w -> n h w c") for decoder in decoders]
        return [[batch[column] for batch in batches] for column in range(len(indices))]

    decode_s: float = 0.0
    detect_start: float = time.perf_counter()
    directions: list[Float64[ndarray, "3"] | None] = []
    for begin in range(0, len(tracked), config.decode_chunk):
        chunk: list[int] = tracked[begin : begin + config.decode_chunk]
        tick: float = time.perf_counter()
        frames: list[list[Tensor]] = framesets(chunk)
        torch.cuda.synchronize(device)
        decode_s += time.perf_counter() - tick
        for offset, (index, lens_frames) in enumerate(zip(chunk, frames, strict=True)):
            evidence: OperatorEvidence = masker.detect(lens_frames)
            directions.append(evidence.direction)
            if debug is not None and (begin + offset) % config.debug_every == 0:
                debug.set_time("frameset", sequence=index)
                for face in range(6):
                    debug.log(f"faces/face_{face}", rr.Image(evidence.face_images[face].cpu().numpy()).compress(90))
                    keep: ndarray = evidence.faces.numpy() == face
                    debug.log(
                        f"faces/face_{face}/persons", rr.Boxes2D(array=evidence.boxes[keep], array_format=rr.Box2DFormat.XYXY, colors=(255, 200, 0))
                    )
    found: int = sum(d is not None for d in directions)
    detect_wall: float = time.perf_counter() - detect_start
    print(f"pass 1 (detect): {found} of {len(tracked)} framesets with a person, {detect_wall:.1f} s")
    filled: list[Float64[ndarray, "3"]] = fill_nearest(directions)
    up: Float64[ndarray, "3"] = view_up(filled)
    print(f"view up hint {up.tolist()}")

    targets: list[str] = [f"{camera}/pinhole/people_mask" for camera in config.cameras]
    for target, source in zip(targets, sources, strict=True):
        rr.log(target, rr.AnnotationContext([(0, "keep", (0, 0, 0, 0)), (OPERATOR_CLASS_ID, "operator", (255, 0, 0, 160))]), static=True)
        rr.log(target, rr.Transform3D(scale=[source.image_hw[1] / source.mask_hw[1], source.image_hw[0] / source.mask_hw[0], 1.0]), static=True)
    if debug is not None:
        debug.log("synthetic/mask", rr.AnnotationContext([(0, "keep", (0, 0, 0, 0)), (OPERATOR_CLASS_ID, "operator", (255, 0, 0, 160))]), static=True)
    track_start: float = time.perf_counter()
    position: int = 0
    for begin in range(0, len(tracked), config.decode_chunk):
        chunk = tracked[begin : begin + config.decode_chunk]
        tick = time.perf_counter()
        frames = framesets(chunk)
        torch.cuda.synchronize(device)
        decode_s += time.perf_counter() - tick
        for index, lens_frames in zip(chunk, frames, strict=True):
            result: OperatorMasks = masker.track(lens_frames, filled[position], up)
            for lens, target in enumerate(targets):
                rr.set_time(config.timeline, duration=np.timedelta64(int(times[lens][index].view(np.int64)), "ns"))
                rr.log(target, rr.SegmentationImage(result.source_masks[lens].to(torch.uint8).cpu().numpy() * OPERATOR_CLASS_ID))
            if debug is not None and position % config.debug_every == 0:
                debug.set_time("frameset", sequence=index)
                debug.log("synthetic", rr.Image(result.synthetic_rgb.cpu().numpy()).compress(90))
                debug.log("synthetic/mask", rr.SegmentationImage(result.synthetic_mask.to(torch.uint8).cpu().numpy()))
                debug.log("operator_direction", rr.Arrows3D(vectors=[filled[position]], colors=(255, 0, 0)))
            position += 1
        elapsed: float = time.perf_counter() - track_start
        if begin % (config.decode_chunk * 25) == 0:
            print(f"frameset {chunk[-1]}: {position} tracked, {position / elapsed:.1f} framesets/s")
    recording: rr.RecordingStream | None = rr.get_data_recording()
    assert recording is not None
    recording.flush()
    if debug is not None:
        debug.flush()
    track_wall: float = time.perf_counter() - track_start
    total: float = time.perf_counter() - process_start
    steps: str = ", ".join(f"{k} {v:.1f} s" for k, v in masker.seconds.items())
    print(
        f"pass 2 (track + map back): {len(tracked)} framesets in {track_wall:.1f} s; decode (both passes) {decode_s:.1f} s; steps: {steps}; "
        f"{masker.prompts} tracker prompts; peak GPU allocated {torch.cuda.max_memory_allocated() / 1e9:.1f} GB; total {total:.1f} s for {len(tracked) * 2} lens masks"
    )
