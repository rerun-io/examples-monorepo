"""The tracker over one catalog segment: read it, decode its cameras in order on NVDEC, track every frame, keep the result.

A segment is read with three catalog queries (statics, the hand timeline, the video packets; ``handtrack.data.catalog``).
Every camera gets its own torchcodec NVDEC decoder; frames are decoded in chunks of consecutive timeline rows, letterboxed
to the net frame on the GPU and fed to the tracker one frame at a time. The same decoded chunk can also go through DetNet
on every camera at once (the DetNet-alone evaluation), so one decode serves both.
"""

import hashlib
import io
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from jaxtyping import Bool, Float32, Int8, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from torch import Tensor, nn

from handtrack.data.catalog import (
    TIMELINE,
    HandTimeline,
    SegmentInfo,
    camera_angles,
    is_show3d,
    layout_for,
    read_hand_timeline,
    read_rig,
    read_statics,
)
from handtrack.data.segment_labels import SegmentLabels, segment_labels
from handtrack.data.stream import FrameDecoder, open_nvdec_decoder
from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH, Letterbox
from handtrack.hand.pose import HandPose
from handtrack.models.detnet import Detections
from handtrack.results import BoxSource, SegmentTrack, TrackMetadata
from handtrack.tracker import DetNetDetector, FrameResult, Tracker
from handtrack.train.checkpoint import read_disk

DECODE_CHUNK: int = 32
"""Timeline rows per decode call and camera."""


@dataclass(frozen=True, slots=True)
class SegmentData:
    """One segment's cameras, ground truth (every timeline row) and compressed video."""

    info: SegmentInfo
    rig: CameraRig
    letterboxes: tuple[Letterbox, ...]
    timeline: HandTimeline
    labels: SegmentLabels
    """Ground truth for every timeline row, in row order."""
    videos: tuple[CatalogVideo, ...]
    """One per camera, in ``rig.names`` order."""
    camera_angles: tuple[float, ...] = ()
    """Native camera roll for UmeTrack perspective crops (``catalog.camera_angles``). SHOW3D and HOT3D: 0 degrees (UmeTrack gets the native image and its own camera model);
    measured 2026-09-29 on 600 frames of show3d__HLU829: -90 -> L 40 / R 128 mm, 0 -> 28 / 46 mm, +90 -> 64 / 46 mm; the hands prefer
    mirrored rolls (-45 -> 25 / 42, +45 -> 37 / 35), an open convention question. HT_SHOW3D_ANGLE overrides for experiments."""

    @property
    def frames(self) -> int:
        return len(self.timeline.video_time_ns)


def read_segment(entry: DatasetEntry, info: SegmentInfo) -> SegmentData:
    """Read a segment from the catalog and label all of its timeline rows."""
    statics: pa.Table = read_statics(entry, info)
    rig_letterboxes: tuple[CameraRig, tuple[Letterbox, ...]] = read_rig(statics, info)
    timeline: HandTimeline = read_hand_timeline(entry, info, statics)
    cameras: tuple[str, ...] = layout_for(info.dataset).cameras
    videos: tuple[CatalogVideo, ...] = read_catalog_videos(entry, info.segment_id, [f"{camera}/pinhole/video" for camera in cameras], TIMELINE)
    rows: Int64[ndarray, "f"] = np.arange(len(timeline.video_time_ns), dtype=np.int64)
    labels: SegmentLabels = segment_labels(timeline, rig_letterboxes[0], rig_letterboxes[1], rows, layout_for(info.dataset).pose_gated)
    angles: tuple[float, ...] = (tuple(float(os.environ.get('HT_SHOW3D_ANGLE', '0')) for _ in cameras) if is_show3d(info.dataset)
                                 else camera_angles(statics, info))
    return SegmentData(info=info, rig=rig_letterboxes[0], letterboxes=rig_letterboxes[1], timeline=timeline, labels=labels, videos=videos, camera_angles=angles)


@dataclass(frozen=True, slots=True)
class FrameImages:
    """One decoded chunk, sharing the original camera frames with the net-frame images."""

    net: UInt8[Tensor, "m c 480 640"]
    """Letterboxed images for DetNet/KeyNet."""
    native: tuple[UInt8[Tensor, "m h w"], ...]
    """Original camera pixels for perspective crop sampling."""


def net_frames(data: SegmentData, frames: int, device: torch.device) -> Iterator[UInt8[Tensor, "m c 480 640"]]:
    """Net images only, preserving the reference ladder's decode API."""
    for chunk in decoded_frames(data, frames, device):
        yield chunk.net


def decoded_frames(data: SegmentData, frames: int, device: torch.device) -> Iterator[FrameImages]:
    """The first ``frames`` timeline rows as letterboxed images of every camera, in chunks of ``DECODE_CHUNK`` rows.

    A camera without a video frame at a row's time gives a black image there.
    """
    times: Int64[ndarray, "f"] = data.timeline.video_time_ns[:frames]
    decoders: list[FrameDecoder] = []
    positions: list[Int64[ndarray, "f"]] = []
    matched: list[Bool[ndarray, "f"]] = []
    for video in data.videos:
        decoders.append(open_nvdec_decoder(video, data.info.fps, device))
        position: Int64[ndarray, "f"] = np.clip(np.searchsorted(video.t_ns, times), 0, len(video.t_ns) - 1)
        positions.append(position)
        matched.append(video.t_ns[position] == times)
    for begin in range(0, frames, DECODE_CHUNK):
        end: int = min(begin + DECODE_CHUNK, frames)
        per_camera: list[UInt8[Tensor, "m 480 640"]] = []
        native: list[UInt8[Tensor, "m h w"]] = []
        for camera, decoder in enumerate(decoders):
            letterbox: Letterbox = data.letterboxes[camera]
            images: UInt8[Tensor, "m h w"] = torch.zeros((end - begin, letterbox.source_height, letterbox.source_width), dtype=torch.uint8, device=device)
            keep: Int64[ndarray, "k"] = np.flatnonzero(matched[camera][begin:end])
            if len(keep):
                decoded: UInt8[Tensor, "k h w"] = decoder.get_frames_at(positions[camera][begin + keep].tolist()).data[:, 0]
                images[torch.from_numpy(keep).to(device)] = decoded
            native.append(images)
            per_camera.append(letterbox.apply(images))
        yield FrameImages(torch.stack(per_camera, dim=1), tuple(native))


@dataclass(frozen=True, slots=True)
class DetNetAlone:
    """DetNet on every frame and camera, net frame, on the CPU."""

    circle: Float32[Tensor, "f c 2 3"]
    probability: Float32[Tensor, "f c 2"]


@dataclass(frozen=True, slots=True)
class TrackerRun:
    """The tracker's frames, the optional DetNet-alone pass over the same decode, and the stage timings."""

    frames: list[FrameResult]
    detnet_alone: DetNetAlone | None
    timings_s: dict[str, float]


def run_tracker(
    data: SegmentData, tracker: Tracker | None, frames: int, device: torch.device, detnet_alone: DetNetDetector | None = None
) -> TrackerRun:
    """Track rows 0..frames-1 in order; with ``detnet_alone``, also run DetNet on every camera of every decoded frame.

    Without a tracker only the DetNet-alone pass runs (a quick DetNet evaluation).
    """
    start: float = time.perf_counter()
    results: list[FrameResult] = []
    circles: list[Float32[Tensor, "m c 2 3"]] = []
    probabilities: list[Float32[Tensor, "m c 2"]] = []
    decode_s: float = 0.0
    detnet_alone_s: float = 0.0
    cameras: int = len(data.letterboxes)
    chunks: Iterator[FrameImages] = decoded_frames(data, frames, device)
    begin: int = 0
    while True:
        decode_start: float = time.perf_counter()
        decoded: FrameImages | None = next(chunks, None)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_s += time.perf_counter() - decode_start
        if decoded is None:
            break
        chunk: UInt8[Tensor, "m c 480 640"] = decoded.net
        m: int = chunk.shape[0]
        if detnet_alone is not None:
            alone_start: float = time.perf_counter()
            detections: Detections = detnet_alone(chunk.reshape(m * cameras, NET_HEIGHT, NET_WIDTH), begin, torch.arange(cameras).repeat(m))
            circles.append(detections.circle.reshape(m, cameras, 2, 3))
            probabilities.append(detections.probability.reshape(m, cameras, 2))
            detnet_alone_s += time.perf_counter() - alone_start
        if tracker is not None:
            for offset in range(m):
                results.append(tracker.step(begin + offset, chunk[offset], data.timeline.world_from_rig[begin + offset],
                                            tuple(images[offset] for images in decoded.native)))
        begin += m
    stages: dict[str, float] = {} if tracker is None else tracker.timings_s
    timings: dict[str, float] = {"decode": decode_s, "detnet_alone": detnet_alone_s, **stages, "total": time.perf_counter() - start}
    alone: DetNetAlone | None = DetNetAlone(torch.cat(circles), torch.cat(probabilities)) if detnet_alone is not None else None
    return TrackerRun(frames=results, detnet_alone=alone, timings_s=timings)


def camera_boxes(letterboxes: tuple[Letterbox, ...], circle: Float32[Tensor, "f c 2 3"]) -> Float32[Tensor, "f c 2 4"]:
    """Net-frame hand circles to (x0, y0, x1, y1) squares in each camera's own pixels."""
    boxes: list[Float32[Tensor, "f 2 4"]] = []
    for camera, letterbox in enumerate(letterboxes):
        centre: Float32[Tensor, "f 2 2"] = letterbox.from_net(circle[:, camera, :, :2])
        radius: Float32[Tensor, "f 2 1"] = circle[:, camera, :, 2:] / letterbox.scale
        boxes.append(torch.cat([centre - radius, centre + radius], dim=-1))
    return torch.stack(boxes, dim=1)


def net_circles(letterboxes: tuple[Letterbox, ...], box: Float32[Tensor, "f c 2 4"]) -> Float32[Tensor, "f c 2 3"]:
    """The inverse of ``camera_boxes``: camera-pixel squares back to net-frame circles."""
    circles: list[Float32[Tensor, "f 2 3"]] = []
    for camera, letterbox in enumerate(letterboxes):
        centre: Float32[Tensor, "f 2 2"] = letterbox.to_net((box[:, camera, :, :2] + box[:, camera, :, 2:]) * 0.5)
        radius: Float32[Tensor, "f 2 1"] = (box[:, camera, :, 2:3] - box[:, camera, :, 0:1]) * 0.5 * letterbox.scale
        circles.append(torch.cat([centre, radius], dim=-1))
    return torch.stack(circles, dim=1)


def segment_track(data: SegmentData, run: TrackerRun, meta: TrackMetadata) -> SegmentTrack:
    """The tracker's frames as the per-segment record (camera pixels)."""
    frames: list[FrameResult] = run.frames
    f: int = len(frames)
    untracked: HandPose = HandPose(torch.full((3, 3), torch.nan), torch.full((3,), torch.nan), torch.full((22,), torch.nan))
    poses: list[list[HandPose]] = [[untracked if pose is None else pose for pose in frame.poses] for frame in frames]
    rotation: Float32[Tensor, "f 2 3 3"] = torch.stack([torch.stack([pose.rotation for pose in row]) for row in poses])
    translation: Float32[Tensor, "f 2 3"] = torch.stack([torch.stack([pose.translation for pose in row]) for row in poses])
    joint_angles: Float32[Tensor, "f 2 22"] = torch.stack([torch.stack([pose.joint_angles for pose in row]) for row in poses])
    keypoints_net: Float32[Tensor, "f c 2 21 2"] = torch.stack([frame.keypoints for frame in frames])
    keypoints_cam: Float32[Tensor, "f c 2 21 2"] = torch.stack(
        [letterbox.from_net(keypoints_net[:, camera]) for camera, letterbox in enumerate(data.letterboxes)], dim=1
    )
    return SegmentTrack(
        meta=meta,
        video_time_ns=data.timeline.video_time_ns[:f].copy(),
        frame_index=np.arange(f, dtype=np.int64),
        tracked=torch.stack([frame.tracked for frame in frames]).numpy(),
        rotation=rotation.numpy(),
        translation=translation.numpy(),
        joint_angles=joint_angles.numpy(),
        landmarks=torch.stack([frame.landmarks for frame in frames]).numpy(),
        box=camera_boxes(data.letterboxes, torch.stack([frame.circle for frame in frames])).numpy(),
        box_source=torch.stack([frame.box_source for frame in frames]).numpy(),
        keypoints_2d=keypoints_cam.numpy(),
        presence=torch.stack([frame.presence for frame in frames]).numpy(),
        detnet_camera=np.array([frame.detnet_camera for frame in frames], dtype=np.int8),
        detnet_presence=torch.stack([frame.detnet_presence for frame in frames]).numpy(),
        fit_energy=torch.stack([frame.fit_energy for frame in frames]).numpy(),
    )


def detnet_alone_track(data: SegmentData, alone: DetNetAlone, meta: TrackMetadata, threshold: float = 0.5) -> SegmentTrack:
    """DetNet-alone as a ``SegmentTrack``: its presence per camera and hand, and its box wherever presence > ``threshold``."""
    f: int = alone.probability.shape[0]
    cameras: int = alone.probability.shape[1]
    reported: Bool[Tensor, "f c 2"] = alone.probability > threshold
    box: Float32[Tensor, "f c 2 4"] = torch.where(reported[..., None], camera_boxes(data.letterboxes, alone.circle), torch.nan)
    source: Int8[Tensor, "f c 2"] = torch.where(reported, int(BoxSource.DETNET), int(BoxSource.NONE)).to(torch.int8)
    return SegmentTrack(
        meta=meta,
        video_time_ns=data.timeline.video_time_ns[:f].copy(),
        frame_index=np.arange(f, dtype=np.int64),
        tracked=np.zeros((f, 2), dtype=bool),
        rotation=np.full((f, 2, 3, 3), np.nan, dtype=np.float32),
        translation=np.full((f, 2, 3), np.nan, dtype=np.float32),
        joint_angles=np.full((f, 2, 22), np.nan, dtype=np.float32),
        landmarks=np.full((f, 2, 21, 3), np.nan, dtype=np.float32),
        box=box.numpy(),
        box_source=source.numpy(),
        keypoints_2d=np.full((f, cameras, 2, 21, 2), np.nan, dtype=np.float32),
        presence=alone.probability.numpy(),
        detnet_camera=np.full(f, -1, dtype=np.int8),
        detnet_presence=np.full((f, 2), np.nan, dtype=np.float32),
        fit_energy=np.full((f, 2), np.nan, dtype=np.float32),
    )


def read_state(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    """A model-only state_dict and its sha256, checked against the ``<file>.sha256`` sidecar (read once, O_DIRECT)."""
    payload: bytes = read_disk(path)
    digest: str = hashlib.sha256(payload).hexdigest()
    expected: str = read_disk(Path(f"{path}.sha256")).decode("ascii").split()[0]
    if digest != expected:
        raise ValueError(f"{path}: sha256 {digest} does not match its sidecar {expected}")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True), digest


def load_weights(model: nn.Module, path: Path) -> str:
    """Load a model-only state_dict after checking its ``<file>.sha256`` sidecar (the checkpoint writer's and
    ``promote.sh``'s convention); returns the digest.

    Both files are read once through ``read_disk`` (O_DIRECT, as the checkpoint reader does) and the state_dict is
    loaded from those bytes, so a checkpoint replaced during a run cannot mix two versions.
    """
    payload: bytes = read_disk(path)
    digest: str = hashlib.sha256(payload).hexdigest()
    expected: str = read_disk(Path(f"{path}.sha256")).decode("ascii").split()[0]
    if digest != expected:
        raise ValueError(f"{path}: sha256 {digest} does not match its sidecar {expected}")
    model.load_state_dict(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True))
    return digest
