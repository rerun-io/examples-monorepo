"""People-mask layer: SAM 3 "person" masks for the video streams of Rerun recordings.

Reads the ``VideoStream`` entities of one or more ``.rrd`` files (layers of one recording) through a local
``rr.server.Server``, decodes the selected frames with NVDEC (torchcodec over the packets muxed into an in-memory MP4,
H.264 / H.265 / AV1), runs posekit's batched ``Sam3Segmenter`` with a text prompt, unions the instances (optionally
also on 2x2 overlapping tiles for small, distant people), dilates, and logs a ``SegmentationImage`` (uint8, 1 = person)
at ``<parent of the video entity>/people_mask`` on the stream's timeline with the frame's own stamp.
"""

import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import rerun as rr
import torch
from einops import rearrange
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from posekit.models.base import SegmentationPrompts
from posekit.models.sam3_segmenter import Sam3Segmenter, Sam3SegmenterConfig
from posekit.ops.people_masks import TileWindow, dilate_mask, overlay_mask, selection_windows, tile_to_image, tile_windows, union_instances
from posekit.predictions import BoxDetections

PERSON_CLASS_ID: int = 1
"""Mask value of a person pixel (0 = keep)."""
PEOPLE_MASK_ANNOTATION: rr.AnnotationContext = rr.AnnotationContext(
    [rr.AnnotationInfo(id=0, label="keep", color=(0, 0, 0, 0)), rr.AnnotationInfo(id=PERSON_CLASS_ID, label="person", color=(255, 0, 0, 160))]
)
"""Class labels of a people-mask entity."""


@dataclass(frozen=True, slots=True)
class MaskOptions:
    """How SAM 3 detections become one person mask per image."""

    prompt: str = "person"
    """Text prompt for SAM 3."""
    score_threshold: float = 0.5
    """Minimum instance confidence kept."""
    mask_threshold: float = 0.5
    """Probability threshold binarizing instance masks."""
    min_area_px: int = 0
    """Instances with fewer pixels (at the resolution they were segmented at: image or tile) are dropped."""
    dilate_px: int = 0
    """Radius of the disc the union is dilated by, in image pixels."""
    tiles: int = 1
    """1 = whole image only; 2 = also 2x2 overlapping tiles, each resized to the model input, ORed into the mask."""
    tile_overlap: float = 0.2
    """Fraction of each side two neighbouring tiles share (tiles = 2)."""
    batch_size: int = 4
    """Images (or tiles) per SAM 3 call."""
    dtype: Literal["float32", "bfloat16", "float16"] = "float16"
    """SAM 3's autocast dtype (:attr:`Sam3SegmenterConfig.dtype`). float16 keeps the clip B people layer within the band
    against FP32 (mean IoU 0.990, recall >= 0.98 per size) at 2.5x SAM 3's speed; bfloat16 drifted (IoU 0.978)."""
    backend: Literal["torch", "tensorrt"] = "tensorrt"
    """How SAM 3 runs (:attr:`Sam3SegmenterConfig.backend`). ``tensorrt`` (fp16 engines for the trunk and the heads, the
    patch embedding accumulated in fp32) takes about half the time of a torch fp16 call on the 5090; against the torch
    layer the clip 007 window keeps mean IoU 0.998-0.999 / 0.983-0.989 per lens (two engine builds) and clip B 0.997 /
    0.981. FP8 trunks (linears, or linears + attention) dropped the small-people lens to 0.94 / 0.93. ``torch`` is the
    eager fallback, under ``dtype`` autocast."""


@dataclass
class PeopleMasksConfig:
    """Write a people-mask layer for the video streams of Rerun recordings."""

    rrds: tuple[Path, ...]
    """Recordings (layers of one recording) that together hold the video entities."""
    entities: tuple[str, ...]
    """Video entity paths or glob patterns (``/world/rig_00/cam_*/pinhole/video``)."""
    rr_config: RerunTyroConfig
    """Output sink; pass ``--rr-config.save <layer.rrd> --rr-config.headless --rr-config.recording-id <segment>``."""
    exclude: tuple[str, ...] = ()
    """Entity paths or glob patterns removed from the selection."""
    stride: int = 1
    """Take every ``stride``-th frame of each stream (ignored when ``frames`` is given)."""
    frames: tuple[int, ...] = ()
    """Explicit frame indices into each stream."""
    timeline: str = "video_time"
    """Timeline of the video streams and of the masks."""
    mask: MaskOptions = field(default_factory=MaskOptions)
    """Mask options."""
    overlay_dir: Path | None = None
    """Write one evidence PNG per image (mask tinted red) here."""
    device: str = "cuda"
    """Inference and decode device."""


@dataclass(frozen=True, slots=True)
class VideoSource:
    """One selected video entity and the dataset (input file) that holds it."""

    entity: str
    """Video entity path."""
    dataset: str
    """Name of the served dataset holding it."""


def select_entities(available: list[str], patterns: tuple[str, ...], exclude: tuple[str, ...]) -> list[str]:
    """The available entities matching any pattern and no exclude pattern, in sorted order.

    Raises:
        ValueError: If a pattern matches nothing.
    """
    for pattern in patterns:
        if not any(fnmatch(entity, pattern) for entity in available):
            raise ValueError(f"entity pattern {pattern!r} matches no VideoStream; available: {sorted(available)}")
    return sorted(e for e in available if any(fnmatch(e, p) for p in patterns) and not any(fnmatch(e, p) for p in exclude))


def mask_entity(video_entity: str) -> str:
    """``<parent of the video entity>/people_mask``."""
    return f"{video_entity.rstrip('/').rsplit('/', 1)[0]}/people_mask"


def set_frame_time(timeline: str, stamp: np.timedelta64 | np.datetime64 | np.integer) -> None:
    """Set the global time to a frame's stamp, keeping the timeline's kind (duration, timestamp or sequence)."""
    if isinstance(stamp, np.timedelta64):
        rr.set_time(timeline, duration=stamp)
    elif isinstance(stamp, np.datetime64):
        rr.set_time(timeline, timestamp=stamp)
    else:
        rr.set_time(timeline, sequence=int(stamp))


class PeopleMasker:
    """Batched SAM 3 person detections (whole image plus optional tiles) and their dilated union mask.

    ``detect`` is the entry point for stages that need instances (tracker initialisation and re-detection);
    ``__call__`` gives the mask layer.
    """

    def __init__(self, options: MaskOptions, device: str) -> None:
        """Load SAM 3."""
        self.options: MaskOptions = options
        self.prompts: SegmentationPrompts = SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text=options.prompt)
        self.segmenter: Sam3Segmenter = Sam3SegmenterConfig(
            device=device,
            score_threshold=options.score_threshold,
            mask_threshold=options.mask_threshold,
            dtype=options.dtype,
            backend=options.backend,
        ).setup()
        self.segmenter.prepare(options.prompt)

    def segment(self, images_rgb: UInt8[Tensor, "b h w 3"]) -> BoxDetections:
        """SAM 3 in chunks of ``batch_size``; instances under ``min_area_px`` are dropped."""
        parts: list[BoxDetections] = []
        for begin in range(0, int(images_rgb.shape[0]), self.options.batch_size):
            detections: BoxDetections = self.segmenter(images_rgb[begin : begin + self.options.batch_size].contiguous(), self.prompts)
            assert detections.masks is not None
            keep: Bool[Tensor, "n"] = detections.masks.sum(dim=(1, 2)) >= self.options.min_area_px
            parts.append(
                BoxDetections(
                    xyxy=detections.xyxy[keep],
                    scores=detections.scores[keep],
                    frame_indices=detections.frame_indices[keep] + begin,
                    masks=detections.masks[keep],
                )
            )
        return BoxDetections(
            xyxy=torch.cat([p.xyxy for p in parts]),
            scores=torch.cat([p.scores for p in parts]),
            frame_indices=torch.cat([p.frame_indices for p in parts]),
            masks=torch.cat([p.masks for p in parts if p.masks is not None]),
        )

    def detect(self, frames_rgb: UInt8[Tensor, "f h w 3"]) -> BoxDetections:
        """Person instances with full-frame masks and boxes in image pixels.

        With ``tiles = 2`` the tile instances are added in image coordinates; an instance seen in the whole image and in
        a tile appears more than once (no de-duplication).
        """
        n_frames: int = int(frames_rgb.shape[0])
        height: int = int(frames_rgb.shape[1])
        width: int = int(frames_rgb.shape[2])
        whole: BoxDetections = self.segment(frames_rgb)
        windows: list[TileWindow] = tile_windows(height, width, grid=self.options.tiles, overlap=self.options.tile_overlap)
        if not windows:
            return whole
        # Crops are window-major: crop t * n_frames + f is window t of frame f.
        crops: UInt8[Tensor, "ft th tw 3"] = torch.cat([frames_rgb[:, w.y0 : w.y1, w.x0 : w.x1] for w in windows], dim=0)
        tiled: BoxDetections = self.segment(crops)
        assert tiled.masks is not None and whole.masks is not None
        masks: list[Tensor] = [whole.masks]
        boxes: list[Tensor] = [whole.xyxy]
        order: list[Tensor] = [whole.frame_indices]
        scores: list[Tensor] = [whole.scores]
        window_of: Tensor = tiled.frame_indices // n_frames
        for t, window in enumerate(windows):
            rows: Bool[Tensor, "n"] = window_of == t
            placed: tuple[Tensor, Tensor] = tile_to_image(tiled.masks[rows], tiled.xyxy[rows], window, height=height, width=width)
            masks.append(placed[0])
            boxes.append(placed[1])
            order.append(tiled.frame_indices[rows] % n_frames)
            scores.append(tiled.scores[rows])
        return BoxDetections(xyxy=torch.cat(boxes), scores=torch.cat(scores), frame_indices=torch.cat(order), masks=torch.cat(masks))

    def __call__(self, frames_rgb: UInt8[Tensor, "f h w 3"]) -> Bool[ndarray, "f h w"]:
        """One dilated person mask per frame."""
        detections: BoxDetections = self.detect(frames_rgb)
        assert detections.masks is not None
        union: Bool[ndarray, "f h w"] = union_instances(
            detections.masks.cpu().numpy(), detections.frame_indices.cpu().numpy(), n_frames=int(frames_rgb.shape[0]), min_area_px=0
        )
        return np.stack([dilate_mask(m, self.options.dilate_px) for m in union])


def write_overlay(overlay_dir: Path, entity: str, index: int, rgb: UInt8[ndarray, "h w 3"], mask: Bool[ndarray, "h w"]) -> None:
    """Save the frame with its person mask tinted red as ``<entity slug>_<index>.png`` (evidence images)."""
    path: Path = overlay_dir / f"{entity.strip('/').replace('/', '_')}_{index:06d}.png"
    cv2.imwrite(str(path), cv2.cvtColor(overlay_mask(rgb, mask), cv2.COLOR_RGB2BGR))
    path.chmod(0o644)


def decode_selected(video: CatalogVideo, indices: list[int], device: str) -> tuple[list[int], UInt8[Tensor, "n h w 3"]]:
    """NVDEC-decode the selected packets, one decoder per keyframe run.

    Returns:
        The decoded packet indices (ascending) and their RGB frames.
    """
    from torchcodec.decoders import VideoDecoder

    decoded: list[int] = []
    frames: list[UInt8[Tensor, "m h w 3"]] = []
    for window in selection_windows(video.keyframes, indices):
        last: int = window.indices[-1]
        data: bytes = wrap_mp4(video.samples[window.first : last + 1], video.keyframes[window.first : last + 1], 30, codec=video.codec)
        decoder = VideoDecoder(data, device=device, seek_mode="exact")
        batch: UInt8[Tensor, "m 3 h w"] = decoder.get_frames_at([i - window.first for i in window.indices]).data
        frames.append(rearrange(batch, "m c h w -> m h w c").contiguous())
        decoded.extend(window.indices)
    return decoded, torch.cat(frames, dim=0)


def main(config: PeopleMasksConfig) -> None:
    """Write the people-mask layer and print speed and GPU memory."""
    torch.cuda.reset_peak_memory_stats()
    with rr.server.Server(datasets={f"in{i}": [str(path)] for i, path in enumerate(config.rrds)}) as server:
        client = server.client()
        sources: dict[str, VideoSource] = {}
        segment_ids: set[str] = set()
        for i in range(len(config.rrds)):
            name: str = f"in{i}"
            dataset = client.get_dataset(name)
            segment_ids.update(dataset.segment_ids())
            for column in dataset.schema().component_columns():
                if column.component == "VideoStream:sample":
                    sources.setdefault(column.entity_path, VideoSource(column.entity_path, name))
        if len(segment_ids) != 1:
            raise ValueError(f"the inputs hold {len(segment_ids)} recordings ({sorted(segment_ids)}); expected layers of one")
        segment_id: str = segment_ids.pop()
        if config.rr_config.recording_id != segment_id:
            raise ValueError(f"pass --rr-config.recording-id {segment_id} so the mask layer joins the input recording")
        entities: list[str] = select_entities(list(sources), config.entities, config.exclude)
        print(f"segment {segment_id}: {len(entities)} video entities: {entities}")

        masker = PeopleMasker(config.mask, config.device)
        if config.overlay_dir is not None:
            config.overlay_dir.mkdir(parents=True, exist_ok=True)
        n_images: int = 0
        segment_seconds: float = 0.0
        for entity in entities:
            target: str = mask_entity(entity)
            rr.log(target, PEOPLE_MASK_ANNOTATION, static=True)
            video: CatalogVideo = read_catalog_videos(client.get_dataset(sources[entity].dataset), segment_id, [entity], config.timeline)[0]
            n_packets: int = len(video.samples)
            selected: list[int] = list(config.frames) if config.frames else list(range(0, n_packets, config.stride))
            for begin in range(0, len(selected), config.mask.batch_size):
                decoded: tuple[list[int], UInt8[Tensor, "n h w 3"]] = decode_selected(
                    video, selected[begin : begin + config.mask.batch_size], config.device
                )
                torch.cuda.synchronize()
                start: float = time.perf_counter()
                masks: Bool[ndarray, "n h w"] = masker(decoded[1])
                torch.cuda.synchronize()
                segment_seconds += time.perf_counter() - start
                n_images += len(decoded[0])
                frames_cpu: UInt8[ndarray, "n h w 3"] | None = decoded[1].cpu().numpy() if config.overlay_dir is not None else None
                for row, index in enumerate(decoded[0]):
                    set_frame_time(config.timeline, video.times[index])
                    rr.log(target, rr.SegmentationImage(masks[row].astype(np.uint8) * PERSON_CLASS_ID))
                    if frames_cpu is not None and config.overlay_dir is not None:
                        write_overlay(config.overlay_dir, entity, index, frames_cpu[row], masks[row])
                person_px: Int64[ndarray, "n"] = masks.sum(axis=(1, 2)).astype(np.int64)
                print(f"{entity} frames {decoded[0]}: person px {person_px.tolist()}")
    recording: rr.RecordingStream | None = rr.get_data_recording()
    assert recording is not None
    recording.flush()
    allocated_gb: float = torch.cuda.max_memory_allocated() / 1e9
    reserved_gb: float = torch.cuda.max_memory_reserved() / 1e9
    print(
        f"{n_images} images, {segment_seconds / max(n_images, 1):.3f} s/image (segmentation incl. tiles), "
        f"peak GPU allocated {allocated_gb:.1f} GB, reserved {reserved_gb:.1f} GB (NVDEC decoders come on top)"
    )
