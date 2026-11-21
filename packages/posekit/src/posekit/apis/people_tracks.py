"""Tracked people masks: SAM 3 detects people on sparse frames, EfficientTAM propagates their masks in between.

The video streams (one tracker slot each, e.g. the two fisheye lenses of a 360 camera) are decoded in order with
NVDEC, one thread per stream, a chunk ahead of the tracker. Every ``redetect_every`` tracker steps SAM 3 runs on all
streams; its instances are matched to the live tracks by mask IoU (matched tracks are re-anchored on the SAM 3 box,
unmatched instances start tracks, duplicate tracks are dropped). A track that stays empty for ``max_misses`` steps, or
that SAM 3 does not confirm at ``max_unconfirmed`` re-detects in a row, is dropped. On frames that are a multiple of
``output_every`` the union of the track masks (and of the SAM 3 masks on a re-detect step) is dilated at full
resolution and logged as a ``SegmentationImage`` (uint8, 1 = person) at ``<parent of the video entity>/people_mask``
with the frame's own stamp, in the mask layer contract (track 3, v2): ``mask_downscale`` times smaller than the frame
(a cell is masked when any of its pixels is, the pooling ``mask_union`` would do), with one static
``Transform3D(scale=[mask_downscale, mask_downscale, 1])`` to frame pixels.
"""

import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, UInt8
from numpy import ndarray
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rerun_log_utils import RerunTyroConfig
from torch import Tensor

from posekit.apis.people_masks import (
    PEOPLE_MASK_ANNOTATION,
    PERSON_CLASS_ID,
    MaskOptions,
    PeopleMasker,
    mask_entity,
    select_entities,
    set_frame_time,
    write_overlay,
)
from posekit.models.base import SegmentationPrompts
from posekit.models.sam2_video import Sam2VideoSegmenter, Sam2VideoSegmenterConfig
from posekit.ops.people_masks import dilate_masks, selection_windows
from posekit.ops.people_tracks import RedetectPlan, count_misses, mask_iou, plan_redetect
from posekit.predictions import BoxDetections


@dataclass(frozen=True, slots=True)
class TrackOptions:
    """How SAM 3 detections are propagated by the video tracker."""

    tracker: Sam2VideoSegmenterConfig = field(default_factory=lambda: Sam2VideoSegmenterConfig(memory_on_device=True, bf16=True))
    """EfficientTAM variant and memory options."""
    stride: int = 10
    """Frames between two tracker steps; must divide ``output_every``."""
    redetect_every: int = 2
    """Tracker steps between two SAM 3 re-detects (step 0 initialises).

    2 at stride 10 keeps every output frame within 10 frames of a SAM 3 detection: at 3 the tracker drifted off the
    operator on one lens between re-detects.
    """
    max_misses: int = 3
    """A track whose mask is empty for this many steps in a row is dropped."""
    max_unconfirmed: int = 2
    """A track that matches no SAM 3 instance at this many re-detects in a row is dropped (drift onto background)."""
    match_iou: float = 0.3
    """Mask IoU at which a SAM 3 instance continues a live track."""
    iou_subsample: int = 8
    """Pixel step (full resolution) of the masks compared for matching; a multiple of ``downscale``."""
    downscale: int = 2
    """The tracker sees frames shrunk by this factor (it segments at 512 px internally); masks are upsampled back."""
    max_tracks: int = 24
    """Tracks per stream at most (new instances are ignored above it)."""
    mask_prompts: bool = True
    """Prompt the tracker with the SAM 3 mask as well as its box (keeps small people and the operator on shape)."""


@dataclass
class PeopleTracksConfig:
    """Write a tracked people-mask layer for the video streams of one Rerun recording."""

    rrd: Path
    """Recording holding the video entities."""
    entities: tuple[str, ...]
    """Video entity paths or glob patterns; each stream is one tracker slot (same length and resolution)."""
    rr_config: RerunTyroConfig
    """Output sink; pass ``--rr-config.save <layer.rrd> --rr-config.headless --rr-config.recording-id <segment>``."""
    start: int = 0
    """First frame index (must be a multiple of ``output_every``)."""
    end: int | None = None
    """Frame index after the last (default: end of the streams)."""
    output_every: int = 10
    """Masks are written for the frame indices that are multiples of this."""
    timeline: str = "video_time"
    """Timeline of the video streams and of the masks."""
    mask_downscale: int = 4
    """Mask size = frame size / this (480 px for 1920 px lenses, FullCircle's grid); must divide the frame size. 1 = full
    resolution."""
    mask: MaskOptions = field(default_factory=MaskOptions)
    """SAM 3 options (prompt, thresholds, tiles) and the output dilation."""
    track: TrackOptions = field(default_factory=TrackOptions)
    """Tracker options."""
    overlay_frames: tuple[int, ...] = ()
    """Frame indices with an evidence PNG (mask tinted red) in ``overlay_dir``."""
    overlay_dir: Path | None = None
    """Directory of the evidence PNGs."""
    decode_chunk: int = 30
    """Tracker frames decoded per NVDEC call."""
    device: str = "cuda"
    """Inference and decode device."""


class StreamDecoder:
    """Sequential NVDEC decode of one stream from the keyframe before ``start``."""

    def __init__(self, video: CatalogVideo, start: int, end: int, device: str) -> None:
        """Mux the packets ``[keyframe before start, end)`` into an in-memory MP4 and open a decoder."""
        from torchcodec.decoders import VideoDecoder

        self.first: int = selection_windows(video.keyframes, [start])[0].first
        data: bytes = wrap_mp4(video.samples[self.first : end], video.keyframes[self.first : end], 30, codec=video.codec)
        self.decoder = VideoDecoder(data, device=device, seek_mode="exact")

    def frames(self, indices: list[int]) -> UInt8[Tensor, "n h w 3"]:
        """RGB frames at ascending stream indices."""
        batch: UInt8[Tensor, "n 3 h w"] = self.decoder.get_frames_at([i - self.first for i in indices]).data
        # NVDEC frames are NHWC in memory (NCHW is a view), so this is a view, not a copy
        return rearrange(batch, "n c h w -> n h w c").contiguous()


def prefetched_chunks(
    videos: tuple[CatalogVideo, ...], chunks: list[list[int]], start: int, end: int, device: str
) -> Iterator[UInt8[Tensor, "s n h w 3"]]:
    """The frames of each chunk, all streams stacked. Decoding starts at the call, and the next chunk decodes on one
    thread per stream while the caller computes on the current one (two NVDEC threads decode twice the frames per
    second of one on the 5090).

    A torchcodec CUDA decoder fails on a thread other than the one that created it, so each decoder is made on its thread.
    """
    threads: list[ThreadPoolExecutor] = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"decode-{slot}") for slot in range(len(videos))]
    decoders: list[Future[StreamDecoder]] = [thread.submit(StreamDecoder, v, start, end, device) for thread, v in zip(threads, videos, strict=True)]

    def decode(slot: int, chunk: list[int]) -> UInt8[Tensor, "n h w 3"]:
        frames: UInt8[Tensor, "n h w 3"] = decoders[slot].result().frames(chunk)
        torch.cuda.synchronize(device)
        return frames

    def submit(chunk: list[int]) -> list[Future[Tensor]]:
        return [thread.submit(decode, slot, chunk) for slot, thread in enumerate(threads)]

    def stacked(pending: list[Future[Tensor]]) -> Iterator[UInt8[Tensor, "s n h w 3"]]:
        try:
            for position in range(len(chunks)):
                batches: list[Tensor] = [future.result() for future in pending]
                pending = submit(chunks[position + 1]) if position + 1 < len(chunks) else []
                yield torch.stack(batches)
        finally:
            for thread in threads:
                thread.shutdown(cancel_futures=True)

    return stacked(submit(chunks[0]) if chunks else [])


class PeopleTracker:
    """SAM 3 (re-)detection plus EfficientTAM propagation over a fixed set of streams."""

    def __init__(self, options: TrackOptions, mask: MaskOptions, device: str) -> None:
        """Load SAM 3 and the tracker."""
        self.options: TrackOptions = options
        self.masker: PeopleMasker = PeopleMasker(mask, device)
        self.segmenter: Sam2VideoSegmenter = options.tracker.setup()
        self.live: list[list[int]] = []
        self.misses: list[dict[int, int]] = []
        self.unconfirmed: list[dict[int, int]] = []
        self.last_masks: list[Bool[Tensor, "t h w"]] = []
        self.next_id: int = 0
        self.steps: int = 0
        self.redetects: int = 0
        self.counts: dict[str, int] = {"reanchored": 0, "started": 0, "duplicate": 0, "unconfirmed": 0, "empty": 0}
        """Track events since the start (for the progress log)."""
        self.detect_seconds: float = 0.0
        """Wall time spent in SAM 3 (synchronised)."""

    def step(self, frames_rgb: UInt8[Tensor, "s h w 3"]) -> Bool[Tensor, "s h w"]:
        """Advance all streams by one tracker step; returns the person mask of each stream."""
        n_slots: int = int(frames_rgb.shape[0])
        if not self.live:
            self.live = [[] for _ in range(n_slots)]
            self.misses = [{} for _ in range(n_slots)]
            self.unconfirmed = [{} for _ in range(n_slots)]
        sub: int = self.options.iou_subsample
        scale: int = self.options.downscale
        union: Bool[Tensor, "s h w"] = torch.zeros(frames_rgb.shape[:3], dtype=torch.bool, device=frames_rgb.device)
        prompts: SegmentationPrompts | None = None
        if self.steps % self.options.redetect_every == 0:
            self.redetects += 1
            detect_start: float = time.perf_counter()
            detections: BoxDetections = self.masker.detect(frames_rgb)
            torch.cuda.synchronize()
            self.detect_seconds += time.perf_counter() - detect_start
            assert detections.masks is not None
            rows: list[int] = []
            boxes: list[Tensor] = []
            prompt_masks: list[Tensor] = []
            ids: list[int] = []
            for slot in range(n_slots):
                in_slot: Tensor = torch.where(detections.frame_indices == slot)[0]
                det_masks: Bool[Tensor, "m h w"] = detections.masks[in_slot]
                if len(in_slot):
                    union[slot] |= det_masks.any(dim=0)
                if self.live[slot]:
                    iou: ndarray = mask_iou(self.last_masks[slot], det_masks[:, ::sub, ::sub]).cpu().numpy()
                else:
                    iou = np.zeros((0, len(in_slot)), dtype=np.float32)
                detection_iou: ndarray = mask_iou(det_masks[:, ::sub, ::sub], det_masks[:, ::sub, ::sub]).cpu().numpy()
                plan: RedetectPlan = plan_redetect(
                    self.live[slot], iou, detection_iou, threshold=self.options.match_iou, next_id=self.next_id, max_tracks=self.options.max_tracks
                )
                self.next_id = plan.next_id
                confirmed: dict[int, bool] = {t: True for t, _ in plan.reanchor} | {t: False for t in plan.unmatched}
                self.unconfirmed[slot], stale = count_misses(self.unconfirmed[slot], confirmed, max_misses=self.options.max_unconfirmed)
                gone: list[int] = [*plan.drop, *stale]
                self.counts["reanchored"] += len(plan.reanchor)
                self.counts["started"] += len(plan.start)
                self.counts["duplicate"] += len(plan.drop)
                self.counts["unconfirmed"] += len(stale)
                self.segmenter.drop_tracks(slot, gone)
                self.live[slot] = [t for t in self.live[slot] if t not in gone]
                for track_id, col in (*plan.reanchor, *plan.start):
                    rows.append(slot)
                    boxes.append(detections.xyxy[in_slot[col]] / scale)
                    prompt_masks.append(det_masks[col])
                    ids.append(track_id)
                self.live[slot] += [track_id for track_id, _ in plan.start]
            if ids:
                device: torch.device = frames_rgb.device
                small_masks: Bool[Tensor, "m hs ws"] | None = None
                if self.options.mask_prompts:
                    full: Tensor = torch.stack(prompt_masks)[:, None].float()
                    small_masks = (F.interpolate(full, scale_factor=1.0 / scale, mode="area")[:, 0] >= 0.5) if scale > 1 else full[:, 0] > 0
                prompts = SegmentationPrompts(
                    frame_indices=torch.tensor(rows, dtype=torch.long, device=device),
                    boxes_xyxy=torch.stack(boxes).float(),
                    track_ids=torch.tensor(ids, dtype=torch.long, device=device),
                    masks=small_masks,
                )
        small: UInt8[Tensor, "s hs ws 3"] = frames_rgb
        if scale > 1:
            chw: Tensor = F.interpolate(frames_rgb.permute(0, 3, 1, 2).float(), scale_factor=1.0 / scale, mode="area")
            small = chw.round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        tracks: BoxDetections = self.segmenter.step(small, prompts)
        self.steps += 1
        assert tracks.masks is not None and tracks.track_ids is not None
        nonempty: list[bool] = tracks.masks.flatten(1).any(dim=1).tolist()
        track_ids: list[int] = tracks.track_ids.tolist()
        slots: list[int] = tracks.frame_indices.tolist()
        track_union: Bool[Tensor, "s hs ws"] = torch.zeros(small.shape[:3], dtype=torch.bool, device=small.device)
        self.last_masks = []
        for slot in range(n_slots):
            rows_in_slot: list[int] = [i for i, s in enumerate(slots) if s == slot]
            if rows_in_slot:
                track_union[slot] = tracks.masks[rows_in_slot].any(dim=0)
            seen: dict[int, bool] = {track_ids[i]: nonempty[i] for i in rows_in_slot}
            self.misses[slot], dropped = count_misses(self.misses[slot], seen, max_misses=self.options.max_misses)
            self.segmenter.drop_tracks(slot, dropped)
            self.counts["empty"] += len(dropped)
            # live follows the tracker's own row order so the rows of last_masks line up with it
            kept: list[int] = [i for i in rows_in_slot if track_ids[i] not in dropped]
            self.live[slot] = [track_ids[i] for i in kept]
            self.last_masks.append(tracks.masks[kept][:, :: sub // scale, :: sub // scale])
        if scale > 1:
            track_union = F.interpolate(track_union[:, None].to(torch.uint8), size=union.shape[1:], mode="nearest")[:, 0].bool()
        return union | track_union


def main(config: PeopleTracksConfig) -> None:
    """Write the tracked people-mask layer and print speed and GPU memory."""
    if config.output_every % config.track.stride or config.start % config.output_every:
        raise ValueError("track.stride must divide output_every, and start must be a multiple of output_every")
    process_start: float = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with rr.server.Server(datasets={"in": [str(config.rrd)]}) as server:
        dataset = server.client().get_dataset("in")
        segment_ids: list[str] = list(dataset.segment_ids())
        if len(segment_ids) != 1:
            raise ValueError(f"{config.rrd} holds {len(segment_ids)} recordings; expected one")
        segment_id: str = segment_ids[0]
        if config.rr_config.recording_id != segment_id:
            raise ValueError(f"pass --rr-config.recording-id {segment_id} so the mask layer joins the input recording")
        available: list[str] = [c.entity_path for c in dataset.schema().component_columns() if c.component == "VideoStream:sample"]
        entities: list[str] = select_entities(available, config.entities, ())
        videos: tuple[CatalogVideo, ...] = read_catalog_videos(dataset, segment_id, entities, config.timeline)
    n_packets: int = min(len(v.samples) for v in videos)
    end: int = n_packets if config.end is None else min(config.end, n_packets)
    tracked: list[int] = list(range(config.start, end, config.track.stride))
    print(f"segment {segment_id}: streams {entities}, {len(tracked)} tracker steps over frames {config.start}..{end - 1}")
    print(f"rrd read {time.perf_counter() - process_start:.1f} s")

    load_start: float = time.perf_counter()
    chunks: list[list[int]] = [tracked[begin : begin + config.decode_chunk] for begin in range(0, len(tracked), config.decode_chunk)]
    decoded: Iterator[UInt8[Tensor, "s n h w 3"]] = prefetched_chunks(videos, chunks, config.start, end, config.device)
    tracker = PeopleTracker(config.track, config.mask, config.device)
    targets: list[str] = [mask_entity(e) for e in entities]
    factor: int = config.mask_downscale
    for target in targets:
        rr.log(target, PEOPLE_MASK_ANNOTATION, static=True)
        rr.log(target, rr.Transform3D(scale=[float(factor), float(factor), 1.0]), static=True)
    if config.overlay_dir is not None:
        config.overlay_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    run_start: float = time.perf_counter()
    print(f"model load + decoder setup {run_start - load_start:.1f} s")
    n_masks: int = 0
    for chunk, frames in zip(chunks, decoded, strict=True):
        n_slots, height, width = int(frames.shape[0]), int(frames.shape[2]), int(frames.shape[3])
        if height % factor or width % factor:
            raise ValueError(f"mask_downscale {factor} does not divide the {width}x{height} frames")
        for column, index in enumerate(chunk):
            union: Bool[Tensor, "s h w"] = tracker.step(frames[:, column])
            if index % config.output_every:
                continue
            dilated: Bool[Tensor, "s h w"] = dilate_masks(union, config.mask.dilate_px)
            cells: Bool[Tensor, "s hm wm"] = dilated.reshape(n_slots, height // factor, factor, width // factor, factor).any(dim=4).any(dim=2)
            masks: UInt8[ndarray, "s hm wm"] = (cells.to(torch.uint8) * PERSON_CLASS_ID).cpu().numpy()
            for slot, entity in enumerate(entities):
                set_frame_time(config.timeline, videos[slot].times[index])
                rr.log(targets[slot], rr.SegmentationImage(masks[slot]))
                n_masks += 1
                if config.overlay_dir is not None and index in config.overlay_frames:
                    write_overlay(config.overlay_dir, entity, index, frames[slot, column].cpu().numpy(), dilated[slot].cpu().numpy())
        live: list[int] = [len(t) for t in tracker.live]
        elapsed: float = time.perf_counter() - run_start
        print(
            f"frame {chunk[-1]}: live tracks {live}, events {tracker.counts}, {tracker.steps * len(entities) / elapsed:.1f} images/s, "
            f"SAM 3 {tracker.detect_seconds:.1f} of {elapsed:.1f} s"
        )
    recording: rr.RecordingStream | None = rr.get_data_recording()
    assert recording is not None
    recording.flush()
    torch.cuda.synchronize()
    wall: float = time.perf_counter() - run_start
    n_images: int = tracker.steps * len(entities)
    print(
        f"tracked {n_images} images ({tracker.redetects} SAM 3 re-detects), wrote {n_masks} masks in {wall:.1f} s "
        f"(model load excluded; SAM 3 {tracker.detect_seconds:.1f} s): {n_images / wall:.1f} images/s, {wall / max(n_masks, 1):.3f} s/mask; "
        f"peak GPU allocated {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, reserved {torch.cuda.max_memory_reserved() / 1e9:.1f} GB"
    )
    print(f"total incl. rrd read and model load {time.perf_counter() - process_start:.1f} s")
