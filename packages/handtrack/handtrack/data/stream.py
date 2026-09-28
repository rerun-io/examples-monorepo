"""The catalog streaming dataset: one decode feeds DetNet and KeyNet batches.

Producer threads (default 4, ``num_workers=0`` style: threads in this process) take segments from the epoch's shuffled
list. Per segment a producer reads the statics, the hand timeline and the video packets (three catalog queries), computes
the labels of the kept frames (every 6th UmeTrack / 12th SHOW3D frame: the 5 fps pool) on CPU, and decodes each camera
with its own torchcodec NVDEC decoder created on that thread. Images the validity rules reject are never decoded. Each
kept image is letterboxed to the 640x480 net frame on the GPU, then

- DetNet: pooled 4x4 to 160x120 and stored as uint8 (the rounded 4x4 mean) with its per-hand targets;
- KeyNet: cut into 96x96 crops (positives and presence negatives, with their keypoint inputs) and stored in uint8.

Producers hand chunks to the main thread through a bounded queue; the main thread moves them into two GPU pools (one per
network) and draws uniform random batches without replacement. All CUDA work runs on the default stream, so the pools
need no cross-stream bookkeeping; producers synchronise before they publish a chunk.

A side that is not consumed cannot block the other: a full pool takes new samples by overwriting random old ones (counted
in ``StreamStats.overwritten``), so memory stays bounded and the queue always drains. Producer exceptions are re-raised in
the main thread by the next call.
"""

import dataclasses
import math
import queue
import random
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
import torch.nn.functional as F
from beartype.roar import BeartypeException
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_video import CatalogVideo, read_catalog_videos
from simplecv.catalog_video_codec import wrap_mp4
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from handtrack.data.batches import CropKind, DetNetBatch, KeyNetBatch
from handtrack.data.catalog import (
    CATALOG_URL,
    TIMELINE,
    DatasetLayout,
    DatasetName,
    HandTimeline,
    SegmentInfo,
    SplitName,
    is_show3d,
    layout_for,
    list_segments,
    read_hand_timeline,
    read_rig,
    read_statics,
    select_split,
)
from handtrack.data.segment_labels import HandProjection, KeypointPriors, SegmentLabels, keypoint_priors, segment_labels
from handtrack.geometry.camera import CameraRig
from handtrack.geometry.letterbox import NET_HEIGHT, NET_WIDTH, Letterbox
from handtrack.labels.crops import (
    BOX_ENLARGE,
    CROP_SIZE,
    CropJitter,
    apply_affine,
    boundary_occlusion,
    count_inside_crop,
    crop_boxes,
    crop_from_net,
    cut_crops,
    sample_jitter,
    scale_intensity,
)
from handtrack.labels.heatmaps import render_distance, render_heatmaps
from handtrack.labels.keypoint_input import add_input_noise, keypoint_input, relative_distances
from handtrack.labels.validity import HandLabel, classify_visibility
from handtrack.train.source import DetNetValidation, KeyNetValidation

NetChoice: TypeAlias = Literal["detnet", "keynet", "both"]
DECODE_CHUNK_PIXELS: int = 10_000_000
"""Pixels per ``get_frames_at`` call and per queued chunk: 32 UmeTrack frames (636x480), 7 SHOW3D frames (1024x1280).
Bounds each producer's transient memory (RGB output, resize in float) to about 0.1 GB."""
MIN_BOX_RADIUS: float = 8.0
"""Our floor on a crop box's circle radius (net px), so a degenerate circle still gives a valid crop."""


@dataclass(frozen=True, slots=True)
class KeyNetAugment:
    """KeyNet's crop jitter, keypoint-input recipe, image augmentation and presence negatives (all our choices)."""

    max_rotation: float = 0.5236
    """Crop rotation drawn from ±this (radians; 30°), applied to image and keypoints together."""
    scale_range: tuple[float, float] = (0.9, 1.25)
    """Crop side multiplier (> 1 zooms out)."""
    max_shift: float = 0.1
    """Box centre shift as a fraction of the side, per axis."""
    zero_input_probability: float = 0.2
    """Keypoint input set to all zeros (an untracked hand, as after a DetNet detection)."""
    stale_input_probability: float = 0.1
    """Keypoint input from the pose 20 tracker steps earlier (the paper's 10%); otherwise the extrapolated pose."""
    uv_noise_std: float = 0.03
    """Gaussian noise on the input's normalised u, v (0.03 ≈ 2.9 crop px)."""
    d_noise_std: float = 0.05
    """Gaussian noise on the input's d = d_rel / 130 mm (0.05 ≈ 6.5 mm)."""
    occlusion_probability: float = 0.2
    """A rectangle at a crop border set to zero (a hand partly out of the frame; its keypoints stay labelled)."""
    occlusion_max_fraction: float = 0.4
    """Depth of that rectangle, up to this fraction of the crop side."""
    intensity_range: tuple[float, float] = (0.6, 1.4)
    """Random intensity scaling of the crop, as DetNet's."""
    drift_probability: float = 0.1
    """Per present hand: a DRIFT negative, its box shifted 1.0-1.6 box sides off the hand."""
    drift_shift_range: tuple[float, float] = (1.0, 1.6)
    other_hand_probability: float = 0.1
    """Per hand slot whose other hand is present: an OTHER_HAND negative (the other hand's box, this slot's mirroring)."""
    edge_probability: float = 0.25
    """Per absent hand with a tracker prior: an EDGE negative, the prior's box moved to overlap the image edge."""
    edge_overlap_range: tuple[float, float] = (0.2, 0.6)
    """How much of that box (as a fraction of its side) overlaps the image."""
    background_probability: float = 0.05
    """Per absent hand: a BACKGROUND negative, a random box anywhere in the image."""
    background_side_range: tuple[float, float] = (48.0, 240.0)
    """Side of a background box in net px."""


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """How to build a ``CatalogStream`` (tyro-friendly)."""

    datasets: tuple[DatasetName, ...] = ("dataforge-umetrack", "dataforge-show3d")
    split: SplitName = "train"
    """Our split (``catalog.select_split``); ignored when ``segment_ids`` is set."""
    segment_ids: tuple[str, ...] = ()
    """A fixed segment list (e.g. 1-2 segments to overfit); looked up in ``datasets``."""
    max_segments: int | None = None
    """Cap on segments per epoch (the first N of each epoch's shuffle), for short runs."""
    nets: NetChoice = "detnet"
    producers: int = 4
    """Decode threads, each with its own NVDEC decoder per camera stream (created on that thread)."""
    fetchers: int = 4
    """Catalog threads that read and label the next segments while the producers decode (queries are latency-bound)."""
    prefetch_segments: int = 8
    """Bound on fetched segments waiting for a producer (compressed video: about 5 MB UmeTrack, 30 MB SHOW3D each)."""
    detnet_batch_size: int = 256
    keynet_batch_size: int = 256
    detnet_buffer: int = 65_536
    """Pooled frames held on the GPU (uint8 160x120 plus labels: 19.6 kB each, 1.29 GB at 65,536)."""
    keynet_buffer: int = 65_536
    """Crops held on the GPU (uint8 96x96 plus labels: 9.8 kB each, 0.64 GB at 65,536)."""
    min_fill: float = 0.25
    """Draws start once a pool holds this fraction of its capacity (or the epoch is fully produced)."""
    intensity_range: tuple[float, float] = (0.6, 1.4)
    """DetNet's random intensity scaling, applied to the pooled frames (our choice)."""
    keynet: KeyNetAugment = field(default_factory=KeyNetAugment)
    seed: int = 0
    device: str = "cuda"
    catalog_url: str = CATALOG_URL
    queue_chunks: int = 64
    """Bound on chunks waiting between producers and the main thread."""
    validation: bool = False
    """An evaluation set instead of a training stream: no augmentation (GT crop boxes, no jitter, no input noise or
    zero inputs, no occlusion or intensity scaling), built once on the first ``start_epoch`` from ``max_segments``
    segments of one seeded shuffle, then replayed in the same order every epoch."""
    validation_samples: int = 5120
    """Samples per network kept in the evaluation set (a seeded subset spanning all its segments)."""
    max_skipped_fraction: float = 0.02
    """A segment that fails twice (fetch, label or decode) is skipped for the rest of the run; more than this fraction of
    the segments (at least 3) failing means something other than bad data, and the stream stops with the failures listed."""


# --- samples and pools -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetNetSamples:
    """Stored DetNet samples: pooled net frames and per-hand targets (slot 0 = left)."""

    pooled: UInt8[Tensor, "n 120 160"]
    """The rounded 4x4 mean of the net frame (0-255), not yet augmented; uint8 halves fp16's memory for an error of at most 0.5/255."""
    circle: Float32[Tensor, "n 2 3"]
    """(cx / 640, cy / 480, r / 640); zero where the hand is not present."""
    presence: Float32[Tensor, "n 2"]
    circle_mask: Bool[Tensor, "n 2"]
    presence_mask: Bool[Tensor, "n 2"]
    dataset: Int64[Tensor, "n"]
    points: Float32[Tensor, "n 2 21 2"]
    """Unaugmented GT keypoints in the net frame (validation metadata)."""
    in_front: Bool[Tensor, "n 2 21"]
    """GT keypoints in front of the camera; all False for an absent hand."""
    camera: Int64[Tensor, "n"]
    """Camera id, unique across datasets (``DatasetLayout.camera_offset`` + camera index)."""


@dataclass(frozen=True, slots=True)
class KeyNetSamples:
    """Stored KeyNet crops (right hands mirrored) with their targets; heatmaps are rendered at draw time."""

    crops: UInt8[Tensor, "n 96 96"]
    points_crop: Float32[Tensor, "n 21 2"]
    """Ground-truth keypoints in crop pixels; zero for negatives."""
    d_rel_mm: Float32[Tensor, "n 21"]
    """Ground-truth d_rel; zero for negatives."""
    keypoints: Float32[Tensor, "n 63"]
    """The keypoint input, noise included; all zeros for an untracked hand."""
    presence: Float32[Tensor, "n"]
    kind: Int64[Tensor, "n"]
    dataset: Int64[Tensor, "n"]
    crop_from_net: Float32[Tensor, "n 3 3"]
    """The affine the crop was cut with, mirror included (validation metadata)."""


def empty_detnet_samples(capacity: int, device: torch.device) -> DetNetSamples:
    """Zeroed storage for ``capacity`` DetNet samples."""
    return DetNetSamples(
        pooled=torch.zeros((capacity, NET_HEIGHT // 4, NET_WIDTH // 4), dtype=torch.uint8, device=device),
        circle=torch.zeros((capacity, 2, 3), device=device),
        presence=torch.zeros((capacity, 2), device=device),
        circle_mask=torch.zeros((capacity, 2), dtype=torch.bool, device=device),
        presence_mask=torch.zeros((capacity, 2), dtype=torch.bool, device=device),
        dataset=torch.zeros(capacity, dtype=torch.int64, device=device),
        points=torch.zeros((capacity, 2, 21, 2), device=device),
        in_front=torch.zeros((capacity, 2, 21), dtype=torch.bool, device=device),
        camera=torch.zeros(capacity, dtype=torch.int64, device=device),
    )


def empty_keynet_samples(capacity: int, device: torch.device) -> KeyNetSamples:
    """Zeroed storage for ``capacity`` KeyNet samples."""
    return KeyNetSamples(
        crops=torch.zeros((capacity, CROP_SIZE, CROP_SIZE), dtype=torch.uint8, device=device),
        points_crop=torch.zeros((capacity, 21, 2), device=device),
        d_rel_mm=torch.zeros((capacity, 21), device=device),
        keypoints=torch.zeros((capacity, 63), device=device),
        presence=torch.zeros(capacity, device=device),
        kind=torch.zeros(capacity, dtype=torch.int64, device=device),
        dataset=torch.zeros(capacity, dtype=torch.int64, device=device),
        crop_from_net=torch.zeros((capacity, 3, 3), device=device),
    )


def sample_count(samples: DetNetSamples | KeyNetSamples) -> int:
    return int(samples.dataset.shape[0])


def select_samples[SampleT: (DetNetSamples, KeyNetSamples)](samples: SampleT, index: Int64[Tensor, "m"] | Bool[Tensor, "n"]) -> SampleT:
    return type(samples)(**{f.name: getattr(samples, f.name)[index] for f in dataclasses.fields(samples)})


def concat_samples[SampleT: (DetNetSamples, KeyNetSamples)](parts: list[SampleT]) -> SampleT:
    return type(parts[0])(**{f.name: torch.cat([getattr(part, f.name) for part in parts]) for f in dataclasses.fields(parts[0])})


class SamplePool[SampleT: (DetNetSamples, KeyNetSamples)]:
    """A fixed-capacity GPU shuffle buffer: uniform random draws without replacement; a full pool overwrites random slots."""

    def __init__(self, storage: SampleT, generator: torch.Generator) -> None:
        self.storage: SampleT = storage
        """Preallocated rows (``empty_detnet_samples`` / ``empty_keynet_samples``); the first ``count`` hold samples."""
        self.capacity: int = sample_count(storage)
        self.count: int = 0
        self.overwritten: int = 0
        self._generator: torch.Generator = generator
        self._device: torch.device = storage.dataset.device

    def _write(self, slots: Int64[Tensor, "m"], samples: SampleT) -> None:
        for f in dataclasses.fields(self.storage):
            getattr(self.storage, f.name)[slots] = getattr(samples, f.name)

    def add(self, samples: SampleT) -> None:
        incoming: int = sample_count(samples)
        free: int = min(incoming, self.capacity - self.count)
        if free > 0:
            self._write(torch.arange(self.count, self.count + free, device=self._device), select_samples(samples, torch.arange(free, device=self._device)))
            self.count += free
        rest: int = incoming - free
        if rest > 0:
            slots: Int64[Tensor, "m"] = torch.randint(0, self.capacity, (rest,), generator=self._generator, device=self._device)
            self._write(slots, select_samples(samples, torch.arange(free, incoming, device=self._device)))
            self.overwritten += rest

    def draw(self, n: int) -> SampleT:
        """n samples chosen uniformly without replacement; the pool shrinks by n (the tail fills the holes)."""
        if n > self.count:
            raise ValueError(f"cannot draw {n} of {self.count} samples")
        chosen: Int64[Tensor, "n"] = torch.randperm(self.count, generator=self._generator, device=self._device)[:n]
        drawn: SampleT = select_samples(self.storage, chosen)
        boundary: int = self.count - n
        tail: Int64[Tensor, "n"] = torch.arange(boundary, self.count, device=self._device)
        movers: Int64[Tensor, "m"] = tail[~torch.isin(tail, chosen)]
        holes: Int64[Tensor, "m"] = chosen[chosen < boundary]
        if holes.numel():
            self._write(holes, select_samples(self.storage, movers))
        self.count = boundary
        return drawn

    def clear(self) -> None:
        self.count = 0


# --- building samples from one chunk of letterboxed images ------------------------------------------------------


def detnet_samples(
    net: UInt8[Tensor, "m 480 640"],
    hand_label: Int64[Tensor, "m 2"],
    circles: Float32[Tensor, "m 2 3"],
    truth: "ImageHands",
    dataset: int,
    camera: int,
) -> DetNetSamples:
    """Pool 4x4 and attach DetNet's targets: circle and presence for present hands, no loss at all for partial ones."""
    pooled: UInt8[Tensor, "m 120 160"] = F.avg_pool2d(net[:, None].float(), 4)[:, 0].round().to(torch.uint8)
    present: Bool[Tensor, "m 2"] = hand_label == int(HandLabel.PRESENT)
    scale: Float32[Tensor, "3"] = torch.tensor([NET_WIDTH, NET_HEIGHT, NET_WIDTH], dtype=torch.float32, device=net.device)
    circle: Float32[Tensor, "m 2 3"] = torch.where(present[..., None], circles / scale, torch.zeros_like(circles))
    return DetNetSamples(
        pooled=pooled,
        circle=circle,
        presence=present.float(),
        circle_mask=present,
        presence_mask=hand_label != int(HandLabel.PARTIAL),
        dataset=torch.full((net.shape[0],), dataset, dtype=torch.int64, device=net.device),
        points=torch.nan_to_num(truth.net_xy, nan=0.0),
        in_front=truth.in_front & (hand_label != int(HandLabel.ABSENT))[..., None],
        camera=torch.full((net.shape[0],), camera, dtype=torch.int64, device=net.device),
    )


@dataclass(frozen=True, slots=True)
class ImageHands:
    """Both hands' keypoints in m images of one camera, in the net frame and the camera (slot 0 = left)."""

    net_xy: Float32[Tensor, "m 2 21 2"]
    points_cam: Float32[Tensor, "m 2 21 3"]
    in_front: Bool[Tensor, "m 2 21"]
    valid: Bool[Tensor, "m 2"]


def image_hands(projection: HandProjection, valid: Bool[Tensor, "k 2"], rows: Int64[Tensor, "m"], camera: int) -> ImageHands:
    return ImageHands(
        net_xy=projection.net_xy[rows, camera],
        points_cam=projection.points_cam[rows, camera],
        in_front=projection.in_front[rows, camera],
        valid=valid[rows],
    )


def bounding_circles(points: Float32[Tensor, "n 21 2"], valid: Bool[Tensor, "n 21"]) -> Float32[Tensor, "n 3"]:
    """A cheap enclosing circle (bounding-box centre, farthest valid point) for tracker-prior boxes; NaN without a valid point."""
    big: float = 1e9
    low: Float32[Tensor, "n 2"] = torch.where(valid[..., None], points, torch.full_like(points, big)).amin(dim=1)
    high: Float32[Tensor, "n 2"] = torch.where(valid[..., None], points, torch.full_like(points, -big)).amax(dim=1)
    centre: Float32[Tensor, "n 2"] = (low + high) * 0.5
    reach: Float32[Tensor, "n 21"] = torch.where(valid, (points - centre[:, None]).norm(dim=-1), torch.zeros_like(valid, dtype=torch.float32))
    circle: Float32[Tensor, "n 3"] = torch.cat([centre, reach.amax(dim=1, keepdim=True)], dim=-1)
    return torch.where(valid.any(dim=1)[:, None], circle, torch.full_like(circle, torch.nan))


def finite_circles(circles: Float32[Tensor, "n 3"]) -> Float32[Tensor, "n 3"]:
    """Replace unusable circles by a placeholder and floor the radius (callers never keep crops built from a placeholder)."""
    placeholder: Float32[Tensor, "3"] = torch.tensor([NET_WIDTH / 2, NET_HEIGHT / 2, 32.0], device=circles.device)
    usable: Bool[Tensor, "n"] = torch.isfinite(circles).all(dim=-1)
    safe: Float32[Tensor, "n 3"] = torch.where(usable[:, None], circles, placeholder.expand_as(circles))
    return torch.cat([safe[:, :2], safe[:, 2:].clamp_min(MIN_BOX_RADIUS)], dim=-1)


def box_at(centre: Float32[Tensor, "n 2"], side: Float32[Tensor, "n"]) -> Float32[Tensor, "n 4"]:
    half: Float32[Tensor, "n 1"] = side[:, None] * 0.5
    return torch.cat([centre - half, centre + half], dim=-1)


def uniform(generator: torch.Generator, n: int, bounds: tuple[float, float], device: torch.device) -> Float32[Tensor, "n"]:
    return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, generator=generator, device=device)


def keynet_samples(
    net: UInt8[Tensor, "m 480 640"],
    truth: ImageHands,
    hand_label: Int64[Tensor, "m 2"],
    circles: Float32[Tensor, "m 2 3"],
    extrapolated: ImageHands,
    stale: ImageHands,
    phi: float,
    dataset: int,
    augment: KeyNetAugment,
    generator: torch.Generator,
) -> KeyNetSamples:
    """KeyNet crops for every present hand plus the presence negatives, each with its keypoint input.

    Positives: the ground truth's enclosing circle, squared, +20%, then jittered. Negatives (``CropKind``) are kept only
    when none of the hand's keypoints is inside the crop (DetNet's absent rule); positives only when at least 17 are.
    Crops with 1-16 inside would get no loss at all, so they are not stored. The keypoint input of a positive is its
    prior mapped through the crop itself; a negative's is the prior mapped through the box the prior itself would give
    (a tracker's crop is centred on its own guess), so the input never tells whether the hand is there.
    """
    device: torch.device = net.device
    images: int = net.shape[0]
    slots: Int64[Tensor, "s"] = torch.arange(2 * images, device=device)
    label: Int64[Tensor, "s"] = hand_label.reshape(-1)
    present: Bool[Tensor, "s"] = label == int(HandLabel.PRESENT)
    absent: Bool[Tensor, "s"] = label == int(HandLabel.ABSENT)
    other_present: Bool[Tensor, "s"] = hand_label.flip(1).reshape(-1) == int(HandLabel.PRESENT)
    prior_xy: Float32[Tensor, "s 21 2"] = extrapolated.net_xy.reshape(-1, 21, 2)
    prior_front: Bool[Tensor, "s 21"] = extrapolated.in_front.reshape(-1, 21) & extrapolated.valid.reshape(-1, 1)
    prior_circles: Float32[Tensor, "s 3"] = bounding_circles(prior_xy, prior_front)
    prior_usable: Bool[Tensor, "s"] = torch.isfinite(prior_circles).all(dim=-1)
    draws: Float32[Tensor, "s 4"] = torch.rand((2 * images, 4), generator=generator, device=device)
    candidates: list[tuple[Bool[Tensor, "s"], CropKind]] = [
        (present, CropKind.POSITIVE),
        (present & (draws[:, 0] < augment.drift_probability), CropKind.DRIFT),
        (other_present & (label != int(HandLabel.PARTIAL)) & (draws[:, 1] < augment.other_hand_probability), CropKind.OTHER_HAND),
        (absent & prior_usable & (draws[:, 2] < augment.edge_probability), CropKind.EDGE),
        (absent & (draws[:, 3] < augment.background_probability), CropKind.BACKGROUND),
    ]
    slot: Int64[Tensor, "n"] = torch.cat([slots[mask] for mask, _ in candidates])
    kind: Int64[Tensor, "n"] = torch.cat([torch.full((int(mask.sum()),), int(k), dtype=torch.int64, device=device) for mask, k in candidates])
    n: int = slot.numel()
    if n == 0:
        return empty_keynet_samples(0, device)
    image: Int64[Tensor, "n"] = slot // 2
    hand: Int64[Tensor, "n"] = slot % 2
    mirror: Bool[Tensor, "n"] = hand == 1
    own_circle: Float32[Tensor, "n 3"] = finite_circles(circles.reshape(-1, 3)[slot])
    other_circle: Float32[Tensor, "n 3"] = finite_circles(circles.flip(1).reshape(-1, 3)[slot])
    prior_circle: Float32[Tensor, "n 3"] = finite_circles(prior_circles[slot])
    is_kind: dict[CropKind, Bool[Tensor, "n"]] = {k: kind == int(k) for k in CropKind}
    # The box each crop is cut from (before jitter).
    base_circle: Float32[Tensor, "n 3"] = torch.where(is_kind[CropKind.OTHER_HAND][:, None], other_circle, torch.where(is_kind[CropKind.EDGE][:, None], prior_circle, own_circle))
    side: Float32[Tensor, "n"] = 2.0 * base_circle[:, 2] * BOX_ENLARGE
    centre: Float32[Tensor, "n 2"] = base_circle[:, :2]
    angle: Float32[Tensor, "n"] = uniform(generator, n, (0.0, 2.0 * math.pi), device)
    drift: Float32[Tensor, "n"] = side * uniform(generator, n, augment.drift_shift_range, device)
    drifted: Float32[Tensor, "n 2"] = centre + drift[:, None] * torch.stack([angle.cos(), angle.sin()], dim=-1)
    overlap: Float32[Tensor, "n"] = side * (uniform(generator, n, augment.edge_overlap_range, device) - 0.5)
    low: Float32[Tensor, "n 2"] = (overlap - 0.5)[:, None].expand(-1, 2)
    high: Float32[Tensor, "n 2"] = torch.stack([NET_WIDTH - 0.5 - overlap, NET_HEIGHT - 0.5 - overlap], dim=-1)
    at_edge: Float32[Tensor, "n 2"] = torch.maximum(torch.minimum(centre, high), low)
    background_side: Float32[Tensor, "n"] = uniform(generator, n, augment.background_side_range, device)
    background_centre: Float32[Tensor, "n 2"] = torch.rand((n, 2), generator=generator, device=device) * torch.tensor([NET_WIDTH, NET_HEIGHT], device=device) - 0.5
    centre = torch.where(is_kind[CropKind.DRIFT][:, None], drifted, torch.where(is_kind[CropKind.EDGE][:, None], at_edge, centre))
    centre = torch.where(is_kind[CropKind.BACKGROUND][:, None], background_centre, centre)
    side = torch.where(is_kind[CropKind.BACKGROUND], background_side, side)
    jitter: CropJitter = sample_jitter(n, generator, device, augment.max_rotation, augment.scale_range, augment.max_shift)
    affine: Float32[Tensor, "n 3 3"] = crop_from_net(box_at(centre, side), mirror, jitter)
    # Ground truth in the crop, and DetNet's rule on the keypoints inside it.
    truth_xy: Float32[Tensor, "n 21 2"] = apply_affine(affine, truth.net_xy.reshape(-1, 21, 2)[slot])
    truth_front: Bool[Tensor, "n 21"] = truth.in_front.reshape(-1, 21)[slot] & truth.valid.reshape(-1, 1)[slot]
    crop_label: Int64[Tensor, "n"] = classify_visibility(count_inside_crop(truth_xy, truth_front))
    positive: Bool[Tensor, "n"] = is_kind[CropKind.POSITIVE]
    keep: Bool[Tensor, "n"] = torch.where(positive, crop_label == int(HandLabel.PRESENT), crop_label == int(HandLabel.ABSENT))
    # The keypoint input: extrapolated (or stale) prior, noisy, or all zeros.
    reference: Float32[Tensor, "n 3 3"] = torch.where(
        positive[:, None, None],
        affine,
        torch.where(
            is_kind[CropKind.DRIFT][:, None, None],
            crop_from_net(crop_boxes(own_circle), mirror, jitter),
            crop_from_net(crop_boxes(prior_circle), mirror, jitter),
        ),
    )
    source_draw: Float32[Tensor, "n"] = torch.rand(n, generator=generator, device=device)
    extrapolated_valid: Bool[Tensor, "n"] = extrapolated.valid.reshape(-1)[slot]
    stale_valid: Bool[Tensor, "n"] = stale.valid.reshape(-1)[slot]
    zero: Bool[Tensor, "n"] = source_draw < augment.zero_input_probability
    use_stale: Bool[Tensor, "n"] = ~zero & (source_draw < augment.zero_input_probability + augment.stale_input_probability) & stale_valid
    use_extrapolated: Bool[Tensor, "n"] = ~zero & ~use_stale & extrapolated_valid
    source_xy: Float32[Tensor, "n 21 2"] = torch.where(use_stale[:, None, None], stale.net_xy.reshape(-1, 21, 2)[slot], prior_xy[slot])
    source_cam: Float32[Tensor, "n 21 3"] = torch.where(use_stale[:, None, None], stale.points_cam.reshape(-1, 21, 3)[slot], extrapolated.points_cam.reshape(-1, 21, 3)[slot])
    phi_n: Float32[Tensor, "n"] = torch.full((n,), phi, dtype=torch.float32, device=device)
    noisy: Float32[Tensor, "n 63"] = add_input_noise(
        keypoint_input(apply_affine(reference, source_xy), relative_distances(source_cam, phi_n)), generator, augment.uv_noise_std, augment.d_noise_std
    )
    keypoints: Float32[Tensor, "n 63"] = torch.where((use_stale | use_extrapolated)[:, None], noisy, torch.zeros_like(noisy))
    truth_d: Float32[Tensor, "n 21"] = relative_distances(truth.points_cam.reshape(-1, 21, 3)[slot], phi_n)
    crops: Float32[Tensor, "q 1 96 96"] = cut_crops(net, image[keep], affine[keep])
    return KeyNetSamples(
        crops=(crops[:, 0] * 255.0).round().to(torch.uint8),
        points_crop=torch.where(positive[keep, None, None], truth_xy[keep], torch.zeros_like(truth_xy[keep])),
        d_rel_mm=torch.where(positive[keep, None], truth_d[keep], torch.zeros_like(truth_d[keep])),
        keypoints=keypoints[keep],
        presence=positive[keep].float(),
        kind=kind[keep],
        dataset=torch.full((int(keep.sum()),), dataset, dtype=torch.int64, device=device),
        crop_from_net=affine[keep],
    )


# --- the stream ------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class StreamStats:
    """Counters since the stream was built (the rate tool reads them)."""

    segments: int = 0
    images_considered: int = 0
    """(kept frame, camera) pairs of the 5 fps pool."""
    images_missing: int = 0
    """Pool images whose camera has no video frame at that time."""
    images_invalid: int = 0
    """Pool images the validity rules dropped."""
    images_decoded: int = 0
    detnet_samples: int = 0
    keynet_samples: int = 0
    keynet_kinds: list[int] = field(default_factory=lambda: [0] * len(CropKind))
    query_s: float = 0.0
    label_s: float = 0.0
    decode_s: float = 0.0
    """Producer seconds in decoding, letterboxing and sample building (all producers summed)."""
    wait_s: float = 0.0
    """Main-thread seconds blocked waiting for chunks."""
    retried_segments: int = 0
    skipped_segments: int = 0
    failures: list[str] = field(default_factory=list)
    """One line per failed attempt: dataset, segment, stage (camera), attempt and error (also printed to stderr)."""


@dataclass(frozen=True, slots=True)
class _SegmentWork:
    """A fetched and labelled segment, waiting for a producer to decode it."""

    generation: int
    position: int
    """Index in the epoch's segment order."""
    info: SegmentInfo
    dataset_index: int
    letterboxes: tuple[Letterbox, ...]
    labels: SegmentLabels
    priors: KeypointPriors | None
    hand_scale: float
    times: Int64[ndarray, "k"]
    """``video_time`` of the labelled rows."""
    videos: tuple[CatalogVideo, ...]


@dataclass(frozen=True, slots=True)
class _Chunk:
    generation: int
    order: tuple[int, int, int]
    """(segment position, camera, first row): the deterministic order of an evaluation set."""
    detnet: DetNetSamples | None
    keynet: KeyNetSamples | None


class _Stale(Exception):
    """The epoch changed or the stream closed while a producer was working."""


class SegmentFailure(Exception):
    """A per-segment failure (bad or corrupted data), retried once and then skipped; the message names the camera."""


@runtime_checkable
class FrameBatchLike(Protocol):
    data: Tensor


@runtime_checkable
class FrameDecoder(Protocol):
    """What the producers need from a decoder (torchcodec's ``VideoDecoder``)."""

    def get_frames_at(self, indices: list[int]) -> FrameBatchLike: ...


@dataclass(frozen=True, slots=True)
class SegmentData:
    """One segment as read from the catalog: rig, letterboxes, hand timeline and the cameras' encoded video."""

    rig: CameraRig
    letterboxes: tuple[Letterbox, ...]
    timeline: HandTimeline
    videos: tuple[CatalogVideo, ...]


SegmentReader: TypeAlias = Callable[[SegmentInfo], SegmentData]
DecoderOpener: TypeAlias = Callable[[CatalogVideo, int, torch.device], FrameDecoder]


def open_nvdec_decoder(video: CatalogVideo, fps: int, device: torch.device) -> FrameDecoder:
    """torchcodec on NVDEC over the segment's packets muxed into one MP4 (create it on the thread that uses it)."""
    return VideoDecoder(wrap_mp4(video.samples, video.keyframes, fps, codec=video.codec), device=device, seek_mode="exact", num_ffmpeg_threads=0)


def is_fatal(error: BaseException) -> bool:
    """Failures of the stream machinery or of our own code (type violations, CUDA, memory): never retried or skipped."""
    return not isinstance(error, Exception) or isinstance(error, BeartypeException | torch.OutOfMemoryError | torch.AcceleratorError | MemoryError | _Stale)


def evaluation_augment(augment: KeyNetAugment) -> KeyNetAugment:
    """The evaluation recipe: GT boxes without jitter, the extrapolated prior without noise (zeros only when there is
    none), no occlusion or intensity scaling; the presence negatives stay (seeded) so presence can be scored."""
    return dataclasses.replace(
        augment,
        max_rotation=0.0,
        scale_range=(1.0, 1.0),
        max_shift=0.0,
        zero_input_probability=0.0,
        stale_input_probability=0.0,
        uv_noise_std=0.0,
        d_noise_std=0.0,
        occlusion_probability=0.0,
        intensity_range=(1.0, 1.0),
    )


def resolve_segments(config: StreamConfig, client: rr.catalog.CatalogClient) -> tuple[SegmentInfo, ...]:
    """The configured segments: the fixed list if given, else ``config.split`` of every dataset."""
    listed: list[SegmentInfo] = [info for name in config.datasets for info in list_segments(client.get_dataset(name), name)]
    if config.segment_ids:
        by_id: dict[str, SegmentInfo] = {info.segment_id: info for info in listed}
        missing: list[str] = [segment for segment in config.segment_ids if segment not in by_id]
        if missing:
            raise ValueError(f"segments not found in {config.datasets}: {missing}")
        return tuple(by_id[segment] for segment in config.segment_ids)
    by_dataset: dict[str, list[SegmentInfo]] = {}
    for info in listed:
        by_dataset.setdefault(info.dataset, []).append(info)
    return tuple(info for name in config.datasets for info in select_split(by_dataset.get(name, []), config.split))


class CatalogStream:
    """DetNet and/or KeyNet batches from one catalog decode (a ``batches.BatchSource``).

    Build it, call ``start_epoch(e)``, then draw with ``next_detnet_batch()`` / ``next_keynet_batch()`` until they return
    None; ``close()`` (or a ``with`` block) stops the producers.
    """

    def __init__(
        self,
        config: StreamConfig,
        segments: tuple[SegmentInfo, ...] | None = None,
        read_segment: SegmentReader | None = None,
        open_decoder: DecoderOpener | None = None,
    ) -> None:
        """``read_segment`` and ``open_decoder`` replace the catalog and NVDEC (tests); then ``segments`` must be given."""
        if config.producers < 1 or config.fetchers < 1:
            raise ValueError("need at least one producer and one fetcher")
        self.config: StreamConfig = config
        self.device: torch.device = torch.device(config.device)
        if open_decoder is None and (self.device.type != "cuda" or not torch.cuda.is_available()):
            raise RuntimeError("the catalog stream decodes with NVDEC and needs a CUDA device")
        self._read_segment: SegmentReader = read_segment if read_segment is not None else self._read_from_catalog
        self._open_decoder: DecoderOpener = open_decoder if open_decoder is not None else open_nvdec_decoder
        self._local: threading.local = threading.local()
        self._skipped: set[str] = set()
        self._detnet_on: bool = config.nets in ("detnet", "both")
        self._keynet_on: bool = config.nets in ("keynet", "both")
        self.segments: tuple[SegmentInfo, ...] = segments if segments is not None else resolve_segments(config, rr.catalog.CatalogClient(config.catalog_url))
        if not self.segments:
            raise ValueError("no segments to stream")
        self.dataset_names: tuple[str, ...] = tuple(dict.fromkeys(info.dataset for info in self.segments))
        self._generator: torch.Generator = torch.Generator(device=self.device)
        self._generator.manual_seed(config.seed)
        self._augment: KeyNetAugment = evaluation_augment(config.keynet) if config.validation else config.keynet
        training: bool = not config.validation
        self._detnet_pool: SamplePool[DetNetSamples] | None = (
            SamplePool(empty_detnet_samples(config.detnet_buffer, self.device), self._generator) if self._detnet_on and training else None
        )
        self._keynet_pool: SamplePool[KeyNetSamples] | None = (
            SamplePool(empty_keynet_samples(config.keynet_buffer, self.device), self._generator) if self._keynet_on and training else None
        )
        self._collected: list[_Chunk] = []
        """Evaluation mode: every chunk of the one build pass."""
        self._evaluation: tuple[DetNetSamples | None, KeyNetSamples | None] | None = None
        self._cursor: list[int] = [0, 0]
        self._last_detnet: DetNetSamples | None = None
        self._last_keynet: KeyNetSamples | None = None
        self.stats: StreamStats = StreamStats()
        self._queue: queue.Queue[_Chunk] = queue.Queue(maxsize=config.queue_chunks)
        self._work: queue.Queue[_SegmentWork] = queue.Queue(maxsize=config.prefetch_segments)
        self._cond: threading.Condition = threading.Condition()
        self._stop: threading.Event = threading.Event()
        self._error: BaseException | None = None
        self._error_segment: str = ""
        """The segment a producer was working on when the stream failed."""
        self._generation: int = -1
        self._epoch_segments: list[SegmentInfo] = []
        self._next: int = 0
        self._done: int = 0
        # torch initialises its CUDA linalg backend lazily, and the first call must not race between producer
        # threads ("lazy wrapper should be called at most once"): make that first call here.
        torch.linalg.inv(torch.eye(3, device=self.device)[None])
        self._threads: list[threading.Thread] = [
            threading.Thread(target=self._fetch, args=(worker,), name=f"catalog-fetcher-{worker}", daemon=True) for worker in range(config.fetchers)
        ] + [threading.Thread(target=self._produce, args=(worker,), name=f"catalog-producer-{worker}", daemon=True) for worker in range(config.producers)]
        for thread in self._threads:
            thread.start()

    # ---- trainer API

    def start_epoch(self, epoch: int) -> None:
        self._raise_if_failed()
        if self.config.validation:
            if self._evaluation is None:
                self._build_evaluation()
            self._cursor = [0, 0]
            return
        self._begin_pass(epoch)
        for pool in (self._detnet_pool, self._keynet_pool):
            if pool is not None:
                pool.clear()

    def _begin_pass(self, epoch: int) -> None:
        order: list[SegmentInfo] = list(self.segments)
        random.Random(self.config.seed + epoch).shuffle(order)
        if self.config.max_segments is not None:
            order = order[: self.config.max_segments]
        with self._cond:
            self._generation += 1
            self._epoch_segments = order
            self._next = 0
            self._done = 0
            self._cond.notify_all()
        self._discard_queue()
        self._discard_work()

    def _build_evaluation(self) -> None:
        """One pass over a fixed seeded subset of the segments, kept in a deterministic order."""
        self._begin_pass(0)
        while True:
            produced: bool = self._epoch_produced()
            self._drain(None)
            if produced:
                break
            self._drain(0.5)
        chunks: list[_Chunk] = sorted(self._collected, key=lambda chunk: chunk.order)
        self._collected = []
        order: torch.Generator = torch.Generator().manual_seed(self.config.seed)
        detnet: DetNetSamples | None = None
        keynet: KeyNetSamples | None = None
        detnet_parts: list[DetNetSamples] = [chunk.detnet for chunk in chunks if chunk.detnet is not None]
        keynet_parts: list[KeyNetSamples] = [chunk.keynet for chunk in chunks if chunk.keynet is not None]
        if detnet_parts:
            detnet = concat_samples(detnet_parts)
            detnet = select_samples(detnet, torch.randperm(sample_count(detnet), generator=order)[: self.config.validation_samples].to(self.device))
        if keynet_parts:
            keynet = concat_samples(keynet_parts)
            keynet = select_samples(keynet, torch.randperm(sample_count(keynet), generator=order)[: self.config.validation_samples].to(self.device))
        self._evaluation = (detnet, keynet)

    def _next_evaluation[SampleT: (DetNetSamples, KeyNetSamples)](self, samples: SampleT | None, slot: int, batch: int) -> SampleT | None:
        start: int = self._cursor[slot]
        if samples is None or start >= sample_count(samples):
            return None
        self._cursor[slot] = start + batch
        return select_samples(samples, torch.arange(start, min(start + batch, sample_count(samples)), device=self.device))

    def next_detnet_batch(self) -> DetNetBatch | None:
        self._check("DetNet", self._detnet_on)
        samples: DetNetSamples | None
        if self._evaluation is not None:
            samples = self._next_evaluation(self._evaluation[0], 0, self.config.detnet_batch_size)
        else:
            samples = self._next_samples(self._require(self._detnet_pool), self.config.detnet_batch_size)
        if samples is None:
            return None
        self._last_detnet = samples
        pooled: Float32[Tensor, "b 1 120 160"] = samples.pooled[:, None].float() / 255.0
        if not self.config.validation:
            pooled = scale_intensity(pooled, self._generator, *self.config.intensity_range)
        return DetNetBatch(
            pooled=pooled,
            circle=samples.circle,
            presence=samples.presence,
            circle_mask=samples.circle_mask,
            presence_mask=samples.presence_mask,
            dataset=samples.dataset,
        )

    def next_keynet_batch(self) -> KeyNetBatch | None:
        self._check("KeyNet", self._keynet_on)
        samples: KeyNetSamples | None
        if self._evaluation is not None:
            samples = self._next_evaluation(self._evaluation[1], 1, self.config.keynet_batch_size)
        else:
            samples = self._next_samples(self._require(self._keynet_pool), self.config.keynet_batch_size)
        if samples is None:
            return None
        self._last_keynet = samples
        crops: Float32[Tensor, "b 1 96 96"] = (samples.crops.float() / 255.0)[:, None]
        if not self.config.validation:
            augment: KeyNetAugment = self._augment
            occluded: Bool[Tensor, "b 96 96"] = boundary_occlusion(
                sample_count(samples), self._generator, self.device, augment.occlusion_probability, augment.occlusion_max_fraction
            )
            crops = scale_intensity(crops.masked_fill(occluded[:, None], 0.0), self._generator, *augment.intensity_range)
        positive: Bool[Tensor, "b"] = samples.kind == int(CropKind.POSITIVE)
        return KeyNetBatch(
            crops=crops,
            keypoints=samples.keypoints,
            heatmaps=render_heatmaps(samples.points_crop) * positive[:, None, None, None],
            distance=render_distance(samples.d_rel_mm) * positive[:, None, None],
            presence=samples.presence,
            positive=positive,
            presence_mask=torch.ones_like(positive),
            kind=samples.kind,
            dataset=samples.dataset,
        )

    def detnet_ready(self) -> bool:
        self._check("DetNet", self._detnet_on)
        return self._evaluation is not None or self._ready(self._require(self._detnet_pool), self.config.detnet_batch_size)

    def keynet_ready(self) -> bool:
        self._check("KeyNet", self._keynet_on)
        return self._evaluation is not None or self._ready(self._require(self._keynet_pool), self.config.keynet_batch_size)

    def detnet_validation(self) -> DetNetValidation:
        """Exact GT of the last DetNet batch (net-frame keypoints, in-front flags, camera ids); does not advance."""
        if self._last_detnet is None:
            raise RuntimeError("no DetNet batch drawn yet")
        return DetNetValidation(points=self._last_detnet.points, in_front=self._last_detnet.in_front, camera=self._last_detnet.camera)

    def keynet_validation(self) -> KeyNetValidation:
        """Exact GT of the last KeyNet batch (crop affine, crop keypoints, unclamped d_rel); does not advance."""
        if self._last_keynet is None:
            raise RuntimeError("no KeyNet batch drawn yet")
        return KeyNetValidation(crop_from_net=self._last_keynet.crop_from_net, points_crop=self._last_keynet.points_crop, distance_mm=self._last_keynet.d_rel_mm)

    def overwritten(self) -> tuple[int, int]:
        """Samples lost to a full pool (the side that was not consumed fast enough), DetNet and KeyNet."""
        return (0 if self._detnet_pool is None else self._detnet_pool.overwritten, 0 if self._keynet_pool is None else self._keynet_pool.overwritten)

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        deadline: float = time.monotonic() + 30.0
        for thread in self._threads:
            while thread.is_alive() and time.monotonic() < deadline:
                self._discard_queue()
                self._discard_work()
                thread.join(timeout=0.2)
        self._discard_queue()
        self._discard_work()

    def __enter__(self) -> "CatalogStream":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- main-thread internals

    def _check(self, name: str, enabled: bool) -> None:
        if not enabled:
            raise RuntimeError(f"this stream was built with nets={self.config.nets!r}; it has no {name} batches")
        if self._generation < 0:
            raise RuntimeError("call start_epoch() before drawing batches")

    def _require[SampleT: (DetNetSamples, KeyNetSamples)](self, pool: SamplePool[SampleT] | None) -> SamplePool[SampleT]:
        if pool is None:
            raise RuntimeError("this stream holds an evaluation set, not pools")
        return pool

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"a catalog producer failed ({self._error_segment}): {self._error!r}") from self._error

    def _epoch_produced(self) -> bool:
        with self._cond:
            return self._done >= len(self._epoch_segments)

    def _threshold(self, capacity: int, batch: int) -> int:
        return max(batch, int(self.config.min_fill * capacity))

    def _insert(self, chunk: _Chunk) -> None:
        if self.config.validation:
            self._collected.append(chunk)
            return
        if chunk.detnet is not None and self._detnet_pool is not None:
            self._detnet_pool.add(chunk.detnet)
        if chunk.keynet is not None and self._keynet_pool is not None:
            self._keynet_pool.add(chunk.keynet)

    def _drain(self, timeout: float | None) -> None:
        """Move every queued chunk of this epoch into the pools; with a timeout, wait that long for the first one."""
        self._raise_if_failed()
        try:
            chunk: _Chunk = self._queue.get(timeout=timeout) if timeout is not None else self._queue.get_nowait()
        except queue.Empty:
            return
        while True:
            if chunk.generation == self._generation:
                self._insert(chunk)
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                return

    def _discard_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _discard_work(self) -> None:
        while True:
            try:
                self._work.get_nowait()
            except queue.Empty:
                return

    def _ready(self, pool: SamplePool[DetNetSamples] | SamplePool[KeyNetSamples], batch: int) -> bool:
        produced: bool = self._epoch_produced()
        self._drain(None)
        return pool.count >= (batch if produced else self._threshold(pool.capacity, batch))

    def _next_samples[SampleT: (DetNetSamples, KeyNetSamples)](self, pool: SamplePool[SampleT], batch: int) -> SampleT | None:
        while True:
            # Read "produced" before draining: a producer queues its last chunk before it counts the segment done.
            produced: bool = self._epoch_produced()
            self._drain(None)
            if pool.count >= (batch if produced else self._threshold(pool.capacity, batch)):
                return pool.draw(batch)
            if produced:
                return None
            start: float = time.perf_counter()
            self._drain(0.5)
            self.stats.wait_s += time.perf_counter() - start

    # ---- fetchers and producers

    def _fail(self, error: BaseException, info: SegmentInfo | None) -> None:
        with self._cond:
            if self._error is None:
                self._error = error
                self._error_segment = "no segment" if info is None else f"{info.dataset} {info.segment_id}"
            self._cond.notify_all()

    def _stale(self, generation: int) -> bool:
        return self._stop.is_set() or generation != self._generation

    def _read_from_catalog(self, info: SegmentInfo) -> SegmentData:
        """Statics, timeline and video packets: three catalog queries on this thread's own client."""
        entries: dict[str, DatasetEntry] | None = getattr(self._local, "entries", None)
        if entries is None:
            client: rr.catalog.CatalogClient = rr.catalog.CatalogClient(self.config.catalog_url)
            entries = {name: client.get_dataset(name) for name in self.dataset_names}
            self._local.entries = entries
        entry: DatasetEntry = entries[info.dataset]
        layout: DatasetLayout = layout_for(info.dataset)
        statics: pa.Table = read_statics(entry, info)
        rig_letterboxes: tuple[CameraRig, tuple[Letterbox, ...]] = read_rig(statics, info)
        timeline: HandTimeline = read_hand_timeline(entry, info, statics)
        videos: tuple[CatalogVideo, ...] = read_catalog_videos(entry, info.segment_id, [f"{camera}/pinhole/video" for camera in layout.cameras], TIMELINE)
        return SegmentData(rig=rig_letterboxes[0], letterboxes=rig_letterboxes[1], timeline=timeline, videos=videos)

    def _record_failure(self, info: SegmentInfo, stage: str, attempt: int, error: BaseException) -> None:
        cause: BaseException | None = error.__cause__ if isinstance(error, SegmentFailure) else None
        line: str = f"{info.dataset} {info.segment_id} {stage}{' ' + str(error) if cause is not None else ''} attempt {attempt}: {cause or error!r}"
        print(f"[catalog stream] {line}", file=sys.stderr, flush=True)
        with self._cond:
            self.stats.failures.append(line)
            if attempt == 1:
                self.stats.retried_segments += 1

    def _skip(self, info: SegmentInfo, generation: int) -> None:
        """Drop a segment for the rest of the run; it still counts as done so the epoch ends."""
        with self._cond:
            self._skipped.add(info.segment_id)
            self.stats.skipped_segments += 1
            if generation == self._generation:
                self._done += 1
            limit: int = max(3, int(self.config.max_skipped_fraction * len(self.segments)))
            if len(self._skipped) > limit:
                failures: str = "\n".join(self.stats.failures[-10:])
                self._error = self._error or RuntimeError(f"{len(self._skipped)} segments failed twice (limit {limit}); last failures:\n{failures}")
            self._cond.notify_all()

    def _fetch(self, worker: int) -> None:
        """Take the epoch's next segment, read and label it (one retry, then skip), and queue it for a producer."""
        info: SegmentInfo | None = None
        try:
            while not self._stop.is_set():
                with self._cond:
                    while not self._stop.is_set() and self._next >= len(self._epoch_segments):
                        self._cond.wait(timeout=0.5)
                    if self._stop.is_set():
                        return
                    generation: int = self._generation
                    position: int = self._next
                    info = self._epoch_segments[position]
                    self._next += 1
                    if info.segment_id in self._skipped:
                        self._done += 1
                        continue
                work: _SegmentWork | None = None
                for attempt in (1, 2):
                    try:
                        work = self._fetch_segment(info, generation, position)
                        break
                    except BaseException as error:
                        if is_fatal(error):
                            raise
                        self._record_failure(info, "fetch", attempt, error)
                if work is None:
                    self._skip(info, generation)
                    continue
                while not self._stale(generation):
                    try:
                        self._work.put(work, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:  # every other failure, beartype's included, is re-raised in the main thread
            self._fail(error, info)

    def _fetch_segment(self, info: SegmentInfo, generation: int, position: int) -> _SegmentWork:
        layout: DatasetLayout = layout_for(info.dataset)
        start: float = time.perf_counter()
        data: SegmentData = self._read_segment(info)
        queried: float = time.perf_counter()
        rows: Int64[ndarray, "k"] = np.arange(0, len(data.timeline.video_time_ns), layout.pool_stride, dtype=np.int64)
        labels: SegmentLabels = segment_labels(data.timeline, data.rig, data.letterboxes, rows, is_show3d(info.dataset))
        priors: KeypointPriors | None = keypoint_priors(data.timeline, data.rig, data.letterboxes, rows, layout.tracker_step) if self._keynet_on else None
        with self._cond:
            self.stats.query_s += queried - start
            self.stats.label_s += time.perf_counter() - queried
        return _SegmentWork(
            generation=generation,
            position=position,
            info=info,
            dataset_index=self.dataset_names.index(info.dataset),
            letterboxes=data.letterboxes,
            labels=labels,
            priors=priors,
            hand_scale=data.timeline.hand_scale,
            times=data.timeline.video_time_ns[rows],
            videos=data.videos,
        )

    def _produce(self, worker: int) -> None:
        """Decode fetched segments; the NVDEC decoders live on this thread. A failed decode is retried once from a fresh fetch."""
        work: _SegmentWork | None = None
        try:
            generator: torch.Generator = torch.Generator(device=self.device)
            generator.manual_seed(self.config.seed * 1_000 + worker + 1)
            while not self._stop.is_set():
                try:
                    work = self._work.get(timeout=0.5)
                except queue.Empty:
                    continue
                if self._stale(work.generation):
                    continue
                decoded: bool = False
                for attempt in (1, 2):
                    try:
                        if attempt == 2:
                            work = self._fetch_segment(work.info, work.generation, work.position)
                        self._decode_segment(work, generator)
                        decoded = True
                        break
                    except _Stale:
                        break
                    except BaseException as error:
                        if is_fatal(error):
                            raise
                        self._record_failure(work.info, "decode", attempt, error)
                if self._stale(work.generation):
                    continue
                if not decoded:
                    self._skip(work.info, work.generation)
                    continue
                with self._cond:
                    if work.generation == self._generation:
                        self._done += 1
                        self.stats.segments += 1
        except BaseException as error:  # every other failure, beartype's included, is re-raised in the main thread
            self._fail(error, None if work is None else work.info)

    def _put(self, chunk: _Chunk) -> None:
        while True:
            if self._stale(chunk.generation):
                raise _Stale
            try:
                self._queue.put(chunk, timeout=0.5)
                return
            except queue.Full:
                continue

    def _decode_segment(self, work: _SegmentWork, generator: torch.Generator) -> None:
        start: float = time.perf_counter()
        if self.config.validation:
            # Evaluation sets must not depend on which producer took which segment.
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.config.seed * 100_003 + work.position)
        layout: DatasetLayout = layout_for(work.info.dataset)
        labels: SegmentLabels = labels_to(work.labels, self.device)
        priors: KeypointPriors | None = None if work.priors is None else priors_to(work.priors, self.device)
        has_pose: Bool[Tensor, "k 2"] = torch.isfinite(labels.landmarks).all(dim=-1).all(dim=-1)
        for camera, video in enumerate(work.videos):
            if self._stale(work.generation):
                raise _Stale
            position: Int64[ndarray, "k"] = np.clip(np.searchsorted(video.t_ns, work.times), 0, len(video.t_ns) - 1)
            matched: Bool[ndarray, "k"] = video.t_ns[position] == work.times
            valid: Bool[ndarray, "k"] = work.labels.image_valid[:, camera].numpy()
            keep: Int64[ndarray, "m"] = np.flatnonzero(matched & valid)
            with self._cond:
                self.stats.images_considered += len(work.times)
                self.stats.images_missing += int((~matched).sum())
                self.stats.images_invalid += int((matched & ~valid).sum())
            if len(keep) == 0:
                continue
            try:
                self._decode_camera(work, camera, video, position, keep, labels, priors, has_pose, generator)
            except BaseException as error:
                if is_fatal(error):
                    raise
                raise SegmentFailure(f"camera {layout.cameras[camera]}") from error
        with self._cond:
            self.stats.decode_s += time.perf_counter() - start

    def _decode_camera(
        self,
        work: _SegmentWork,
        camera: int,
        video: CatalogVideo,
        position: Int64[ndarray, "k"],
        keep: Int64[ndarray, "m"],
        labels: SegmentLabels,
        priors: KeypointPriors | None,
        has_pose: Bool[Tensor, "k 2"],
        generator: torch.Generator,
    ) -> None:
        """Decode one camera's kept images in chunks and queue their DetNet and KeyNet samples."""
        layout: DatasetLayout = layout_for(work.info.dataset)
        decoder: FrameDecoder = self._open_decoder(video, work.info.fps, self.device)
        letterbox: Letterbox = work.letterboxes[camera]
        chunk: int = max(1, DECODE_CHUNK_PIXELS // (letterbox.source_width * letterbox.source_height))
        for begin in range(0, len(keep), chunk):
            chunk_rows: Int64[ndarray, "m"] = keep[begin : begin + chunk]
            frames: UInt8[Tensor, "m h w"] = decoder.get_frames_at(position[chunk_rows].tolist()).data[:, 0]
            net: UInt8[Tensor, "m 480 640"] = letterbox.apply(frames)
            del frames
            index: Int64[Tensor, "m"] = torch.from_numpy(chunk_rows).to(self.device)
            detnet: DetNetSamples | None = None
            if self._detnet_on:
                detnet = detnet_samples(
                    net,
                    labels.hand_label[index, camera],
                    labels.circles[index, camera],
                    image_hands(labels.projection, has_pose, index, camera),
                    work.dataset_index,
                    layout.camera_offset + camera,
                )
            keynet: KeyNetSamples | None = None
            if priors is not None:
                keynet = keynet_samples(
                    net,
                    image_hands(labels.projection, has_pose, index, camera),
                    labels.hand_label[index, camera],
                    labels.circles[index, camera],
                    image_hands(priors.extrapolated, priors.extrapolated_valid, index, camera),
                    image_hands(priors.stale, priors.stale_valid, index, camera),
                    work.hand_scale,
                    work.dataset_index,
                    self._augment,
                    generator,
                )
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            with self._cond:
                self.stats.images_decoded += len(chunk_rows)
                self.stats.detnet_samples += 0 if detnet is None else sample_count(detnet)
                if keynet is not None:
                    self.stats.keynet_samples += sample_count(keynet)
                    counts: list[int] = torch.bincount(keynet.kind, minlength=len(CropKind)).tolist()
                    self.stats.keynet_kinds = [a + b for a, b in zip(self.stats.keynet_kinds, counts, strict=True)]
            self._put(_Chunk(work.generation, (work.position, camera, int(chunk_rows[0])), detnet, keynet))


def projection_to(projection: HandProjection, device: torch.device) -> HandProjection:
    return HandProjection(**{f.name: getattr(projection, f.name).to(device) for f in dataclasses.fields(projection)})


def labels_to(labels: SegmentLabels, device: torch.device) -> SegmentLabels:
    return SegmentLabels(
        rows=labels.rows,
        landmarks=labels.landmarks.to(device),
        projection=projection_to(labels.projection, device),
        circles=labels.circles.to(device),
        image_valid=labels.image_valid.to(device),
        labelled=labels.labelled.to(device),
        hand_label=labels.hand_label.to(device),
    )


def priors_to(priors: KeypointPriors, device: torch.device) -> KeypointPriors:
    return KeypointPriors(
        extrapolated=projection_to(priors.extrapolated, device),
        extrapolated_valid=priors.extrapolated_valid.to(device),
        stale=projection_to(priors.stale, device),
        stale_valid=priors.stale_valid.to(device),
    )
