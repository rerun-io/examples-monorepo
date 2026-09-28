"""CPU-compatible metrics. Positions use metres unless explicitly labelled pixels or mm."""
from dataclasses import dataclass
from itertools import groupby

import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from handtrack.labels.circles import enclosing_circles


@dataclass(frozen=True, slots=True)
class Counts:
    """Additive detection counts; empty denominators have no score."""
    true_positive: int = 0
    """Predictions satisfying every detection criterion."""
    predicted: int = 0
    """Positive predictions."""
    ground_truth: int = 0
    """Eligible GT hands."""

    @property
    def precision(self) -> float | None:
        return self.true_positive / self.predicted if self.predicted else None

    @property
    def recall(self) -> float | None:
        return self.true_positive / self.ground_truth if self.ground_truth else None

    def __add__(self, other: 'Counts') -> 'Counts':
        return Counts(self.true_positive + other.true_positive, self.predicted + other.predicted, self.ground_truth + other.ground_truth)


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Counts grouped by camera and hand, with micro-averaged totals."""
    total: Counts
    """All evaluated observations."""
    by_camera_hand: dict[tuple[int, int], Counts]
    """Counts keyed by (camera ID, hand slot)."""

    def __add__(self, other: 'DetectionMetrics') -> 'DetectionMetrics':
        groups: dict[tuple[int, int], Counts] = dict(self.by_camera_hand)
        for key, counts in other.by_camera_hand.items():
            groups[key] = groups.get(key, Counts()) + counts
        return DetectionMetrics(self.total + other.total, groups)


def detection_metrics(
    boxes: Float32[Tensor, 'n 4'], probability: Float32[Tensor, 'n'],
    points: Float32[Tensor, 'n 21 2'], in_front: Bool[Tensor, 'n 21'],
    camera: Int64[Tensor, 'n'], hand: Int64[Tensor, 'n'], eligible: Bool[Tensor, 'n'],
    containment_scale: float = 1.0,
) -> DetectionMetrics:
    """Apply §5.4 to unexpanded square boxes in 640x480 pixels.

    Recall includes hands with at least MIN_VISIBLE_KEYPOINTS (17) in-front points
    in the NATIVE camera image. The caller supplies this eligibility from native
    labels; letterbox padding must not add points to the recall denominator.
    TP also requires probability >0.5, width within 20% of the smallest
    enclosing-circle diameter, and all in-front points inside the predicted
    closed square scaled about its centre by containment_scale (1.0 is §5.4).
    The width test always uses the unscaled box. Nonfinite predictions cannot be TP. Behind-camera points
    have no image position and are excluded from the circle and containment
    tests: §5.4 projected keypoints are the in-front ones.
    """
    circles: Float32[Tensor, "n 3"] = torch.from_numpy(enclosing_circles(points.detach().cpu().numpy(), in_front.cpu().numpy())).to(points.device)
    width: Float32[Tensor, "n"] = boxes[:, 2] - boxes[:, 0]
    diameter: Float32[Tensor, "n"] = 2 * circles[:, 2]
    # Expand from the original bounds so scale 1 preserves the closed-box boundary exactly.
    margin: Float32[Tensor, "n 2"] = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * (containment_scale - 1.0))
    low: Float32[Tensor, "n 2"] = boxes[:, :2] - margin
    high: Float32[Tensor, "n 2"] = boxes[:, 2:] + margin
    contains: Bool[Tensor, "n"] = (((points >= low[:, None]) & (points <= high[:, None])).all(-1) | ~in_front).all(-1)
    positive: Bool[Tensor, "n"] = probability > 0.5
    tp: Bool[Tensor, "n"] = positive & eligible & contains & (diameter > 0) & ((width - diameter).abs() <= diameter * 0.2)
    tp &= torch.isfinite(boxes).all(-1) & torch.isclose(width, boxes[:, 3] - boxes[:, 1])
    groups: dict[tuple[int, int], Counts] = {}
    for c, h in zip(camera.tolist(), hand.tolist(), strict=True):
        key: tuple[int, int] = (int(c), int(h))
        if key not in groups:
            select: Bool[Tensor, "n"] = (camera == c) & (hand == h)
            groups[key] = Counts(int(tp[select].sum()), int(positive[select].sum()), int(eligible[select].sum()))
    return DetectionMetrics(Counts(int(tp.sum()), int(positive.sum()), int(eligible.sum())), groups)


@dataclass(frozen=True, slots=True)
class KeypointMetrics:
    """Sums and counts permit exact aggregation across unequal batch sizes."""
    pixel_error_sum: float
    """Sum of 2D Euclidean errors in net-frame pixels."""
    distance_error_sum: float
    """Sum of absolute d_rel errors in mm."""
    keypoints: int
    """Number of scored keypoints."""
    presence: Counts
    """Presence counts after masking."""

    @property
    def error_px(self) -> float | None:
        return self.pixel_error_sum / self.keypoints if self.keypoints else None

    @property
    def distance_mm(self) -> float | None:
        return self.distance_error_sum / self.keypoints if self.keypoints else None

    def __add__(self, other: 'KeypointMetrics') -> 'KeypointMetrics':
        return KeypointMetrics(self.pixel_error_sum + other.pixel_error_sum, self.distance_error_sum + other.distance_error_sum,
                               self.keypoints + other.keypoints, self.presence + other.presence)


def presence_counts(probability: Float32[Tensor, 'b'], presence: Float32[Tensor, 'b'], presence_mask: Bool[Tensor, 'b']) -> Counts:
    """Presence counts over unmasked crops, with probability >=0.5 positive."""
    predicted: Bool[Tensor, "b"] = (probability >= 0.5) & presence_mask
    gt: Bool[Tensor, "b"] = (presence > 0.5) & presence_mask
    return Counts(int((predicted & gt).sum()), int(predicted.sum()), int(gt.sum()))


def keynet_metrics(
    predicted_crop: Float32[Tensor, 'b 21 2'], target_crop: Float32[Tensor, 'b 21 2'],
    crop_from_net: Float32[Tensor, 'b 3 3'], predicted_distance: Float32[Tensor, 'b 21'],
    target_distance: Float32[Tensor, 'b 21'], probability: Float32[Tensor, 'b'],
    presence: Float32[Tensor, 'b'], positive: Bool[Tensor, 'b'], presence_mask: Bool[Tensor, 'b'],
) -> KeypointMetrics:
    """Score GT-box crops; inverse affine includes any right-hand mirror.

    Geometric errors use positive samples only. Presence uses unmasked samples
    only, with probability >=0.5 positive. d_rel values are already in mm.
    """
    inverse: Float32[Tensor, "positive 3 3"] = torch.linalg.inv(crop_from_net[positive])
    delta: Float32[Tensor, "positive 21 2"] = predicted_crop[positive] - target_crop[positive]
    net_delta: Float32[Tensor, "positive 21 2"] = torch.einsum('bij,bkj->bki', inverse[:, :2, :2], delta)
    return KeypointMetrics(float(net_delta.norm(dim=-1).sum()), float((predicted_distance[positive] - target_distance[positive]).abs().sum()),
                           int(positive.sum()) * 21, presence_counts(probability, presence, presence_mask))


@dataclass(frozen=True, slots=True)
class PipelineMetrics:
    """Position and temporal errors in physical units."""
    mkpe_mm: float | None
    """Mean position error over GT hands."""
    mka_mm: float | None
    """Mean prediction acceleration, mm/frame²."""
    mka_gt_mm: float | None
    """Mean GT acceleration, mm/frame²."""


def pipeline_metrics(predicted: Float32[Tensor, 't h 21 3'], target: Float32[Tensor, 't h 21 3'], valid: Bool[Tensor, 't h']) -> PipelineMetrics:
    """Score contiguous frames of ONE sequence, in metres.

    MKPE averages every keypoint of every GT-valid hand/frame. Predictions
    must be finite there; missing predictions must not silently reduce MKPE.
    MKA averages norms of p[t-1]+p[t+1]-2p[t] only where all three GT frames
    are valid. Call separately at sequence boundaries. Empty scores are None.
    """
    if not torch.isfinite(predicted[valid]).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Nonfinite positions on a GT-valid frame')
    triple: Bool[Tensor, "middle h"] = valid[:-2] & valid[1:-1] & valid[2:]
    error: Float32[Tensor, "valid 21"] = (predicted[valid] - target[valid]).norm(dim=-1)
    acceleration: Float32[Tensor, "valid 21"] = (predicted[:-2] + predicted[2:] - 2 * predicted[1:-1])[triple].norm(dim=-1)
    gt_acceleration: Float32[Tensor, "valid 21"] = (target[:-2] + target[2:] - 2 * target[1:-1])[triple].norm(dim=-1)
    return PipelineMetrics(float(error.mean()) * 1000 if error.numel() else None,
                           float(acceleration.mean()) * 1000 if acceleration.numel() else None,
                           float(gt_acceleration.mean()) * 1000 if gt_acceleration.numel() else None)


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    """Per-appearance delays preserve failures and right-censored events."""
    visible_tracked_fraction: float | None
    """Tracked visible frames divided by all visible frames."""
    acquire_frames: tuple[int | None, ...]
    """Delay per appearance; None means never acquired during that appearance."""
    drop_frames: tuple[int | None, ...]
    """Delay per disappearance; None means not dropped before reappearance/end."""
    tracked_without_hand: int
    """Total frames tracked while GT is invisible, including before first appearance."""


def tracking_metrics(visible: Bool[Tensor, 't'], tracked: Bool[Tensor, 't'], observation_valid: Bool[Tensor, 't'] | None = None) -> TrackingMetrics:
    """Score one hand in one contiguous sequence (equal-length boolean arrays).

    Appearance starts at each False→True visibility transition, including
    visible frame zero. Acquisition delay is the zero-based offset to the
    first tracked frame in that visible run. Disappearance starts at each
    True→False transition; drop delay is the offset to the first untracked
    frame in that invisible run (zero if already dropped). Initial invisible
    frames are not a disappearance. Unknown rows end each observation interval:
    pending events become None, and the first row after a gap starts no event.
    Unknown rows do not count as absence. Unobserved events are None, not successes.
    """
    if visible.shape != tracked.shape:
        raise ValueError('Visibility and tracking lengths differ')
    observed: Bool[Tensor, "t"] = torch.ones_like(visible) if observation_valid is None else observation_valid
    if observed.shape != visible.shape:
        raise ValueError("Observation validity and visibility lengths differ")
    visible = visible & observed
    v: list[int] = torch.where(observed, visible.to(torch.int64), -1).tolist()
    tr: list[bool] = tracked.tolist()
    acquire: list[int | None] = []
    drop: list[int | None] = []
    for value, run in groupby(range(len(v)), key=v.__getitem__):
        frames: list[int] = list(run)
        start: int = frames[0]
        if value == -1 or (start > 0 and v[start - 1] == -1):
            continue
        if value == 1:
            acquire.append(next((i - start for i in frames if tr[i]), None))
        elif start:
            drop.append(next((i - start for i in frames if not tr[i]), None))
    count: int = int(visible.sum())
    return TrackingMetrics(int((visible & tracked).sum()) / count if count else None, tuple(acquire), tuple(drop), int((observed & ~visible & tracked).sum()))
