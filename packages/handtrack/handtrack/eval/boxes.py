"""Detector box metrics: recall, precision, side accuracy, median IoU and false alarms on hand-free frames.

A hand's box is the axis-aligned square around its circle, in net pixels (640 x 480). A slot is positive when its presence
probability exceeds ``PRESENCE_THRESHOLD``; a positive slot detects its hand when the boxes overlap with IoU >= ``MIN_IOU``.
Slots without a presence label (PARTIAL hands, ``presence_mask`` False) are not scored.
"""
from dataclasses import dataclass, field

import torch
from jaxtyping import Bool, Float32
from serde import serde
from torch import Tensor

PRESENCE_THRESHOLD: float = 0.5
MIN_IOU: float = 0.5
IOU_BINS: int = 100
"""Histogram bins over [0, 1] that keep median IoU additive across batches."""
RATIO_BINS: int = 64
RATIO_LOG2_SPAN: float = 2.0
"""Radius ratios are binned as log2(predicted r / true r) over [-2, 2] (a quarter to four times), clamped at the ends."""
NET_SCALE: tuple[float, float, float] = (640.0, 480.0, 640.0)
"""Normalized (cx / 640, cy / 480, r / 640) to net pixels."""


def square_iou(a: Float32[Tensor, "*b 3"], b: Float32[Tensor, "*b 3"]) -> Float32[Tensor, "*b"]:
    """IoU of the squares around two net-pixel circles (cx, cy, r)."""
    low: Float32[Tensor, "*b 2"] = torch.maximum(a[..., :2] - a[..., 2:], b[..., :2] - b[..., 2:])
    high: Float32[Tensor, "*b 2"] = torch.minimum(a[..., :2] + a[..., 2:], b[..., :2] + b[..., 2:])
    overlap: Float32[Tensor, "*b"] = (high - low).clamp(min=0.0).prod(-1)
    union: Float32[Tensor, "*b"] = (2 * a[..., 2]) ** 2 + (2 * b[..., 2]) ** 2 - overlap
    return torch.where(union > 0, overlap / union.clamp(min=1e-9), torch.zeros_like(union))


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BoxCounts:
    """Additive counts of one group of images (a dataset, a side, or both)."""

    present: int = 0
    """Scored hands that are present (the recall denominator)."""
    detected: int = 0
    """Present hands whose own slot is positive with IoU >= MIN_IOU."""
    predicted: int = 0
    """Positive scored slots (the precision denominator)."""
    swapped: int = 0
    """Positive slots that miss their own side but box the other side's present hand (IoU >= MIN_IOU)."""
    absent_frames: int = 0
    """Images whose two slots are both scored and absent."""
    absent_frame_alarms: int = 0
    """Positive slots on those images."""
    iou_histogram: list[int] = field(default_factory=lambda: [0] * IOU_BINS)
    """IoU of present hands whose slot is positive, binned over [0, 1]."""
    detected_true_radius: int = 0
    """Present hands whose own slot is positive with IoU >= MIN_IOU once its radius is replaced by the true one (centre only)."""
    radius_ratio_histogram: list[int] = field(default_factory=lambda: [0] * RATIO_BINS)
    """log2(predicted r / true r) of present hands whose positive slot puts its centre inside the true circle."""

    def recall_at(self, iou: float) -> float | None:
        """Recall at another IoU threshold, from the IoU histogram (to its bin width, 0.01)."""
        return sum(self.iou_histogram[round(iou * IOU_BINS):]) / self.present if self.present else None

    def radius_ratio_percentile(self, q: float) -> float | None:
        """The q-quantile of predicted r / true r, at its bin centre."""
        total: int = sum(self.radius_ratio_histogram)
        if not total:
            return None
        seen: int = 0
        for index, count in enumerate(self.radius_ratio_histogram):
            seen += count
            if seen >= q * total:
                return float(2.0 ** ((index + 0.5) / RATIO_BINS * 2 * RATIO_LOG2_SPAN - RATIO_LOG2_SPAN))
        raise AssertionError("unreachable: the histogram sums to total")

    @property
    def recall(self) -> float | None:
        return self.detected / self.present if self.present else None

    @property
    def precision(self) -> float | None:
        return self.detected / self.predicted if self.predicted else None

    @property
    def side_accuracy(self) -> float | None:
        """Of the hands boxed in either slot, the share boxed in the right one."""
        boxed: int = self.detected + self.swapped
        return self.detected / boxed if boxed else None

    @property
    def false_alarms_per_1k(self) -> float | None:
        return 1000.0 * self.absent_frame_alarms / self.absent_frames if self.absent_frames else None

    @property
    def median_iou(self) -> float | None:
        total: int = sum(self.iou_histogram)
        if not total:
            return None
        seen: int = 0
        for index, count in enumerate(self.iou_histogram):
            seen += count
            if 2 * seen >= total:
                return (index + 0.5) / IOU_BINS
        raise AssertionError("unreachable: the histogram sums to total")

    def __add__(self, other: "BoxCounts") -> "BoxCounts":
        return BoxCounts(self.present + other.present, self.detected + other.detected, self.predicted + other.predicted,
                         self.swapped + other.swapped, self.absent_frames + other.absent_frames,
                         self.absent_frame_alarms + other.absent_frame_alarms,
                         [a + b for a, b in zip(self.iou_histogram, other.iou_histogram, strict=True)],
                         self.detected_true_radius + other.detected_true_radius,
                         [a + b for a, b in zip(self.radius_ratio_histogram, other.radius_ratio_histogram, strict=True)])


def box_counts(predicted: Float32[Tensor, "n 2 3"], probability: Float32[Tensor, "n 2"], target: Float32[Tensor, "n 2 3"],
               presence: Float32[Tensor, "n 2"], scored: Bool[Tensor, "n 2"]) -> BoxCounts:
    """Counts for images whose circles are normalized (cx / 640, cy / 480, r / 640); slot 0 is the left hand."""
    scale: Float32[Tensor, "3"] = torch.tensor(NET_SCALE, device=predicted.device)
    mine: Float32[Tensor, "n 2"] = square_iou(predicted * scale, target * scale)
    other: Float32[Tensor, "n 2"] = square_iou(predicted * scale, target.flip(1) * scale)
    positive: Bool[Tensor, "n 2"] = (probability > PRESENCE_THRESHOLD) & scored & torch.isfinite(predicted).all(-1)
    present: Bool[Tensor, "n 2"] = scored & (presence > 0.5)
    detected: Bool[Tensor, "n 2"] = positive & present & (mine >= MIN_IOU)
    swapped: Bool[Tensor, "n 2"] = positive & ~detected & (present.flip(1)) & (other >= MIN_IOU)
    hand_free: Bool[Tensor, "n"] = (scored & ~present).all(-1)
    ious: Float32[Tensor, "m"] = mine[positive & present]
    histogram: list[int] = torch.histc(ious.float(), bins=IOU_BINS, min=0.0, max=1.0).long().tolist() if ious.numel() else [0] * IOU_BINS
    true_radius: Float32[Tensor, "n 2 3"] = torch.cat((predicted[..., :2], target[..., 2:]), dim=-1)
    centred: Bool[Tensor, "n 2"] = positive & present & (square_iou(true_radius * scale, target * scale) >= MIN_IOU)
    pixels_pred: Float32[Tensor, "n 2 3"] = predicted * scale
    pixels_true: Float32[Tensor, "n 2 3"] = target * scale
    inside: Bool[Tensor, "n 2"] = positive & present & ((pixels_pred[..., :2] - pixels_true[..., :2]).norm(dim=-1) < pixels_true[..., 2])
    ratios: Float32[Tensor, "k"] = torch.log2(pixels_pred[..., 2][inside].clamp(min=1e-6) / pixels_true[..., 2][inside].clamp(min=1e-6))
    ratio_histogram: list[int] = (torch.histc(ratios.clamp(-RATIO_LOG2_SPAN, RATIO_LOG2_SPAN).float(), bins=RATIO_BINS, min=-RATIO_LOG2_SPAN, max=RATIO_LOG2_SPAN)
                                  .long().tolist() if ratios.numel() else [0] * RATIO_BINS)
    return BoxCounts(int(present.sum()), int(detected.sum()), int(positive.sum()), int(swapped.sum()), int(hand_free.sum()),
                     int((positive & hand_free[:, None]).sum()), histogram, int(centred.sum()), ratio_histogram)
