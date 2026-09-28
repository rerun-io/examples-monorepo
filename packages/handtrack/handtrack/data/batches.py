"""The training batches the catalog stream hands to the DetNet and KeyNet trainers; every tensor is on the training device."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

from jaxtyping import Bool, Float32, Int64
from torch import Tensor


class CropKind(IntEnum):
    """Where a KeyNet crop came from; positives train every head, the rest train presence only."""

    POSITIVE = 0
    DRIFT = 1
    """The hand's box shifted off the hand, as a drifting tracker would send it."""
    EDGE = 2
    """A box at the image edge where the hand has left this view."""
    OTHER_HAND = 3
    """The other hand's box."""
    BACKGROUND = 4
    """A random box in a view where the hand is absent."""


@dataclass(frozen=True, slots=True)
class DetNetBatch:
    """Net frames pooled 4x4 (160x120) with per-hand circle and presence targets; slot 0 = left, 1 = right."""

    pooled: Float32[Tensor, "b 1 120 160"]
    """Intensity in [0, 1], augmented."""
    circle: Float32[Tensor, "b 2 3"]
    """(cx / 640, cy / 480, r / 640) in the net frame; undefined where ``circle_mask`` is False."""
    presence: Float32[Tensor, "b 2"]
    circle_mask: Bool[Tensor, "b 2"]
    """True for present hands (>= 17 keypoints visible)."""
    presence_mask: Bool[Tensor, "b 2"]
    """False for partly visible hands (1-16 keypoints visible): no loss at all."""
    dataset: Int64[Tensor, "b"]
    """Index into the stream's dataset list, for per-dataset statistics."""


@dataclass(frozen=True, slots=True)
class KeyNetBatch:
    """96x96 left-hand crops (right hands mirrored) with keypoint inputs and heatmap and presence targets."""

    crops: Float32[Tensor, "b 1 96 96"]
    """Intensity in [0, 1], augmented."""
    keypoints: Float32[Tensor, "b 63"]
    """21 x (u, v, d) keypoint input (``labels.keypoint_input``); all zeros for an untracked hand."""
    heatmaps: Float32[Tensor, "b 21 18 18"]
    distance: Float32[Tensor, "b 21 18"]
    presence: Float32[Tensor, "b"]
    positive: Bool[Tensor, "b"]
    """Heatmap and distance losses apply here only."""
    presence_mask: Bool[Tensor, "b"]
    """False for crops with 1-16 of the hand's keypoints inside: no loss at all."""
    kind: Int64[Tensor, "b"]
    """``CropKind`` per crop."""
    dataset: Int64[Tensor, "b"]


@runtime_checkable
class BatchSource(Protocol):
    """What a trainer draws batches from (``handtrack.data.stream.CatalogStream``); None exhausts a pool for the epoch.

    Call ``start_epoch`` before the first batch and again after an epoch ends. A source built for one network raises on
    the other network's method. ``CatalogStream`` also offers non-blocking ``detnet_ready()`` / ``keynet_ready()``.
    """

    def next_detnet_batch(self) -> DetNetBatch | None: ...

    def next_keynet_batch(self) -> KeyNetBatch | None: ...

    def start_epoch(self, epoch: int) -> None: ...

    def cancel(self) -> None:
        """Stop pending waits; nonblocking sources may implement a no-op."""
        ...

    def close(self) -> None: ...
