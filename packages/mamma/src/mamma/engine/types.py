"""Shared per-tick data types for the streaming engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float32
from numpy import ndarray


@dataclass(slots=True)
class TrackedObject:
    """One person's track state in one camera at one tick."""

    obj_id: int
    """Stable person id, shared across cameras."""
    mask: Bool[torch.Tensor, "h w"]
    """Binary person mask at engine resolution, on the compute device."""
    bbox_xyxy: Float32[ndarray, "4"] | None
    """Mask-derived box ``[x1, y1, x2, y2]``; ``None`` when the mask is empty."""
    score: float
    """Tracker confidence (predicted mask IoU)."""


CameraTracks = dict[int, TrackedObject]
"""Per-person tracks in one camera, keyed by ``obj_id``."""
