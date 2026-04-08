"""Event types for async Rerun logging.

Each event is a frozen dataclass containing only CPU-serializable data
(numpy arrays, Python scalars, tuples).  No live ``Frame`` objects, no
shared tensors, no ``lietorch`` types.

Events are produced by the pipeline thread (cheap CPU snapshots) and
consumed by the ``AsyncRerunLogger`` thread (expensive JPEG compression,
focal estimation, ``rr.log()`` calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from jaxtyping import Float32
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class KeyframeSnapshot:
    """CPU snapshot of a newly created keyframe."""

    kf_idx: int
    """Keyframe buffer index."""
    world_sim3_cam_data: Float32[ndarray, "8"]
    """Raw lietorch Sim3 data (8 floats) on CPU."""
    rgb: Float32[ndarray, "H W 3"]
    """Normalized RGB image in [0, 1] range, HWC layout."""
    X_canon: Float32[ndarray, "hw 3"]
    """Canonical 3D point map."""
    C: Float32[ndarray, "hw 1"]
    """Per-point confidence values."""
    img_shape: tuple[int, int]
    """(height, width) of the processed image."""


@dataclass(frozen=True, slots=True)
class LogCurrentFrame:
    """Per-frame current camera state.  Droppable when queue is full."""

    frame_idx: int
    """Frame index in the dataset sequence."""
    timestamp_ns: int | None
    """Video timestamp in nanoseconds, or None if not from MP4."""
    world_sim3_cam_data: Float32[ndarray, "8"]
    """Raw lietorch Sim3 data (8 floats) on CPU."""
    rgb: Float32[ndarray, "H W 3"]
    """Normalized RGB image in [0, 1] range, HWC layout."""
    X_canon: Float32[ndarray, "hw 3"] | None
    """Canonical 3D point map, or None."""
    C: Float32[ndarray, "hw 1"] | None
    """Per-point confidence values, or None."""
    img_shape: tuple[int, int]
    """(height, width) of the processed image."""


@dataclass(frozen=True, slots=True)
class LogMapUpdate:
    """Batched structural changes.  Never dropped.

    Sent when any structural change happens: new keyframe added, backend
    refines poses, factor graph edges change, or gravity-alignment
    (orient transform) is recomputed.
    """

    frame_idx: int
    """Frame index at which these changes were observed."""
    timestamp_ns: int | None
    """Video timestamp in nanoseconds, or None if not from MP4."""
    new_keyframes: list[KeyframeSnapshot] = field(default_factory=list)
    """Newly created keyframes this iteration (empty if none)."""
    pose_updates: list[tuple[int, Float32[ndarray, "8"]]] = field(default_factory=list)
    """(kf_idx, world_sim3_cam_data) pairs for backend-refined poses."""
    edge_positions: tuple[Float32[ndarray, "n 3"], Float32[ndarray, "n 3"]] | None = None
    """(positions_i, positions_j) for factor graph edges, or None if unchanged."""
    orient: tuple[Float32[ndarray, "3 3"], Float32[ndarray, "3"]] | None = None
    """(orient_R, orient_t) gravity-alignment transform, or None if unchanged."""


@dataclass(frozen=True, slots=True)
class LogText:
    """A text log entry."""

    path: str
    """Rerun entity path for the text log."""
    message: str
    """Log message content."""
    level: Literal["INFO", "WARN", "DEBUG", "ERROR"] = "INFO"
    """Log level."""


@dataclass(frozen=True, slots=True)
class LogTerminate:
    """Signals the logger thread to flush remaining events and exit."""


LogEvent = LogCurrentFrame | LogMapUpdate | LogText | LogTerminate
