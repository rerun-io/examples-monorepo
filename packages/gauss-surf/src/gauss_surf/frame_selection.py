"""Pure sharp-frame scoring and selection policies."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from arkitscenes_download.ingest.timestamps import TimedeltaNs
from einops import rearrange
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

SCORE_HEIGHT: int = 540
"""Fixed scoring height inherited from the reference frame-selection fork."""
SCORE_WIDTH: int = 720
"""Fixed scoring width inherited from the reference frame-selection fork."""

@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Dense frame measurements and sparse chosen-frame metadata."""

    timestamps: TimedeltaNs
    """Packet timestamp for every scored frame, as ``timedelta64[ns]``."""
    sharpness: Float32[ndarray, "n_frames"]
    """Resolution-normalized variance-of-Laplacian score for every frame."""
    chosen_indices: Int64[ndarray, "n_chosen"]
    """Zero-based source-frame indices admitted by the selection rule."""
    chosen_timestamps: TimedeltaNs
    """Packet timestamps for admitted frames, as ``timedelta64[ns]``."""
    windows: Int64[ndarray, "n_chosen"]
    """Fixed window index for each admitted frame, or its source index when unwindowed."""
    rules: tuple[str, ...]
    """Small admission-rule code for each admitted frame."""
    rejected: bool
    """Whether the segment failed the minimum chosen-frame policy."""
    rejection_reason: str | None
    """Human-readable rejection reason, or None for an accepted result."""


@dataclass(frozen=True, slots=True)
class FrameScores:
    """Per-frame sharpness and texture measurements."""

    sharpness: Float32[Tensor, "b"]
    """Variance of the grayscale Laplacian at the fixed scoring resolution."""
    texture: Float32[Tensor, "b"]
    """Grayscale population standard deviation at the fixed scoring resolution."""


def _reference_gray(frames_bhw3: UInt8[Tensor, "b h w 3"]) -> Float32[Tensor, "b 1 score_h=540 score_w=720"]:
    """Convert frames to grayscale at the fixed scoring resolution.

    Scoring at one reference resolution makes scores comparable across source
    camera resolutions.
    """
    if frames_bhw3.ndim != 4 or frames_bhw3.shape[-1] != 3:
        raise ValueError("Frames must have shape (B, H, W, 3)")
    luma_weights_3: Float32[Tensor, "channels=3"] = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32, device=frames_bhw3.device)
    gray_bhw: Float32[Tensor, "b h w"] = torch.sum(frames_bhw3.to(dtype=torch.float32) * luma_weights_3, dim=-1)
    return functional.interpolate(rearrange(gray_bhw, "b h w -> b 1 h w"), size=(SCORE_HEIGHT, SCORE_WIDTH), mode="area")


def sharpness_scores(frames_bhw3: UInt8[Tensor, "b h w 3"]) -> Float32[Tensor, "b"]:
    """Variance of the grayscale Laplacian per frame — the production selection signal.

    Args:
        frames_bhw3: uint8 RGB frames with shape ``(B, H, W, 3)`` on any Torch device.

    Returns:
        Float32 sharpness with shape ``(B,)`` on the input device.
    """
    if frames_bhw3.ndim == 4 and len(frames_bhw3) == 0:
        return torch.empty(0, dtype=torch.float32, device=frames_bhw3.device)
    reference_b1hw: Float32[Tensor, "b 1 score_h=540 score_w=720"] = _reference_gray(frames_bhw3)
    laplacian_kernel_1133: Float32[Tensor, "1 1 kh=3 kw=3"] = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
        device=frames_bhw3.device,
    )
    laplacian_b1hw: Float32[Tensor, "b 1 lap_h lap_w"] = functional.conv2d(reference_b1hw, laplacian_kernel_1133)
    return torch.var(laplacian_b1hw, dim=(1, 2, 3), correction=0)


def score_frames(frames_bhw3: UInt8[Tensor, "b h w 3"]) -> FrameScores:
    """Measure sharpness and texture for a CPU or CUDA RGB frame batch (diagnostics).

    Args:
        frames_bhw3: uint8 RGB frames with shape ``(B, H, W, 3)`` on any Torch device.

    Returns:
        Float32 sharpness and texture tensors with shape ``(B,)`` on the input device.
    """
    sharpness_b: Float32[Tensor, "b"] = sharpness_scores(frames_bhw3)
    if len(frames_bhw3) == 0:
        return FrameScores(sharpness=sharpness_b, texture=sharpness_b)
    texture_b: Float32[Tensor, "b"] = torch.std(_reference_gray(frames_bhw3), dim=(1, 2, 3), correction=0)
    return FrameScores(sharpness=sharpness_b, texture=texture_b)


def _validate_selection_inputs(
    sharpness_n: Float32[ndarray, "n_frames"],
    timestamps_n: TimedeltaNs,
) -> TimedeltaNs:
    """Validate dense score/timestamp arrays and normalize timestamp precision."""
    if sharpness_n.ndim != 1 or timestamps_n.ndim != 1:
        raise ValueError("Sharpness scores and timestamps must be one-dimensional")
    if len(sharpness_n) != len(timestamps_n):
        raise ValueError("Sharpness scores and timestamps must have the same length")
    normalized_timestamps_n: TimedeltaNs = np.asarray(timestamps_n, dtype="timedelta64[ns]")
    return normalized_timestamps_n


def select_wide_frames(
    sharpness_n: Float32[ndarray, "n_frames"],
    timestamps_n: TimedeltaNs,
    *,
    window_size: int = 6,
    min_chosen: int = 200,
) -> SelectionResult:
    """Select the sharpest frame in every strict window.

    Args:
        sharpness_n: Float32 sharpness score per frame with shape ``(N,)``.
        timestamps_n: Packet timestamps with shape ``(N,)`` and dtype ``timedelta64[ns]``.
        window_size: Number of consecutive source frames per non-overlapping window.
        min_chosen: Minimum admitted frames required to accept the segment.

    Returns:
        Dense measurements, sparse winner metadata, and the segment rejection state.
    """
    normalized_timestamps_n: TimedeltaNs = _validate_selection_inputs(sharpness_n, timestamps_n)
    if window_size < 1 or min_chosen < 1:
        raise ValueError("window_size and min_chosen must be positive")

    chosen_indices: list[int] = []
    chosen_windows: list[int] = []
    window_start: int
    for window_start in range(0, len(sharpness_n), window_size):
        window_index: int = window_start // window_size
        window_stop: int = min(window_start + window_size, len(sharpness_n))
        winner_index: int = window_start + int(np.argmax(sharpness_n[window_start:window_stop]))
        chosen_indices.append(winner_index)
        chosen_windows.append(window_index)

    chosen_indices_n: Int64[ndarray, "n_chosen"] = np.asarray(chosen_indices, dtype=np.int64)
    windows_n: Int64[ndarray, "n_chosen"] = np.asarray(chosen_windows, dtype=np.int64)
    chosen_timestamps_n: TimedeltaNs = normalized_timestamps_n[chosen_indices_n]
    rejected: bool = len(chosen_indices_n) < min_chosen
    rejection_reason: str | None = None
    if rejected:
        rejection_reason = f"wide selection kept {len(chosen_indices_n)} frames, fewer than min_chosen={min_chosen}"
    return SelectionResult(
        timestamps=normalized_timestamps_n,
        sharpness=sharpness_n,
        chosen_indices=chosen_indices_n,
        chosen_timestamps=chosen_timestamps_n,
        windows=windows_n,
        rules=("window_best",) * len(chosen_indices_n),
        rejected=rejected,
        rejection_reason=rejection_reason,
    )


def select_ultrawide_frames(
    sharpness_n: Float32[ndarray, "n_frames"],
    timestamps_n: TimedeltaNs,
) -> SelectionResult:
    """Keep every ultrawide frame.

    Args:
        sharpness_n: Float32 sharpness score per frame with shape ``(N,)``.
        timestamps_n: Packet timestamps with shape ``(N,)`` and dtype ``timedelta64[ns]``.

    Returns:
        Dense measurements and sparse unwindowed chosen-frame metadata.
    """
    normalized_timestamps_n: TimedeltaNs = _validate_selection_inputs(sharpness_n, timestamps_n)
    chosen_indices_n: Int64[ndarray, "n_chosen"] = np.arange(len(sharpness_n), dtype=np.int64)
    chosen_timestamps_n: TimedeltaNs = normalized_timestamps_n[chosen_indices_n]
    return SelectionResult(
        timestamps=normalized_timestamps_n,
        sharpness=sharpness_n,
        chosen_indices=chosen_indices_n,
        chosen_timestamps=chosen_timestamps_n,
        windows=chosen_indices_n.copy(),
        rules=("all",) * len(chosen_indices_n),
        rejected=False,
        rejection_reason=None,
    )
