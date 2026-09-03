"""Edge-stratified depth metrics on a validity-aware teacher gradient stencil."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from serde import serde
from torch import Tensor

EDGE_QUANTILE: float = 0.90
"""Default teacher-gradient quantile defining the edge stratum."""
MIN_VALID_FOR_QUANTILE: int = 100
"""Fewest valid pixels accepted for a stable per-frame edge quantile."""

DepthInput: TypeAlias = Float32[Tensor, "h w"] | Float32[ndarray, "h w"]
ValidityInput: TypeAlias = Bool[Tensor, "h w"] | Bool[ndarray, "h w"]


@serde
@dataclass(frozen=True, slots=True)
class DepthMetrics:
    """Teacher-relative metrics for one depth prediction."""

    edge_mae_m: float
    """Mean absolute error on teacher edge pixels, in metres."""
    flat_mae_m: float
    """Mean absolute error on valid non-edge pixels, in metres."""
    overall_mae_m: float
    """Mean absolute error over all valid pixels, in metres."""
    ewmae_m: float
    """Absolute error weighted by valid teacher gradient magnitude, in metres."""


@dataclass(frozen=True, slots=True)
class EdgeStratifiedResult:
    """Reusable teacher stencil and the metrics scored against it."""

    prediction: DepthMetrics
    """Primary prediction metrics."""
    baseline: DepthMetrics | None
    """Optional baseline metrics measured on the same stencil."""
    gradient_hw: Float32[Tensor, "h w"]
    """Validity-aware teacher left/up gradient magnitude."""
    edge_hw: Bool[Tensor, "h w"]
    """Valid pixels in the teacher's upper gradient quantile."""


def edge_stratified_mae(
    prediction_depth_hw: DepthInput,
    target_depth_hw: DepthInput,
    target_valid_hw: ValidityInput,
    baseline_depth_hw: DepthInput | None = None,
    *,
    edge_quantile: float = EDGE_QUANTILE,
) -> EdgeStratifiedResult | None:
    """Score predictions on one reusable validity-aware teacher edge stencil.

    Args:
        prediction_depth_hw: Float32 predicted depth with shape ``(H, W)``.
        target_depth_hw: Float32 teacher depth with shape ``(H, W)``.
        target_valid_hw: Boolean teacher validity with shape ``(H, W)``.
        baseline_depth_hw: Optional Float32 baseline depth with shape ``(H, W)``.
        edge_quantile: Per-frame valid teacher-gradient quantile defining edges.

    Returns:
        The teacher stencil and prediction/baseline metrics, or ``None`` when
        the frame has too few valid pixels, an empty stratum, or no gradient.
    """
    if not 0.0 <= edge_quantile <= 1.0:
        raise ValueError("edge_quantile must be in [0, 1].")
    target: Float32[Tensor, "h w"] = (
        target_depth_hw if isinstance(target_depth_hw, Tensor) else torch.from_numpy(np.ascontiguousarray(target_depth_hw))
    ).to(dtype=torch.float32)
    prediction: Float32[Tensor, "h w"] = (
        prediction_depth_hw if isinstance(prediction_depth_hw, Tensor) else torch.from_numpy(np.ascontiguousarray(prediction_depth_hw))
    ).to(device=target.device, dtype=torch.float32)
    valid: Bool[Tensor, "h w"] = (
        target_valid_hw if isinstance(target_valid_hw, Tensor) else torch.from_numpy(np.ascontiguousarray(target_valid_hw))
    ).to(device=target.device, dtype=torch.bool)
    if target.ndim != 2 or prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("prediction, target, and validity mask must share one 2D shape.")

    gradient: Float32[Tensor, "h w"] = torch.zeros_like(target)
    valid_x: Bool[Tensor, "h width_minus_one"] = valid[:, 1:] & valid[:, :-1]
    valid_y: Bool[Tensor, "height_minus_one w"] = valid[1:, :] & valid[:-1, :]
    gradient[:, 1:] += torch.where(valid_x, (target[:, 1:] - target[:, :-1]).abs(), 0.0)
    gradient[1:, :] += torch.where(valid_y, (target[1:, :] - target[:-1, :]).abs(), 0.0)
    valid_gradients: Float32[Tensor, "n_valid"] = gradient[valid]
    if valid_gradients.numel() < MIN_VALID_FOR_QUANTILE:
        return None
    gradient_weight_sum: Float32[Tensor, ""] = valid_gradients.sum()
    if float(gradient_weight_sum.item()) <= 0.0:
        return None
    threshold: Float32[Tensor, ""] = torch.quantile(valid_gradients, edge_quantile)
    edge: Bool[Tensor, "h w"] = valid & (gradient >= threshold)
    flat: Bool[Tensor, "h w"] = valid & ~edge
    if not bool(edge.any()) or not bool(flat.any()):
        return None

    def score(depth_hw: Float32[Tensor, "h w"]) -> DepthMetrics:
        error: Float32[Tensor, "h w"] = (depth_hw - target).abs()
        return DepthMetrics(
            edge_mae_m=float(error[edge].mean().item()),
            flat_mae_m=float(error[flat].mean().item()),
            overall_mae_m=float(error[valid].mean().item()),
            ewmae_m=float((error[valid] * valid_gradients).sum().div(gradient_weight_sum).item()),
        )

    baseline_metrics: DepthMetrics | None = None
    if baseline_depth_hw is not None:
        baseline: Float32[Tensor, "h w"] = (
            baseline_depth_hw if isinstance(baseline_depth_hw, Tensor) else torch.from_numpy(np.ascontiguousarray(baseline_depth_hw))
        ).to(device=target.device, dtype=torch.float32)
        if baseline.shape != target.shape:
            raise ValueError("baseline and target must share one 2D shape.")
        baseline_metrics = score(baseline)
    return EdgeStratifiedResult(
        prediction=score(prediction),
        baseline=baseline_metrics,
        gradient_hw=gradient,
        edge_hw=edge,
    )
