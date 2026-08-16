"""Masked normal losses from GausSurf with an optional NeuRIS gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class GausSurfLosses:
    """Unweighted loss terms and active-pixel fractions for diagnostics."""

    normal_prior_cosine: torch.Tensor
    """Direct-normal cosine loss against the MoGe prior."""
    depth_normal_cosine: torch.Tensor
    """Direct-normal cosine loss against the depth-derived normal."""
    normal_prior_mask_fraction: torch.Tensor
    """Fraction of pixels supervised by the MoGe prior."""
    consistency_mask_fraction: torch.Tensor
    """Fraction of pixels supervised by normal consistency."""
    consistency_weight_mean: torch.Tensor
    """Mean optional consistency weight over valid surface pixels."""
    consistency_weight_below_half_fraction: torch.Tensor
    """Valid-surface fraction whose optional consistency weight is below one half."""


@dataclass(frozen=True, slots=True)
class GausSurfTargets:
    """Resized normal targets in the renderer's channel-last layout."""

    prior_normal_hw3: torch.Tensor
    """Unit MoGe prior normals shaped ``h w 3``."""
    prior_valid_hw1: torch.Tensor
    """Nearest-resized MoGe validity mask shaped ``h w 1``."""


def linear_ramp_weight(*, step: int, start_step: int, ramp_steps: int) -> float:
    """Return a zero-before-start, linearly increasing, saturated loss weight."""

    if start_step < 0:
        raise ValueError("ramp start step must be non-negative")
    if ramp_steps <= 0:
        raise ValueError("ramp steps must be positive")
    if step < start_step:
        return 0.0
    return min(1.0, (step - start_step + 1) / ramp_steps)


def normal_contradiction_gate(
    rendered_normal_hw3: torch.Tensor,
    prior_normal_hw3: torch.Tensor,
    prior_valid_hw1: torch.Tensor,
    *,
    maximum_angle_degrees: float = 30.0,
) -> torch.Tensor:
    """Return a detached NeuRIS-style angular contradiction filter."""

    if rendered_normal_hw3.shape != prior_normal_hw3.shape:
        raise ValueError("rendered and prior normals must share a shape")
    if prior_valid_hw1.shape != (*rendered_normal_hw3.shape[:2], 1):
        raise ValueError("prior validity must have shape [H,W,1]")
    rendered = F.normalize(rendered_normal_hw3.detach(), dim=-1, eps=1e-6)
    prior = F.normalize(prior_normal_hw3.detach(), dim=-1, eps=1e-6)
    cosine_threshold = torch.cos(
        torch.deg2rad(
            torch.tensor(
                maximum_angle_degrees,
                dtype=rendered.dtype,
                device=rendered.device,
            )
        )
    )
    cosine = (rendered * prior).sum(dim=-1, keepdim=True)
    return (prior_valid_hw1.bool() & (cosine >= cosine_threshold)).detach()


def pgsr_edge_weights(
    rgb_hw3: torch.Tensor,
    *,
    power: float = 2.0,
    minimum_weight: float = 0.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return detached PGSR-style weights that suppress RGB edges and borders."""

    if rgb_hw3.ndim != 3 or rgb_hw3.shape[-1] != 3:
        raise ValueError("RGB image must have shape [H,W,3]")
    height, width, _ = rgb_hw3.shape
    if height < 3 or width < 3:
        raise ValueError("RGB image must be at least 3x3")
    if power <= 0.0:
        raise ValueError("edge-weight power must be positive")
    if not 0.0 <= minimum_weight < 1.0:
        raise ValueError("minimum edge weight must be in [0, 1)")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    rgb_f32 = rgb_hw3.detach().to(dtype=torch.float32)
    gradient_x = (rgb_f32[:, 2:] - rgb_f32[:, :-2]).abs().mean(dim=-1)
    gradient_y = (rgb_f32[:-2] - rgb_f32[2:]).abs().mean(dim=-1)
    interior_gradient = torch.maximum(gradient_x[1:-1], gradient_y[:, 1:-1])
    gradient_min = interior_gradient.amin()
    gradient_span = interior_gradient.amax() - gradient_min
    normalized = torch.where(
        gradient_span > epsilon,
        (interior_gradient - gradient_min) / gradient_span.clamp_min(epsilon),
        torch.zeros_like(interior_gradient),
    )
    inverse_gradient = (1.0 - normalized.clamp(0.0, 1.0)).pow(power)
    interior_weight = minimum_weight + (1.0 - minimum_weight) * inverse_gradient
    weight = F.pad(interior_weight, (1, 1, 1, 1), mode="constant", value=0.0)
    return weight.unsqueeze(-1).to(dtype=rgb_hw3.dtype).detach()


def prepare_gaussurf_targets(
    *,
    prior_normal_3hw: torch.Tensor,
    prior_valid_1hw: torch.Tensor,
    height: int,
    width: int,
) -> GausSurfTargets:
    """Resize prior normals and validity without blurring boolean evidence."""

    if prior_normal_3hw.shape[0] != 3:
        raise ValueError("expected prior normal [3,H,W]")
    if prior_valid_1hw.shape[0] != 1:
        raise ValueError("expected prior validity [1,H,W]")
    prior_normal_3hw = F.interpolate(
        prior_normal_3hw[None],
        (height, width),
        mode="bilinear",
        align_corners=False,
    )[0]
    prior_normal_3hw = F.normalize(prior_normal_3hw, dim=0, eps=1e-6)

    prior_valid_resized_1hw: torch.Tensor = F.interpolate(
        prior_valid_1hw[None].float(),
        (height, width),
        mode="nearest",
    )[0].bool()
    return GausSurfTargets(
        prior_normal_hw3=prior_normal_3hw.permute(1, 2, 0),
        prior_valid_hw1=prior_valid_resized_1hw.permute(1, 2, 0),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero when the mask contains no active pixels."""

    mask_float = mask.to(dtype=values.dtype)
    safe_values = torch.where(mask.bool(), values, torch.zeros_like(values))
    return (safe_values * mask_float).sum() / mask_float.sum().clamp_min(1.0)


def _mask_fraction(mask: torch.Tensor) -> torch.Tensor:
    return mask.to(dtype=torch.float32).mean()


def compute_gaussurf_losses(
    *,
    direct_normal_hw3: torch.Tensor,
    depth_normal_hw3: torch.Tensor,
    prior_normal_hw3: torch.Tensor,
    prior_valid_hw1: torch.Tensor,
    surface_valid_hw1: torch.Tensor,
    normal_prior_gate_hw1: torch.Tensor | None = None,
    consistency_weight_hw1: torch.Tensor | None = None,
    normalize_consistency_weights: bool = True,
) -> GausSurfLosses:
    """Compute monocular-normal and surface-consistency losses.

    The MoGeV2 prior applies on valid prior and surface pixels, subject to the
    optional detached NeuRIS contradiction gate. Depth-normal consistency
    applies wherever the Gaussian surface renderer is valid.
    """

    expected_hw1: torch.Size = surface_valid_hw1.shape
    for name, mask in (
        ("prior_valid", prior_valid_hw1),
        ("surface_valid", surface_valid_hw1),
    ):
        if mask.shape != expected_hw1:
            raise ValueError(f"{name} mask shape must match depth")
    if direct_normal_hw3.shape != (*expected_hw1[:2], 3):
        raise ValueError("direct normal must have shape [H,W,3]")
    if depth_normal_hw3.shape != direct_normal_hw3.shape:
        raise ValueError("direct and depth-derived normal shapes must match")
    if prior_normal_hw3.shape != direct_normal_hw3.shape:
        raise ValueError("prior and rendered normal shapes must match")

    prior_valid_hw1 = prior_valid_hw1.bool()
    surface_valid_hw1 = surface_valid_hw1.bool()
    prior_mask_hw1: torch.Tensor = prior_valid_hw1 & surface_valid_hw1
    if normal_prior_gate_hw1 is not None:
        if normal_prior_gate_hw1.shape != expected_hw1:
            raise ValueError("normal-prior gate shape must match depth")
        normal_prior_gate_hw1 = normal_prior_gate_hw1.detach().bool()
        prior_mask_hw1 &= normal_prior_gate_hw1

    direct_normal_hw3 = F.normalize(direct_normal_hw3, dim=-1, eps=1e-6)
    depth_normal_hw3 = F.normalize(depth_normal_hw3, dim=-1, eps=1e-6)
    prior_normal_hw3 = F.normalize(prior_normal_hw3, dim=-1, eps=1e-6)
    prior_cosine_error_hw1 = 1.0 - (
        direct_normal_hw3 * prior_normal_hw3
    ).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    consistency_error_hw1 = 1.0 - (
        direct_normal_hw3 * depth_normal_hw3
    ).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    if consistency_weight_hw1 is None:
        consistency_loss = _masked_mean(consistency_error_hw1, surface_valid_hw1)
        consistency_weight_mean = torch.ones(
            (), dtype=consistency_error_hw1.dtype, device=consistency_error_hw1.device
        )
        consistency_weight_below_half_fraction = torch.zeros_like(
            consistency_weight_mean
        )
    else:
        if consistency_weight_hw1.shape != expected_hw1:
            raise ValueError("consistency weight shape must match depth")
        consistency_weight_hw1 = consistency_weight_hw1.detach().to(
            dtype=consistency_error_hw1.dtype
        )
        if not torch.isfinite(consistency_weight_hw1).all() or torch.any(
            consistency_weight_hw1 < 0.0
        ):
            raise ValueError("consistency weights must be finite and non-negative")
        surface_weight = consistency_weight_hw1 * surface_valid_hw1.to(
            dtype=consistency_error_hw1.dtype
        )
        weighted_consistency = consistency_error_hw1 * surface_weight
        if normalize_consistency_weights:
            consistency_loss = weighted_consistency.sum() / (
                surface_weight.sum().clamp_min(1e-6)
            )
        else:
            consistency_loss = weighted_consistency.mean()
        consistency_weight_mean = _masked_mean(
            consistency_weight_hw1, surface_valid_hw1
        )
        consistency_weight_below_half_fraction = _mask_fraction(
            surface_valid_hw1 & (consistency_weight_hw1 < 0.5)
        ) / _mask_fraction(surface_valid_hw1).clamp_min(1e-6)

    return GausSurfLosses(
        normal_prior_cosine=_masked_mean(prior_cosine_error_hw1, prior_mask_hw1),
        depth_normal_cosine=consistency_loss,
        normal_prior_mask_fraction=_mask_fraction(prior_mask_hw1),
        consistency_mask_fraction=_mask_fraction(surface_valid_hw1),
        consistency_weight_mean=consistency_weight_mean,
        consistency_weight_below_half_fraction=(
            consistency_weight_below_half_fraction
        ),
    )
