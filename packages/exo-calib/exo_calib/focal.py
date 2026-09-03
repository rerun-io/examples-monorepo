"""Per-camera focal refinement with fixed poses and 3D points."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from torch import Tensor

from exo_calib.correspondences import ObservationSet
from exo_calib.triangulation import valid_observation_mask

TorchDevice: TypeAlias = Literal["cuda", "cpu"]


@dataclass(slots=True)
class FocalRefinementResult:
    """Refined intrinsics and optimization diagnostics."""

    intrinsics: Float64[ndarray, "v 3 3"]
    """Float64 refined camera intrinsics with shape ``(v, 3, 3)``."""
    focal_scale: Float64[ndarray, "v"]
    """Per-camera multiplicative scale applied to both focal axes."""
    initial_loss: float
    """Confidence-weighted Cauchy loss before optimization."""
    final_loss: float
    """Confidence-weighted Cauchy loss after optimization."""


def refine_focals(
    intrinsics: Float64[ndarray, "v 3 3"],
    cam_T_world: Float64[ndarray, "v 4 4"],
    points_xyz: Float64[ndarray, "n 3"],
    obs: ObservationSet,
    *,
    iterations: int,
    learning_rate: float,
    robust_scale_px: float,
    device: TorchDevice = "cuda",
) -> FocalRefinementResult:
    """Refine one log-focal scale per camera with fixed poses and points.

    Args:
        intrinsics: Float64 initial intrinsics with shape ``(v, 3, 3)``.
        cam_T_world: Float64 fixed world-to-camera transforms with shape ``(v, 4, 4)``.
        points_xyz: Float64 fixed world points with shape ``(n, 3)``.
        obs: Sparse pixel observations of ``points_xyz``.
        iterations: Adam update count.
        learning_rate: Adam learning rate for log-focal scales.
        robust_scale_px: Cauchy residual scale in pixels.
        device: Torch device the solve runs on.

    Returns:
        Refined intrinsics, focal scales, and loss diagnostics.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if robust_scale_px <= 0.0:
        raise ValueError("robust_scale_px must be positive")
    n_views: int = intrinsics.shape[0]
    valid_mask: Bool[ndarray, "n_obs"] = valid_observation_mask(obs, n_views, n_points=points_xyz.shape[0])
    valid_idx: Int64[ndarray, "q"] = np.flatnonzero(valid_mask).astype(np.int64)
    if valid_idx.size == 0:
        raise ValueError("no valid observations for focal refinement")
    observation_point_idx: Int64[ndarray, "q"] = obs.obs_point_idx[valid_idx]
    finite_point_mask: Bool[ndarray, "q"] = np.isfinite(points_xyz[observation_point_idx]).all(axis=1)
    valid_idx = valid_idx[finite_point_mask]
    observation_point_idx = obs.obs_point_idx[valid_idx]
    observation_view_idx: Int64[ndarray, "q"] = obs.obs_view_idx[valid_idx]
    points_homo: Float64[ndarray, "q 4"] = np.column_stack(
        (points_xyz[observation_point_idx], np.ones(observation_point_idx.size, dtype=np.float64))
    )
    points_cam: Float64[ndarray, "q 3"] = np.einsum(
        "qij,qj->qi", cam_T_world[observation_view_idx, :3], points_homo
    )
    positive_depth: Bool[ndarray, "q"] = points_cam[:, 2] > 0.0
    valid_idx = valid_idx[positive_depth]
    observation_view_idx = observation_view_idx[positive_depth]
    points_cam = points_cam[positive_depth]
    if valid_idx.size == 0:
        raise ValueError("no positive-depth observations for focal refinement")

    dtype: torch.dtype = torch.float64
    view_idx: Int64[Tensor, "q"] = torch.as_tensor(observation_view_idx, dtype=torch.long, device=device)
    normalized_xy: Float64[Tensor, "q 2"] = torch.as_tensor(points_cam[:, :2] / points_cam[:, 2:3], dtype=dtype, device=device)
    observed_xy: Float64[Tensor, "q 2"] = torch.as_tensor(obs.obs_xy[valid_idx], dtype=dtype, device=device)
    weights: Float64[Tensor, "q"] = torch.as_tensor(obs.obs_conf[valid_idx], dtype=dtype, device=device)
    focal_lengths: Float64[Tensor, "v 2"] = torch.as_tensor(np.column_stack((intrinsics[:, 0, 0], intrinsics[:, 1, 1])), dtype=dtype, device=device)
    principal_points: Float64[Tensor, "v 2"] = torch.as_tensor(np.column_stack((intrinsics[:, 0, 2], intrinsics[:, 1, 2])), dtype=dtype, device=device)
    skew: Float64[Tensor, "v"] = torch.as_tensor(intrinsics[:, 0, 1], dtype=dtype, device=device)
    log_focal_delta: torch.nn.Parameter = torch.nn.Parameter(torch.zeros(n_views, dtype=dtype, device=device))

    def loss_value() -> Float64[Tensor, ""]:
        observation_focal_scale: Float64[Tensor, "q"] = torch.exp(log_focal_delta[view_idx])
        projected_xy: Float64[Tensor, "q 2"] = normalized_xy * focal_lengths[view_idx] * observation_focal_scale[:, None]
        projected_xy[:, 0] += skew[view_idx] * normalized_xy[:, 1]
        projected_xy += principal_points[view_idx]
        residual_sq: Float64[Tensor, "q"] = torch.sum((projected_xy - observed_xy) ** 2, dim=1)
        return torch.sum(weights * torch.log1p(residual_sq / robust_scale_px**2)) / torch.sum(weights)

    initial_loss: float = float(loss_value().detach().cpu())
    optimizer: torch.optim.Adam = torch.optim.Adam([log_focal_delta], lr=learning_rate)
    for _iteration in range(iterations):
        optimizer.zero_grad()
        loss: Float64[Tensor, ""] = loss_value()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_focal_delta.clamp_(min=-0.25, max=0.25)
    final_loss: float = float(loss_value().detach().cpu())
    focal_scale: Float64[ndarray, "v"] = torch.exp(log_focal_delta.detach()).cpu().numpy()
    refined_intrinsics: Float64[ndarray, "v 3 3"] = intrinsics.copy()
    refined_intrinsics[:, 0, 0] *= focal_scale
    refined_intrinsics[:, 1, 1] *= focal_scale
    return FocalRefinementResult(
        intrinsics=refined_intrinsics,
        focal_scale=focal_scale,
        initial_loss=initial_loss,
        final_loss=final_loss,
    )
