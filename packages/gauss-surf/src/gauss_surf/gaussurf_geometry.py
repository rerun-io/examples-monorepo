"""Differentiable plane geometry used by the GausSurf-compatible renderer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def quat_to_rotmat(quat_n4: torch.Tensor) -> torch.Tensor:
    """Convert normalized WXYZ quaternions into rotation matrices."""
    w_n, x_n, y_n, z_n = torch.unbind(quat_n4, dim=-1)
    values_n9 = torch.stack((1 - 2 * (y_n**2 + z_n**2), 2 * (x_n * y_n - w_n * z_n), 2 * (x_n * z_n + w_n * y_n), 2 * (x_n * y_n + w_n * z_n), 1 - 2 * (x_n**2 + z_n**2), 2 * (y_n * z_n - w_n * x_n), 2 * (x_n * z_n - w_n * y_n), 2 * (y_n * z_n + w_n * x_n), 1 - 2 * (x_n**2 + y_n**2)), dim=-1)
    return values_n9.reshape((*quat_n4.shape[:-1], 3, 3))


@dataclass(frozen=True, slots=True)
class SurfaceRender:
    """Surface depth, direct Gaussian normal, and their shared validity mask."""

    depth_hw1: torch.Tensor
    normal_hw3: torch.Tensor
    valid_hw1: torch.Tensor


def smallest_axis_world_normals(
    log_scales_n3: torch.Tensor,
    quats_wxyz_n4: torch.Tensor,
) -> torch.Tensor:
    """Return each Gaussian rotation axis associated with its minimum scale."""

    if log_scales_n3.shape[-1] != 3 or quats_wxyz_n4.shape[-1] != 4:
        raise ValueError("expected Gaussian scales [N,3] and WXYZ quaternions [N,4]")
    rotations_n33 = quat_to_rotmat(F.normalize(quats_wxyz_n4, dim=-1))
    minimum_axis_n11 = log_scales_n3.argmin(dim=-1).reshape(-1, 1, 1)
    minimum_axis_n31 = minimum_axis_n11.expand(-1, 3, 1)
    normals_n3 = rotations_n33.gather(dim=2, index=minimum_axis_n31).squeeze(2)
    return F.normalize(normals_n3, dim=-1)


def gaussian_plane_features(
    means_world_n3: torch.Tensor,
    log_scales_n3: torch.Tensor,
    quats_wxyz_n4: torch.Tensor,
    world_to_camera_44: torch.Tensor,
) -> torch.Tensor:
    """Build camera-space normal and signed plane-offset features per Gaussian.

    The minimum-scale axis is sign-oriented along the ray from the camera to
    the Gaussian. This is the opposite sign of PGSR's camera-facing normal but
    represents the same plane and matches this repository's depth-normal
    convention. The plane equation is ``normal dot point = offset``.
    """

    if world_to_camera_44.shape != (4, 4):
        raise ValueError("world_to_camera must have shape [4,4]")
    camera_to_world_44 = torch.linalg.inv(world_to_camera_44)
    camera_center_world_3 = camera_to_world_44[:3, 3]
    normals_world_n3 = smallest_axis_world_normals(log_scales_n3, quats_wxyz_n4)
    camera_rays_world_n3 = means_world_n3 - camera_center_world_3
    flip_n1 = (normals_world_n3 * camera_rays_world_n3).sum(dim=-1, keepdim=True) < 0
    normals_world_n3 = torch.where(flip_n1, -normals_world_n3, normals_world_n3)

    rotation_33 = world_to_camera_44[:3, :3]
    translation_3 = world_to_camera_44[:3, 3]
    means_camera_n3 = means_world_n3 @ rotation_33.T + translation_3
    normals_camera_n3 = F.normalize(normals_world_n3 @ rotation_33.T, dim=-1)
    offsets_n1 = (normals_camera_n3 * means_camera_n3).sum(dim=-1, keepdim=True)
    return torch.cat((normals_camera_n3, offsets_n1), dim=-1)


def plane_depth_from_rendered_features(
    rendered_features_hw4: torch.Tensor,
    alpha_hw1: torch.Tensor,
    intrinsics_33: torch.Tensor,
    *,
    min_alpha: float = 0.01,
    min_denominator: float = 1e-4,
    min_depth: float = 0.01,
    max_depth: float = 1e4,
) -> SurfaceRender:
    """Intersect camera rays with alpha-composited Gaussian tangent planes."""

    if rendered_features_hw4.ndim != 3 or rendered_features_hw4.shape[-1] != 4:
        raise ValueError("rendered plane features must have shape [H,W,4]")
    if alpha_hw1.shape != (*rendered_features_hw4.shape[:2], 1):
        raise ValueError("alpha must have shape [H,W,1]")
    if intrinsics_33.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3,3]")

    height, width = rendered_features_hw4.shape[:2]
    dtype = rendered_features_hw4.dtype
    device = rendered_features_hw4.device
    pixel_x, pixel_y = torch.meshgrid(
        torch.arange(width, dtype=dtype, device=device) + 0.5,
        torch.arange(height, dtype=dtype, device=device) + 0.5,
        indexing="xy",
    )
    homogeneous_hw3 = torch.stack(
        (pixel_x, pixel_y, torch.ones_like(pixel_x)),
        dim=-1,
    )
    rays_hw3 = homogeneous_hw3 @ torch.linalg.inv(intrinsics_33).T

    normal_sum_hw3 = rendered_features_hw4[..., :3]
    offset_sum_hw1 = rendered_features_hw4[..., 3:4]
    denominator_hw1 = (normal_sum_hw3 * rays_hw3).sum(dim=-1, keepdim=True)
    denominator_ok_hw1 = denominator_hw1.abs() >= min_denominator
    safe_denominator_hw1 = torch.where(
        denominator_ok_hw1,
        denominator_hw1,
        torch.ones_like(denominator_hw1),
    )
    depth_hw1 = offset_sum_hw1 / safe_denominator_hw1
    normal_length_hw1 = torch.linalg.vector_norm(normal_sum_hw3, dim=-1, keepdim=True)
    normal_hw3 = F.normalize(normal_sum_hw3, dim=-1)
    valid_hw1 = (
        (alpha_hw1 >= min_alpha)
        & denominator_ok_hw1
        & (normal_length_hw1 >= min_denominator)
        & torch.isfinite(depth_hw1)
        & (depth_hw1 >= min_depth)
        & (depth_hw1 <= max_depth)
    )
    return SurfaceRender(
        depth_hw1=torch.where(valid_hw1, depth_hw1, torch.zeros_like(depth_hw1)),
        normal_hw3=torch.where(valid_hw1.expand_as(normal_hw3), normal_hw3, torch.zeros_like(normal_hw3)),
        valid_hw1=valid_hw1,
    )
