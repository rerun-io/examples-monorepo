"""Fused gsplat renderer and its retained two-call parity reference."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from gsplat import rasterization

from gauss_surf.gaussurf_geometry import gaussian_plane_features, plane_depth_from_rendered_features
from gauss_surf.train_gsplat.cache import RasterCamera


@dataclass(frozen=True, slots=True)
class RenderOutput:
    """Appearance and plane-intersection signals from one camera."""

    rgb_hw3: torch.Tensor
    alpha_hw1: torch.Tensor
    center_depth_hw1: torch.Tensor
    plane_features_hw4: torch.Tensor
    surface_depth_hw1: torch.Tensor
    surface_valid_hw1: torch.Tensor
    direct_normal_hw3: torch.Tensor
    depth_normal_hw3: torch.Tensor
    background_3: torch.Tensor
    appearance_info: dict[str, Any]


def fill_depth_holes(depth_hw1: torch.Tensor, alpha_hw1: torch.Tensor) -> torch.Tensor:
    """Fill auxiliary expected-depth holes with Splatfacto's detached maximum."""
    if depth_hw1.shape != alpha_hw1.shape:
        raise ValueError("depth and alpha must have identical [H,W,1] shapes")
    return torch.where(alpha_hw1 > 0.0, depth_hw1, depth_hw1.detach().max())


def normal_from_depth(depth_hw1: torch.Tensor, intrinsics_33: torch.Tensor, valid_hw1: torch.Tensor) -> torch.Tensor:
    """Derive RDF away-from-camera normals with the historical Sobel operator."""
    height: int = depth_hw1.shape[0]
    width: int = depth_hw1.shape[1]
    dtype: torch.dtype = depth_hw1.dtype
    device: torch.device = depth_hw1.device
    pixel_x_hw: torch.Tensor
    pixel_y_hw: torch.Tensor
    pixel_x_hw, pixel_y_hw = torch.meshgrid(
        torch.arange(width, dtype=dtype, device=device) + 0.5,
        torch.arange(height, dtype=dtype, device=device) + 0.5,
        indexing="xy",
    )
    pixels_hw3: torch.Tensor = torch.stack((pixel_x_hw, pixel_y_hw, torch.ones_like(pixel_x_hw)), dim=-1)
    rays_hw3: torch.Tensor = pixels_hw3 @ torch.linalg.inv(intrinsics_33).T
    points_13hw: torch.Tensor = (rays_hw3 * depth_hw1).movedim(-1, 0).unsqueeze(0)
    sobel_x_1133: torch.Tensor = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], dtype=dtype, device=device
    ).unsqueeze(0)
    sobel_y_1133: torch.Tensor = sobel_x_1133.transpose(-1, -2)
    padded_13hw: torch.Tensor = F.pad(points_13hw, (1, 1, 1, 1), mode="replicate")
    gradient_x_13hw: torch.Tensor = F.conv2d(padded_13hw, sobel_x_1133.repeat(3, 1, 1, 1), groups=3)
    gradient_y_13hw: torch.Tensor = F.conv2d(padded_13hw, sobel_y_1133.repeat(3, 1, 1, 1), groups=3)
    normal_13hw: torch.Tensor = F.normalize(torch.cross(gradient_x_13hw, gradient_y_13hw, dim=1), dim=1, eps=1e-6)
    normal_hw3: torch.Tensor = normal_13hw.squeeze(0).movedim(0, -1)
    return torch.where(valid_hw1.expand_as(normal_hw3), normal_hw3, torch.zeros_like(normal_hw3))


def _render_splats(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    camera: RasterCamera,
    *,
    downscale: int,
    sh_degree: int,
    background_3: torch.Tensor,
    absgrad: bool,
    fused: bool,
) -> RenderOutput:
    """Render shared outputs with either fused or reference rasterization calls."""
    device: torch.device = splats["means"].device
    viewmat_144: torch.Tensor = camera.viewmat_44.to(device).unsqueeze(0)
    intrinsics_133: torch.Tensor = camera.K_33.to(device).unsqueeze(0).clone()
    intrinsics_133[:, :2] /= downscale
    width: int = camera.width // downscale
    height: int = camera.height // downscale
    quats_n4: torch.Tensor = F.normalize(splats["quats"], dim=-1)
    scales_n3: torch.Tensor = torch.exp(splats["scales"])
    opacity_n: torch.Tensor = torch.sigmoid(splats["opacities"]).squeeze(-1)
    colors_nk3: torch.Tensor = torch.cat((splats["sh0"], splats["shN"]), dim=1)
    plane_features_n4: torch.Tensor = gaussian_plane_features(
        splats["means"], splats["scales"], quats_n4, viewmat_144.squeeze(0)
    )

    rendered_appearance_1hw4: torch.Tensor
    appearance_alpha_1hw1: torch.Tensor
    appearance_info: dict[str, Any]
    rendered_plane_1hw4: torch.Tensor
    plane_alpha_1hw1: torch.Tensor
    if fused:
        rendered_appearance_1hw4, appearance_alpha_1hw1, appearance_info = rasterization(
            means=splats["means"],
            quats=quats_n4,
            scales=scales_n3,
            opacities=opacity_n,
            colors=colors_nk3,
            viewmats=viewmat_144,
            Ks=intrinsics_133,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=absgrad,
            rasterize_mode="classic",
            extra_signals=plane_features_n4,
            extra_signals_sh_degree=None,
        )
        rendered_plane_1hw4 = appearance_info["render_extra_signals"]
        plane_alpha_1hw1 = appearance_alpha_1hw1
    else:
        rendered_appearance_1hw4, appearance_alpha_1hw1, appearance_info = rasterization(
            means=splats["means"],
            quats=quats_n4,
            scales=scales_n3,
            opacities=opacity_n,
            colors=colors_nk3,
            viewmats=viewmat_144,
            Ks=intrinsics_133,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB+ED",
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=absgrad,
            rasterize_mode="classic",
        )
        rendered_plane_1hw4, plane_alpha_1hw1, _plane_info = rasterization(
            means=splats["means"],
            quats=quats_n4,
            scales=scales_n3,
            opacities=opacity_n,
            colors=plane_features_n4,
            viewmats=viewmat_144,
            Ks=intrinsics_133,
            width=width,
            height=height,
            packed=False,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode="classic",
        )
    rgb_hw3: torch.Tensor = rendered_appearance_1hw4[0, ..., :3] + (1.0 - appearance_alpha_1hw1[0]) * background_3
    rgb_hw3 = rgb_hw3.clamp(0.0, 1.0)
    center_depth_hw1: torch.Tensor = fill_depth_holes(rendered_appearance_1hw4[0, ..., 3:4], appearance_alpha_1hw1[0])
    surface = plane_depth_from_rendered_features(
        rendered_plane_1hw4.squeeze(0),
        plane_alpha_1hw1.squeeze(0),
        intrinsics_133.squeeze(0),
        min_alpha=0.01,
        min_denominator=1e-4,
    )
    depth_normal_hw3: torch.Tensor = normal_from_depth(surface.depth_hw1, intrinsics_133.squeeze(0), surface.valid_hw1)
    return RenderOutput(
        rgb_hw3=rgb_hw3,
        alpha_hw1=appearance_alpha_1hw1.squeeze(0),
        center_depth_hw1=center_depth_hw1,
        plane_features_hw4=rendered_plane_1hw4.squeeze(0),
        surface_depth_hw1=surface.depth_hw1,
        surface_valid_hw1=surface.valid_hw1,
        direct_normal_hw3=surface.normal_hw3,
        depth_normal_hw3=depth_normal_hw3,
        background_3=background_3,
        appearance_info=appearance_info,
    )


def render_splats_two_call(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    camera: RasterCamera,
    *,
    downscale: int,
    sh_degree: int,
    background_3: torch.Tensor,
    absgrad: bool,
) -> RenderOutput:
    """Render appearance and plane features with the retained two-call reference."""
    return _render_splats(
        splats,
        camera,
        downscale=downscale,
        sh_degree=sh_degree,
        background_3=background_3,
        absgrad=absgrad,
        fused=False,
    )


def render_splats(
    splats: torch.nn.ParameterDict | dict[str, torch.Tensor],
    camera: RasterCamera,
    *,
    downscale: int,
    sh_degree: int,
    background_3: torch.Tensor,
    absgrad: bool,
) -> RenderOutput:
    """Render RGB, expected center depth, and four plane channels in one 8-channel pass."""
    return _render_splats(
        splats,
        camera,
        downscale=downscale,
        sh_degree=sh_degree,
        background_3=background_3,
        absgrad=absgrad,
        fused=True,
    )
