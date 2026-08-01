"""Stage-2 sanity check: rasterize the initial mesh-Gaussians through real views.

Renders the untrained Gaussians (gt-mesh vertices) with each view's pose and
intrinsics and writes side-by-side PNGs against the decoded RGB frames. If the
pose convention or intrinsics handling is wrong, the render is garbage or
misaligned immediately — no training required.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat import rasterization
from jaxtyping import Float32, UInt8
from numpy import ndarray
from torch import Tensor

from rerun_gsplat.apis.mesh_init import ColoredPoints, GaussianInit, gaussians_from_points, load_gt_mesh
from rerun_gsplat.apis.segment_views import SegmentViewsConfig, SplatView, load_segment_views


@dataclass(frozen=True, slots=True)
class Config:
    """Render-overlay check over a handful of views."""

    views: SegmentViewsConfig = field(default_factory=lambda: SegmentViewsConfig(target_view_count=12))
    """Catalog/segment source; a dozen views is plenty for a pose check."""
    max_points: int = 200_000
    """Gaussian count cap for the init cloud."""
    out_dir: Path = Path("/tmp/rerun-gsplat-check-init")
    """Where the side-by-side PNGs land."""


def render_view(init: GaussianInit, view: SplatView, device: str) -> UInt8[ndarray, "h w 3"]:
    """Rasterize the initial Gaussians through one view's camera.

    Args:
        init: Initial Gaussian parameters (CPU, natural form).
        view: Target view supplying pose, intrinsics, and resolution.
        device: CUDA device for gsplat.
    """
    height: int
    width: int
    height, width = int(view.rgb_hwc.shape[0]), int(view.rgb_hwc.shape[1])
    outputs: tuple[Tensor, Tensor, dict[str, Tensor]] = rasterization(
        means=init.means_n3.to(device),
        quats=init.quats_n4.to(device),
        scales=init.log_scales_n3.exp().to(device),
        opacities=init.logit_opacities_n.sigmoid().to(device),
        colors=init.rgbs_n3.to(device),
        viewmats=view.cam_t_world_44[None].to(device),
        Ks=view.k_33[None].to(device),
        width=width,
        height=height,
    )
    rendered: Float32[Tensor, "h w 3"] = outputs[0][0].clamp(0.0, 1.0)
    return (rendered * 255.0).to(torch.uint8).cpu().numpy()


def main(config: Config) -> None:
    """Render every loaded view next to its ground-truth frame."""
    device: str = "cuda"
    views: list[SplatView] = load_segment_views(config.views)
    mesh_points: ColoredPoints = load_gt_mesh(config.views)
    init: GaussianInit = gaussians_from_points(mesh_points, max_points=config.max_points)
    # Move the Gaussians to the GPU once; render_view's .to(device) is then a no-op.
    init = dataclasses.replace(
        init,
        means_n3=init.means_n3.to(device),
        rgbs_n3=init.rgbs_n3.to(device),
        log_scales_n3=init.log_scales_n3.to(device),
        quats_n4=init.quats_n4.to(device),
        logit_opacities_n=init.logit_opacities_n.to(device),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(views):
        rendered_hwc: UInt8[ndarray, "h w 3"] = render_view(init, view, device=device)
        side_by_side: UInt8[ndarray, "h w2 3"] = np.concatenate([view.rgb_hwc.numpy(), rendered_hwc], axis=1)
        out_path: Path = config.out_dir / f"view_{index:03d}.png"
        if not cv2.imwrite(str(out_path), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {out_path}")
    print(f"wrote {len(views)} side-by-side renders to {config.out_dir}")
