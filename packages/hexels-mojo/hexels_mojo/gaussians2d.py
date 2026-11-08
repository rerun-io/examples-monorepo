"""Image-GS's 2D Gaussian projection and top-K rasterizer (hexel-kernel's `gsplat`) on the Mojo kernels.

`project_gaussians_2d_scale_rot` and `rasterize_gaussians_sum` are drop-ins for the `gsplat` functions of the same
names that Image-GS's `model.py` and hexels-src's `image_compression_training.py` import (recipes/hexels-src patches
0003/0004 redirect the imports). `rasterize_gaussians_sum(..., topk_norm=False)` (off in Image-GS's configs) still
calls hexel-kernel. The lower-case `*_forward` / `*_backward` functions mirror the `gsplat.cuda` bindings.
"""

from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from hexels_mojo._launch import launch
from hexels_mojo.raster2d import TOP_K, bin_and_sort_gaussians

TILE: int = 16


def project_gaussians_2d_scale_rot_forward(
    num_points: int, means2d: Tensor, scales2d: Tensor, rotation: Tensor, img_height: int, img_width: int, tile_bounds: tuple[int, int, int]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`gsplat.cuda.project_gaussians_2d_scale_rot_forward`: (xys (N, 2), radii i32 (N), conics (N, 3), num_tiles_hit i32 (N))."""
    del tile_bounds  # 16x16 tiles over the image, as hexel-kernel's compiled constants
    device: torch.device = means2d.device
    xys: Tensor = torch.zeros(num_points, 2, dtype=torch.float32, device=device)
    radii: Tensor = torch.zeros(num_points, dtype=torch.int32, device=device)
    conics: Tensor = torch.zeros(num_points, 3, dtype=torch.float32, device=device)
    num_tiles_hit: Tensor = torch.zeros(num_points, dtype=torch.int32, device=device)
    launch(
        "gaussians2d_project",
        (0, img_width, img_height),
        (means2d.contiguous(), scales2d.contiguous(), rotation.contiguous(), xys, radii, conics, num_tiles_hit),
    )
    return xys, radii, conics, num_tiles_hit


def project_gaussians_2d_scale_rot_backward(
    num_points: int,
    means2d: Tensor,
    scales2d: Tensor,
    rotation: Tensor,
    img_height: int,
    img_width: int,
    radii: Tensor,
    conics: Tensor,
    v_xys: Tensor,
    v_conics: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`gsplat.cuda.project_gaussians_2d_scale_rot_backward`: (v_cov2d (N, 3), v_mean2d (N, 2), v_scale (N, 2), v_rot (N, 1))."""
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=means2d.device)  # noqa: E731
    v_cov2d, v_mean2d, v_scale, v_rot = zeros(num_points, 3), zeros(num_points, 2), zeros(num_points, 2), zeros(num_points, 1)
    launch(
        "gaussians2d_project",
        (1, img_width, img_height),
        (scales2d.contiguous(), rotation.contiguous(), radii.contiguous(), conics.contiguous(), v_xys.contiguous(), v_conics.contiguous(),
         v_cov2d, v_mean2d, v_scale, v_rot),
    )  # fmt: skip
    return v_cov2d, v_mean2d, v_scale, v_rot


def nd_rasterize_forward_topk_norm(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    conics: Tensor,
    colors: Tensor,
) -> tuple[Tensor, Tensor]:
    """`gsplat.cuda.nd_rasterize_forward_topk_norm`: (out_img (H, W, C), pixel_topk i32 (H, W, 10))."""
    del tile_bounds, block
    width, height = img_size[0], img_size[1]
    out_img: Tensor = torch.zeros(height, width, colors.shape[1], dtype=torch.float32, device=xys.device)
    pixel_topk: Tensor = torch.zeros(height, width, TOP_K, dtype=torch.int32, device=xys.device)
    launch(
        "gaussians2d_forward",
        (),
        (gaussian_ids_sorted.to(torch.int32).contiguous(), tile_bins.to(torch.int32).contiguous(), xys.contiguous(), conics.contiguous(),
         colors.contiguous(), out_img, pixel_topk),
    )  # fmt: skip
    return out_img, pixel_topk


def nd_rasterize_backward_topk_norm(
    img_height: int,
    img_width: int,
    block_h: int,
    block_w: int,
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    conics: Tensor,
    colors: Tensor,
    v_output: Tensor,
    pixel_topk: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """`gsplat.cuda.nd_rasterize_backward_topk_norm`: (v_xy (N, 2), v_conic (N, 3), v_colors (N, C))."""
    del img_height, img_width, block_h, block_w, gaussian_ids_sorted, tile_bins  # reads only the saved top-10
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=xys.device)  # noqa: E731
    num_points = xys.shape[0]
    v_xy, v_conic, v_colors = zeros(num_points, 2), zeros(num_points, 3), zeros(num_points, colors.shape[1])
    launch(
        "gaussians2d_backward",
        (),
        (xys.contiguous(), conics.contiguous(), colors.contiguous(), v_output.contiguous(), pixel_topk.to(torch.int32).contiguous(), v_xy,
         v_conic, v_colors),
    )  # fmt: skip
    return v_xy, v_conic, v_colors


class _ProjectGaussians2dScaleRot(Function):
    @staticmethod
    def forward(ctx: Any, means2d: Tensor, scales2d: Tensor, rotation: Tensor, img_height: int, img_width: int, tile_bounds: tuple) -> Any:
        num_points: int = means2d.shape[-2]
        if num_points < 1 or means2d.shape[-1] != 2:
            raise ValueError(f"Invalid shape for means2d: {means2d.shape}")
        xys, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot_forward(
            num_points, means2d, scales2d, rotation, img_height, img_width, tile_bounds
        )
        ctx.img_height, ctx.img_width, ctx.num_points = img_height, img_width, num_points
        ctx.save_for_backward(means2d, scales2d, rotation, radii, conics)
        return xys, radii, conics, num_tiles_hit

    @staticmethod
    def backward(ctx: Any, v_xys: Tensor, v_radii: Tensor, v_conics: Tensor, v_num_tiles_hit: Tensor) -> Any:  # pyrefly: ignore[bad-override]
        means2d, scales2d, rotation, radii, conics = ctx.saved_tensors
        _, v_mean2d, v_scale, v_rot = project_gaussians_2d_scale_rot_backward(
            ctx.num_points, means2d, scales2d, rotation, ctx.img_height, ctx.img_width, radii, conics, v_xys, v_conics
        )
        return v_mean2d, v_scale, v_rot, None, None, None


def project_gaussians_2d_scale_rot(
    means2d: Tensor, scales2d: Tensor, rotation: Tensor, img_height: int, img_width: int, tile_bounds: tuple[int, int, int]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`gsplat.project_gaussians_2d_scale_rot`: (xys, radii, conics, num_tiles_hit); centers are in [0, 1]^2."""
    return _ProjectGaussians2dScaleRot.apply(  # pyrefly: ignore[bad-return]
        means2d.contiguous(), scales2d.contiguous(), rotation.contiguous(), img_height, img_width, tile_bounds
    )


class _RasterizeGaussiansSum(Function):
    @staticmethod
    def forward(ctx: Any, xys: Tensor, radii: Tensor, conics: Tensor, num_tiles_hit: Tensor, colors: Tensor, img_height: int, img_width: int) -> Any:
        num_points: int = xys.shape[0]
        tile_bounds = ((img_width + TILE - 1) // TILE, (img_height + TILE - 1) // TILE, 1)
        cum_tiles_hit: Tensor = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
        num_intersects: int = int(cum_tiles_hit[-1].item())
        ctx.num_intersects = num_intersects
        if num_intersects < 1:
            ctx.save_for_backward(xys, conics, colors)
            return torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
        _, _, _, gaussian_ids_sorted, tile_bins = bin_and_sort_gaussians(num_points, num_intersects, xys, radii, cum_tiles_hit, tile_bounds)
        out_img, pixel_topk = nd_rasterize_forward_topk_norm(
            tile_bounds, (TILE, TILE, 1), (img_width, img_height, 1), gaussian_ids_sorted, tile_bins, xys, conics, colors
        )
        ctx.save_for_backward(xys, conics, colors, pixel_topk)
        return out_img

    @staticmethod
    def backward(ctx: Any, v_out_img: Tensor) -> Any:  # pyrefly: ignore[bad-override]
        if ctx.num_intersects < 1:
            xys, conics, colors = ctx.saved_tensors
            v_xy, v_conic, v_colors = torch.zeros_like(xys), torch.zeros_like(conics), torch.zeros_like(colors)
        else:
            xys, conics, colors, pixel_topk = ctx.saved_tensors
            v_xy, v_conic, v_colors = nd_rasterize_backward_topk_norm(0, 0, TILE, TILE, xys, xys, xys, conics, colors, v_out_img, pixel_topk)
        return v_xy, None, v_conic, None, v_colors, None, None


def rasterize_gaussians_sum(
    xys: Tensor,
    radii: Tensor,
    conics: Tensor,
    num_tiles_hit: Tensor,
    colors: Tensor,
    img_height: int,
    img_width: int,
    BLOCK_H: int = 16,  # noqa: N803 (upstream's names)
    BLOCK_W: int = 16,  # noqa: N803
    topk_norm: bool = False,
) -> Tensor:
    """`gsplat.rasterize_gaussians_sum`: the (H, W, C) image; top-10 normalized blending runs on Mojo."""
    if not topk_norm or (BLOCK_H, BLOCK_W) != (TILE, TILE):
        from gsplat import rasterize_gaussians_sum as cuda_rasterize_gaussians_sum  # pyrefly: ignore[missing-import]

        return cuda_rasterize_gaussians_sum(xys, radii, conics, num_tiles_hit, colors, img_height, img_width, BLOCK_H, BLOCK_W, topk_norm)
    if colors.dtype == torch.uint8:
        colors = colors.float() / 255
    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")
    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")
    return _RasterizeGaussiansSum.apply(  # pyrefly: ignore[bad-return]
        xys.contiguous(), radii.contiguous(), conics.contiguous(), num_tiles_hit.contiguous(), colors.contiguous(), img_height, img_width
    )
