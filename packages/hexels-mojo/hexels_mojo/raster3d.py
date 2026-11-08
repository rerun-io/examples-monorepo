"""hexel-kernel's 3D hexel rasterizer forward on the Mojo kernels (`mojo/raster3d.mojo`).

`rasterize_hexels3d_forward` / `rasterize_hexels3d_forward_cached` take the arguments of the
`gsplat.cuda_hexels3d._C` bindings and return their outputs bit for bit, saved state included, for the base forward
without prefiltering (the default training and rendering path). recipes/hexels-src patch 0006 wraps hexel-kernel's
`_C` in `overlay`: those two calls come from Mojo, everything else (prefiltering, the contribution / overdraw /
surface / owner variants, and every backward) stays CUDA.
"""

from typing import Any

import torch
from torch import Tensor

from hexels_mojo._launch import launch

TILE: int = 16


def _forward(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_quads: Tensor,
    view_grads: Tensor,
    specular_ks: Tensor,
    specular_log_s: Tensor,
    opacities: Tensor,
    softnesses: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    softness: float,
    scale_modifier: float,
    color_model: int,
    view_model: int,
    use_view_grads: bool,
    prefilter_edge_on: bool,
    conservative_bounds: bool,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    if prefilter_edge_on:
        raise NotImplementedError("hexels_mojo.raster3d: prefilter_edge_on is not ported yet")
    num_points: int = means3d.shape[0]
    device: torch.device = means3d.device
    f32 = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)  # noqa: E731
    i32 = lambda *shape: torch.zeros(*shape, dtype=torch.int32, device=device)  # noqa: E731
    stride: int = 3 if softnesses.dim() == 2 else 1
    radii, tile_radii, tiles_touched = i32(num_points), i32(num_points, 2), i32(num_points)
    points_xy, depths, transmats, opacities_out = f32(num_points, 2), f32(num_points), f32(num_points, 9), f32(num_points)
    softnesses_out = f32(num_points, 3) if stride == 3 else f32(num_points)
    view_dirs = f32(num_points, 3)
    want_normals: bool = view_model == 2
    normals = f32(num_points, 3) if want_normals else f32(0)
    placeholder = means3d
    launch(
        "raster3d",
        (0, image_width, image_height, scale_modifier, softness, stride, int(softnesses.numel() > 0), int(conservative_bounds), int(want_normals)),
        (means3d, scales, rotations, opacities, softnesses if softnesses.numel() else placeholder, viewmatrix, projmatrix, campos, radii,
         tile_radii, points_xy, depths, transmats, tiles_touched, opacities_out, softnesses_out, view_dirs,
         normals if want_normals else placeholder),
    )  # fmt: skip
    offsets: Tensor = torch.cumsum(tiles_touched, dim=0, dtype=torch.int32)
    num_rendered: int = int(offsets[-1].item()) if num_points else 0
    tiles_x, tiles_y = (image_width + TILE - 1) // TILE, (image_height + TILE - 1) // TILE
    ranges: Tensor = i32(tiles_x * tiles_y, 2)
    values_sorted: Tensor = i32(0)
    out_color = f32(3, image_height, image_width)
    final_transmittance, last_index = f32(image_height, image_width), i32(image_height, image_width)
    if num_rendered:
        keys = torch.zeros(num_rendered, dtype=torch.int64, device=device)
        values = i32(num_rendered)
        launch("raster3d", (1, image_width, image_height), (points_xy, depths, offsets, radii, tile_radii, keys, values))
        # cub's radix sort is stable on the (tile << 32 | depth bits) keys; so is torch.sort(stable=True).
        keys_sorted, order = torch.sort(keys, stable=True)
        values_sorted = torch.gather(values, 0, order)
        launch("raster2d_binning", (1, tiles_x, tiles_y), (keys_sorted, ranges))
    launch(
        "raster3d",
        (2, image_width, image_height, color_model, view_model, int(use_view_grads), stride),
        (ranges, values_sorted if num_rendered else ranges, colors, color_grads if color_grads.numel() else placeholder,
         color_quads if color_quads.numel() else placeholder, view_grads if view_grads.numel() else placeholder,
         specular_ks if specular_ks.numel() else placeholder, specular_log_s if specular_log_s.numel() else placeholder, view_dirs,
         normals if want_normals else placeholder, transmats, opacities_out, softnesses_out, background, out_color, final_transmittance,
         last_index),
    )  # fmt: skip
    saved = [radii, transmats, opacities_out, softnesses_out, view_dirs, normals, f32(0), ranges, values_sorted, final_transmittance, last_index]
    return out_color, radii, saved


def render(
    ranges: Tensor,
    values_sorted: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_quads: Tensor,
    view_grads: Tensor,
    specular_ks: Tensor,
    specular_log_s: Tensor,
    view_dirs: Tensor,
    normals: Tensor,
    transmats: Tensor,
    opacities: Tensor,
    softnesses: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    color_model: int,
    view_model: int,
    use_view_grads: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """The render stage alone, from the preprocess / binning intermediates: (out_color, final_transmittance, last_index)."""
    device: torch.device = colors.device
    out_color = torch.zeros(3, image_height, image_width, dtype=torch.float32, device=device)
    final_transmittance = torch.zeros(image_height, image_width, dtype=torch.float32, device=device)
    last_index = torch.zeros(image_height, image_width, dtype=torch.int32, device=device)
    pick = lambda t: t.contiguous() if t.numel() else colors  # noqa: E731
    launch(
        "raster3d",
        (2, image_width, image_height, color_model, view_model, int(use_view_grads), 3 if softnesses.dim() == 2 else 1),
        (ranges.contiguous(), pick(values_sorted), colors.contiguous(), pick(color_grads), pick(color_quads), pick(view_grads), pick(specular_ks),
         pick(specular_log_s), view_dirs.contiguous(), pick(normals), transmats.contiguous(), opacities.contiguous(), softnesses.contiguous(),
         background.contiguous(), out_color, final_transmittance, last_index),
    )  # fmt: skip
    return out_color, final_transmittance, last_index


def rasterize_hexels3d_forward_cached(
    means3d: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_quads: Tensor,
    view_grads: Tensor,
    specular_ks: Tensor,
    specular_log_s: Tensor,
    opacities: Tensor,
    softnesses: Tensor,
    viewmatrix: Tensor,
    projmatrix: Tensor,
    campos: Tensor,
    background: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    scale_modifier: float,
    color_model: int,
    view_model: int,
    use_view_grads: bool,
    prefilter_edge_on: bool,
    conservative_bounds: bool,
    debug: bool,
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """(out_color (3, H, W), radii (P), saved): saved is the binding's list (radii, transmats, opacities, softnesses,
    view_dirs, normals, points_xy_center, ranges, values_sorted, final_transmittance, last_index)."""
    del tan_fovx, tan_fovy, debug  # unused by the CUDA kernels too
    return _forward(
        means3d, scales, rotations, colors, color_grads, color_quads, view_grads, specular_ks, specular_log_s, opacities, softnesses, viewmatrix,
        projmatrix, campos, background, image_width, image_height, softness, scale_modifier, color_model, view_model, use_view_grads,
        prefilter_edge_on, conservative_bounds,
    )  # fmt: skip


def rasterize_hexels3d_forward(*args: object) -> tuple[Tensor, Tensor]:
    """`rasterize_hexels3d_forward`: the cached forward's (out_color, radii), with the binding's 27 arguments."""
    out_color, radii, _ = rasterize_hexels3d_forward_cached(*args)  # pyrefly: ignore[bad-argument-type]
    return out_color, radii


PREFILTER_ARG: int = 24
"""Index of `prefilter_edge_on` among the forward bindings' arguments."""


class _Overlay:
    """hexel-kernel's `gsplat.cuda_hexels3d._C` with the two base forwards on Mojo (CUDA when prefiltering is on).

    The cached forward returns the binding's saved state bit for bit, so `rasterize_hexels3d_backward_cached`
    (still CUDA) consumes it unchanged.
    """

    def __init__(self, module: object) -> None:
        self._module = module

    def __getattr__(self, name: str) -> object:
        return getattr(self._module, name)

    def rasterize_hexels3d_forward(self, *args: object) -> object:
        if args[PREFILTER_ARG]:
            return self._module.rasterize_hexels3d_forward(*args)  # pyrefly: ignore[missing-attribute]
        return rasterize_hexels3d_forward(*args)

    def rasterize_hexels3d_forward_cached(self, *args: object) -> object:
        if args[PREFILTER_ARG]:
            return self._module.rasterize_hexels3d_forward_cached(*args)  # pyrefly: ignore[missing-attribute]
        return rasterize_hexels3d_forward_cached(*args)  # pyrefly: ignore[bad-argument-type]


def overlay(module: object) -> Any:
    """`module` (hexel-kernel's `gsplat.cuda_hexels3d._C`) with the Mojo forward in front."""
    return _Overlay(module)
