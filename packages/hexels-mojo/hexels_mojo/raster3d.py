"""hexel-kernel's 3D hexel rasterizer forward on the Mojo kernels (`mojo/raster3d.mojo`).

`rasterize_hexels3d_forward` / `rasterize_hexels3d_forward_cached` take the arguments of the
`gsplat.cuda_hexels3d._C` bindings and return their outputs bit for bit, saved state included, for the base forward
without prefiltering (the default training and rendering path). `rasterize_hexels3d_backward_cached` takes that saved
state and returns the binding's twelve gradients (to a float-atomics tolerance; `mojo/raster3d_backward.mojo`).
recipes/hexels-src patch 0006 wraps hexel-kernel's `_C` in `overlay`: those three calls come from Mojo, everything
else (prefiltering, the contribution / overdraw / surface / owner variants and their backward, the uncached
backward) stays CUDA.
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


def rasterize_hexels3d_backward_cached(
    means3d: Tensor,
    means2d: Tensor,
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
    grad_out_color: Tensor,
    grad_out_overdraw: Tensor,
    grad_out_surf_depth: Tensor,
    grad_out_surf_alpha: Tensor,
    grad_out_surf_normals: Tensor,
    image_width: int,
    image_height: int,
    tan_fovx: float,
    tan_fovy: float,
    softness: float,
    scale_modifier: float,
    color_model: int,
    view_model: int,
    use_view_grads: bool,
    overdraw_base_count: int,
    prefilter_edge_on: bool,
    conservative_bounds: bool,
    debug: bool,
    saved: list[Tensor],
) -> tuple[Tensor, ...]:
    """The binding's (means3d, means2d, scales, rotations, colors, color_grads, color_quads, view_grads, specular_ks,
    specular_log_s, opacities, softnesses) gradients from the cached forward's `saved` list."""
    del viewmatrix, tan_fovx, tan_fovy, softness, overdraw_base_count, conservative_bounds, debug  # unused by CUDA too
    if prefilter_edge_on or any(t.numel() for t in (grad_out_overdraw, grad_out_surf_depth, grad_out_surf_alpha, grad_out_surf_normals)):
        raise NotImplementedError("hexels_mojo.raster3d: prefiltering and auxiliary gradients are not ported yet")
    radii, transmats, opacities_out, softnesses_out, view_dirs, normals, _, ranges, values_sorted, final_transmittance, last_index = saved
    zeros_like = lambda t: torch.zeros_like(t, dtype=torch.float32)  # noqa: E731
    num_points: int = means3d.shape[0]
    grad_means3d = torch.zeros(num_points, 3, dtype=torch.float32, device=means3d.device)
    grad_means2d, grad_scales, grad_rotations, grad_colors = zeros_like(means2d), zeros_like(scales), zeros_like(rotations), zeros_like(colors)
    grad_color_grads, grad_color_quads, grad_view_grads = zeros_like(color_grads), zeros_like(color_quads), zeros_like(view_grads)
    grad_specular_ks, grad_specular_log_s, grad_opacities = zeros_like(specular_ks), zeros_like(specular_log_s), zeros_like(opacities)
    grad_softnesses = zeros_like(softnesses)
    grads = (grad_means3d, grad_means2d, grad_scales, grad_rotations, grad_colors, grad_color_grads, grad_color_quads, grad_view_grads,
             grad_specular_ks, grad_specular_log_s, grad_opacities, grad_softnesses)  # fmt: skip
    if values_sorted.numel() == 0:
        return grads
    grad_transmats = torch.zeros(num_points, 9, dtype=torch.float32, device=means3d.device)
    grad_view_dirs = torch.zeros(num_points, 3, dtype=torch.float32, device=means3d.device)
    has_normals: bool = normals.numel() > 0
    grad_normals = torch.zeros(num_points, 3, dtype=torch.float32, device=means3d.device) if has_normals else grad_view_dirs
    stride: int = means2d.shape[1] if means2d.numel() else 3
    pick = lambda t: t.contiguous() if t.numel() else colors  # noqa: E731
    launch(
        "raster3d_backward",
        (0, image_width, image_height, color_model, view_model, int(use_view_grads), 3 if softnesses_out.dim() == 2 else 1,
         int(softnesses.numel() > 0), int(stride == 4)),
        (ranges, values_sorted, colors.contiguous(), pick(color_grads), pick(color_quads), pick(view_grads), pick(specular_ks),
         pick(specular_log_s), view_dirs, pick(normals), transmats, opacities_out, softnesses_out, background.contiguous(),
         grad_out_color.contiguous(), final_transmittance, last_index, grad_colors, pick(grad_color_grads), pick(grad_color_quads),
         pick(grad_view_grads), pick(grad_specular_ks), pick(grad_specular_log_s), grad_opacities, pick(grad_softnesses), grad_transmats,
         pick(grad_means2d), grad_view_dirs, grad_normals),
    )  # fmt: skip
    if means2d.numel():
        launch("raster3d_backward", (1, image_width, image_height, stride), (radii, transmats, grad_transmats, grad_means2d))
    launch(
        "raster3d_backward",
        (2, image_width, image_height, scale_modifier, int(has_normals)),
        (means3d.contiguous(), scales.contiguous(), rotations.contiguous(), projmatrix.contiguous(), campos.contiguous(), radii,
         grad_transmats, grad_view_dirs, grad_normals, grad_means3d, grad_scales, grad_rotations),
    )  # fmt: skip
    return grads


PREFILTER_ARG: int = 24
"""Index of `prefilter_edge_on` among the forward bindings' arguments."""
BACKWARD_PREFILTER_ARG: int = 31
"""Index of `prefilter_edge_on` among the backward bindings' arguments (grad_out_overdraw .. surf normals: 17-20)."""


class _Overlay:
    """hexel-kernel's `gsplat.cuda_hexels3d._C` with the two base forwards and the cached backward on Mojo (CUDA when
    prefiltering is on or auxiliary gradients flow).

    The cached forward returns the binding's saved state bit for bit, so either backward consumes it unchanged.
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

    def rasterize_hexels3d_backward_cached(self, *args: object) -> object:
        aux: list[Tensor] = list(args[17:21])  # pyrefly: ignore[bad-assignment]
        if args[BACKWARD_PREFILTER_ARG] or any(t.numel() for t in aux):
            return self._module.rasterize_hexels3d_backward_cached(*args)  # pyrefly: ignore[missing-attribute]
        return rasterize_hexels3d_backward_cached(*args)  # pyrefly: ignore[bad-argument-type]


def overlay(module: object) -> Any:
    """`module` (hexel-kernel's `gsplat.cuda_hexels3d._C`) with the Mojo forward and cached backward in front."""
    return _Overlay(module)
