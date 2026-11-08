"""hexel-kernel's 2D hexel rasterizer, the color-gradient family, on the Mojo kernels (`mojo/raster2d.mojo`).

The functions take the arguments and return the tensors of the `gsplat.cuda` bindings and `gsplat.utils`
helpers of the same names, so `gsplat_hexels.py` can call either. They cover Image-GS's rich 2D fit
(`hexels-fit-2d --variant rich`: color gradients, per-hexel softness); the other hexel-kernel kernels are CUDA.
"""

from typing import Any

import torch
from torch import Tensor

from hexels_mojo._launch import launch

TOP_K: int = 10


def bin_and_sort_gaussians(
    num_points: int, num_intersects: int, xys: Tensor, radii: Tensor, cum_tiles_hit: Tensor, tile_bounds: tuple[int, int, int]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """`gsplat.utils.bin_and_sort_gaussians`: (isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins)."""
    device: torch.device = xys.device
    isect_ids: Tensor = torch.zeros(num_intersects, dtype=torch.int64, device=device)
    gaussian_ids: Tensor = torch.zeros(num_intersects, dtype=torch.int32, device=device)
    launch(
        "raster2d_binning",
        (0, tile_bounds[0], tile_bounds[1]),
        (xys.contiguous(), radii.to(torch.int32).contiguous(), cum_tiles_hit.to(torch.int32).contiguous(), isect_ids, gaussian_ids),
    )
    isect_ids_sorted, order = torch.sort(isect_ids)
    gaussian_ids_sorted: Tensor = torch.gather(gaussian_ids, 0, order)
    tile_bins: Tensor = torch.zeros(tile_bounds[0] * tile_bounds[1], 2, dtype=torch.int32, device=device)
    if num_intersects:
        launch("raster2d_binning", (1, tile_bounds[0], tile_bounds[1]), (isect_ids_sorted, tile_bins))
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins


def _softness_args(softnesses: Tensor | None, num_points: int) -> tuple[int, Tensor | None]:
    if softnesses is None or softnesses.numel() == 0:
        return 0, None
    if softnesses.dim() == 1 and softnesses.shape[0] == num_points:
        return 1, softnesses.contiguous()
    if softnesses.dim() == 2 and softnesses.shape == (num_points, 3):
        return 3, softnesses.contiguous()
    raise ValueError("softnesses must have dimensions (num_points,) or (num_points, 3)")


def _optional(t: Tensor, num_points: int, name: str, width: int) -> Tensor | None:
    if t.numel() == 0:
        return None
    if t.numel() != num_points * width:
        raise ValueError(f"{name} must have {num_points * width} elements, got {t.numel()}")
    return t.contiguous()


def _forward(
    img_size: tuple[int, int, int],
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_grad_scale: float,
    aa_kernel_size: float,
    softnesses: Tensor | None,
    softness_default: float,
    sharpness: Tensor,
    emission: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    width, height = img_size[0], img_size[1]
    num_points, channels = colors.shape
    if color_grads.shape != (num_points, 2, channels):
        raise ValueError("color_grads must have dimensions (num_points, 2, channels)")
    stride, soft = _softness_args(softnesses, num_points)
    sharp = _optional(sharpness, num_points, "sharpness", 1)
    emit = _optional(emission, num_points, "emission", channels)
    device: torch.device = xys.device
    out_img: Tensor = torch.zeros(height, width, channels, dtype=torch.float32, device=device)
    pixel_topk: Tensor = torch.zeros(height, width, TOP_K, dtype=torch.int32, device=device)
    weights_sum: Tensor = torch.zeros(height, width, dtype=torch.float32, device=device)
    entropy: Tensor = torch.zeros(height, width, dtype=torch.float32, device=device)
    placeholder: Tensor = xys
    launch(
        "raster2d_forward",
        (color_grad_scale, aa_kernel_size, stride, softness_default, int(sharp is not None), int(emit is not None)),
        (
            gaussian_ids_sorted.to(torch.int32).contiguous(), tile_bins.to(torch.int32).contiguous(), xys.contiguous(), scales.contiguous(),
            rotation.contiguous(), colors.contiguous(), color_grads.contiguous(), soft if soft is not None else placeholder,
            sharp if sharp is not None else placeholder, emit if emit is not None else placeholder, out_img, pixel_topk, weights_sum, entropy,
        ),
    )  # fmt: skip
    return out_img, pixel_topk, weights_sum, entropy


def nd_rasterize_forward_topk_norm_hexels_color_grad(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_grad_scale: float,
    aa_kernel_size: float,
    softness: float,
    sharpness: Tensor,
    emission: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One scalar softness. Returns (out_img (H, W, C), pixel_topk (H, W, 10), weights_sum (H, W), entropy (H, W))."""
    del tile_bounds, block  # fixed 16x16 tiles, as hexel-kernel's compiled constants
    return _forward(
        img_size, gaussian_ids_sorted, tile_bins, xys, scales, rotation, colors, color_grads, color_grad_scale, aa_kernel_size, None, softness,
        sharpness, emission,
    )  # fmt: skip


def nd_rasterize_forward_topk_norm_hexels_color_grad_softness(
    tile_bounds: tuple[int, int, int],
    block: tuple[int, int, int],
    img_size: tuple[int, int, int],
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_grad_scale: float,
    aa_kernel_size: float,
    softnesses: Tensor,
    softness_default: float,
    sharpness: Tensor,
    emission: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-hexel (N,) or per-edge (N, 3) softness; outputs as the scalar variant."""
    del tile_bounds, block
    return _forward(
        img_size, gaussian_ids_sorted, tile_bins, xys, scales, rotation, colors, color_grads, color_grad_scale, aa_kernel_size, softnesses,
        softness_default, sharpness, emission,
    )  # fmt: skip


def _backward(
    xys: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    softnesses: Tensor | None,
    color_grad_scale: float,
    v_output: Tensor,
    v_weights_sum: Tensor,
    v_entropy: Tensor,
    pixel_topk: Tensor,
    aa_kernel_size: float,
    softness_default: float,
    sharpness: Tensor,
    emission: Tensor,
) -> list[Tensor]:
    num_points, channels = colors.shape
    stride, soft = _softness_args(softnesses, num_points)
    sharp = _optional(sharpness, num_points, "sharpness", 1)
    emit = _optional(emission, num_points, "emission", channels)
    device: torch.device = xys.device
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)  # noqa: E731
    v_xy, v_scale, v_rot = zeros(num_points, 2), zeros(num_points, 2), zeros(num_points, 1)
    v_colors, v_color_grads = zeros(num_points, channels), zeros(num_points, 2, channels)
    v_emission: Tensor = zeros(num_points, channels) if emit is not None else zeros(0)
    v_sharpness: Tensor = zeros(num_points, 1)
    v_softness: Tensor = zeros(*soft.shape) if soft is not None else zeros(0)
    placeholder: Tensor = xys
    launch(
        "raster2d_backward",
        (color_grad_scale, aa_kernel_size, stride, softness_default, int(sharp is not None), int(emit is not None)),
        (
            xys.contiguous(), scales.contiguous(), rotation.contiguous(), colors.contiguous(), color_grads.contiguous(),
            emit if emit is not None else placeholder, v_output.contiguous(), v_weights_sum.contiguous(), v_entropy.contiguous(),
            pixel_topk.to(torch.int32).contiguous(), soft if soft is not None else placeholder, sharp if sharp is not None else placeholder,
            v_xy, v_scale, v_rot, v_colors, v_color_grads, v_emission if emit is not None else placeholder, v_sharpness,
            v_softness if soft is not None else placeholder,
        ),
    )  # fmt: skip
    return [v_xy, v_scale, v_rot, v_colors, v_color_grads, v_emission, v_sharpness, v_softness]


def nd_rasterize_backward_topk_norm_hexels_color_grad(
    img_height: int,
    img_width: int,
    block_h: int,
    block_w: int,
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    color_grad_scale: float,
    v_output: Tensor,
    v_weights_sum: Tensor,
    v_entropy: Tensor,
    pixel_topk: Tensor,
    aa_kernel_size: float,
    softness: float,
    sharpness: Tensor,
    emission: Tensor,
) -> tuple[Tensor, ...]:
    """(v_xy, v_scale, v_rot, v_colors, v_color_grads, v_emission, v_sharpness) for one scalar softness."""
    del img_height, img_width, block_h, block_w, gaussian_ids_sorted, tile_bins  # the backward reads only the saved top-10
    grads = _backward(
        xys, scales, rotation, colors, color_grads, None, color_grad_scale, v_output, v_weights_sum, v_entropy, pixel_topk, aa_kernel_size,
        softness, sharpness, emission,
    )  # fmt: skip
    return tuple(grads[:7])


def nd_rasterize_backward_topk_norm_hexels_color_grad_softness(
    img_height: int,
    img_width: int,
    block_h: int,
    block_w: int,
    gaussian_ids_sorted: Tensor,
    tile_bins: Tensor,
    xys: Tensor,
    scales: Tensor,
    rotation: Tensor,
    colors: Tensor,
    color_grads: Tensor,
    softnesses: Tensor,
    color_grad_scale: float,
    v_output: Tensor,
    v_weights_sum: Tensor,
    v_entropy: Tensor,
    pixel_topk: Tensor,
    aa_kernel_size: float,
    softness_default: float,
    sharpness: Tensor,
    emission: Tensor,
) -> tuple[Tensor, ...]:
    """As the scalar variant, plus v_softness shaped like `softnesses`."""
    del img_height, img_width, block_h, block_w, gaussian_ids_sorted, tile_bins
    return tuple(
        _backward(
            xys, scales, rotation, colors, color_grads, softnesses, color_grad_scale, v_output, v_weights_sum, v_entropy, pixel_topk,
            aa_kernel_size, softness_default, sharpness, emission,
        )
    )  # fmt: skip


PORTED: frozenset[str] = frozenset(
    {
        "bin_and_sort_gaussians",
        "nd_rasterize_forward_topk_norm_hexels_color_grad",
        "nd_rasterize_forward_topk_norm_hexels_color_grad_softness",
        "nd_rasterize_backward_topk_norm_hexels_color_grad",
        "nd_rasterize_backward_topk_norm_hexels_color_grad_softness",
    }
)
"""The `gsplat.cuda` / `gsplat.utils` functions this module replaces."""


class _Overlay:
    def __init__(self, module: object) -> None:
        self._module = module

    def __getattr__(self, name: str) -> object:
        return globals()[name] if name in PORTED else getattr(self._module, name)


def overlay(module: object) -> Any:
    """`module` (hexel-kernel's `gsplat.cuda` or `gsplat` API) with this module's ports in front of its own functions."""
    return _Overlay(module)
