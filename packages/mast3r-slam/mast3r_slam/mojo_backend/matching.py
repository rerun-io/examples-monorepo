"""The matching ops of MASt3R-SLAM's frontend on the Mojo kernels (`mojo/matching.mojo`), with the signatures of the
former CUDA extension's `iter_proj` and `refine_matches`."""

import torch
from jaxtyping import Bool, Float16, Float32, Int64
from torch import Tensor

from mast3r_slam.mojo_backend._launch import launch


def iter_proj(
    rays: Float32[Tensor, "b h w 9"],
    pts: Float32[Tensor, "b n 3"],
    p_init: Float32[Tensor, "b n 2"],
    max_iter: int,
    lambda_init: float,
    cost_thresh: float,
) -> tuple[Float32[Tensor, "b n 2"], Bool[Tensor, "b n"]]:
    """For each point of pts (unit directions), the pixel (u, v) whose ray in `rays` matches it, by up to max_iter
    Levenberg-Marquardt steps from p_init; and whether its final cost is below cost_thresh."""
    rays, pts, p_init = rays.contiguous(), pts.contiguous(), p_init.contiguous()
    p_new: Tensor = torch.empty_like(p_init)
    converged: Tensor = torch.zeros(p_init.shape[:2], dtype=torch.bool, device=p_init.device)
    launch("iter_proj", (max_iter, lambda_init, cost_thresh), (rays, pts, p_init, p_new, converged))
    return p_new, converged


def refine_matches(
    D11: Float16[Tensor, "b h w f"], D21: Float16[Tensor, "b n f"], p1: Int64[Tensor, "b n 2"], radius: int, dilation_max: int
) -> tuple[Int64[Tensor, "b n 2"]]:
    """For each query descriptor of D21, the pixel of D11 within `radius` of p1 (coarse to fine over dilations
    dilation_max .. 1) whose descriptor has the largest dot product with it."""
    D11, D21, p1 = D11.contiguous(), D21.contiguous(), p1.contiguous()
    p_new: Tensor = torch.empty_like(p1)
    aligned: bool = D11.data_ptr() % 16 == 0 and D21.data_ptr() % 16 == 0
    launch("refine_matches", (radius, dilation_max, aligned), (D11, D21, p1, p_new))
    return (p_new,)
