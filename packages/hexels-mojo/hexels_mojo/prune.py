"""Hexels' 2D prune scores (upstream `scripts/image_compression_prune_scores.cu`) on the Mojo kernel."""

import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from hexels_mojo._launch import launch

TOP_K: int = 10
"""Renderer-selected hexels per pixel (hexel-kernel `config.h` TOP_K)."""


def removal_scores_cuda(
    xy: Float32[Tensor, "n 2"],
    scales: Float32[Tensor, "n 2"],
    rotations: Float32[Tensor, "n 1"],
    colors: Float32[Tensor, "n 3"],
    gradients: Float32[Tensor, "n 2 3"],
    softness: Float32[Tensor, "n"],
    image: Float32[Tensor, "h w 4"],
    target: Float32[Tensor, "h w 3"],
    selected: Int32[Tensor, "h w 10"],
    rich: bool,
) -> list[Tensor]:
    """Per-hexel MSE change when that hexel alone is removed, and the largest |weight sum - rendered alpha|.

    `image` packs the rendered RGB and the alpha denominator; `selected` holds the renderer's top-10 ids per
    pixel (-1 = empty). Returns `[scores (n,), max_error ()]` like upstream's `removal_scores_cuda`.
    """
    scores: Tensor = torch.zeros(xy.shape[0], dtype=torch.float32, device=xy.device)
    max_error: Tensor = torch.zeros((), dtype=torch.float32, device=xy.device)
    # The kernel takes the running maximum as unsigned bits (non-negative floats order like their bits).
    launch(
        "removal_scores", (rich,), (xy, scales, rotations, colors, gradients, softness, image, target, selected, scores, max_error.view(torch.int32))
    )
    return [scores, max_error]
