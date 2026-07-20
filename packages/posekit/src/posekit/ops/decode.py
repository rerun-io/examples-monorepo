"""GPU keypoint decoders: UDP heatmaps (DARK-refined) and SimCC.

Both decoders return keypoints in *crop input* coordinates plus per-keypoint
scores, all as torch tensors on the inference device. Use
:func:`posekit.ops.crops.crop_coords_to_image` to map them back to the source
image.
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Float32
from numpy import ndarray
from torch import Tensor

from posekit.ops.crops import _float32_vector


def decode_udp_heatmaps(
    heatmaps: Float[Tensor, "n k hm_h hm_w"],
    *,
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
    blur_kernel_size: int = 11,
) -> tuple[Float[Tensor, "n k 2"], Float[Tensor, "n k"]]:
    """Decode UDP heatmaps with DARK log-space Hessian refinement.

    Args:
        heatmaps: Predicted heatmaps (any float dtype; upcast to float32).
        input_size: Crop input size as ``(width, height)``.
        heatmap_size: Heatmap size as ``(width, height)``.
        blur_kernel_size: Gaussian modulation kernel used before refinement.

    Returns:
        Keypoints in crop input coordinates and per-keypoint max scores.
    """
    heatmaps = heatmaps.float()
    num_instances: int = int(heatmaps.shape[0])
    num_keypoints: int = int(heatmaps.shape[1])
    heatmap_height: int = int(heatmaps.shape[2])
    heatmap_width: int = int(heatmaps.shape[3])
    scores: Float[Tensor, "n k"]
    flat_indices: Tensor
    scores, flat_indices = heatmaps.reshape(num_instances, num_keypoints, -1).max(dim=-1)
    keypoints: Float[Tensor, "n k 2"] = torch.stack(
        [(flat_indices % heatmap_width).float(), torch.div(flat_indices, heatmap_width, rounding_mode="floor").float()], dim=-1
    )
    keypoints = torch.where((scores > 0.0)[:, :, None], keypoints, torch.full_like(keypoints, -1.0))
    original_max: Tensor = heatmaps.reshape(num_instances * num_keypoints, -1).amax(dim=1).clamp_min(1e-12)
    blurred: Float[Tensor, "n k hm_h hm_w"] = _gaussian_blur_heatmaps(heatmaps, blur_kernel_size)
    blurred = blurred * (original_max / blurred.reshape(num_instances * num_keypoints, -1).amax(dim=1).clamp_min(1e-12)).reshape(
        num_instances, num_keypoints, 1, 1
    )
    padded: Tensor = F.pad(
        blurred.clamp(1e-3, 50.0).log().reshape(num_instances * num_keypoints, 1, heatmap_height, heatmap_width), (1, 1, 1, 1), mode="replicate"
    ).reshape(num_instances, num_keypoints, heatmap_height + 2, heatmap_width + 2)
    x_idx: Tensor = keypoints[..., 0].round().long().clamp(0, heatmap_width - 1) + 1
    y_idx: Tensor = keypoints[..., 1].round().long().clamp(0, heatmap_height - 1) + 1
    instance_idx: Tensor = _long_arange(num_instances, heatmaps.device)[:, None]
    keypoint_idx: Tensor = _long_arange(num_keypoints, heatmaps.device)[None, :]
    center: Tensor = padded[instance_idx, keypoint_idx, y_idx, x_idx]
    ix1: Tensor = padded[instance_idx, keypoint_idx, y_idx, x_idx + 1]
    iy1: Tensor = padded[instance_idx, keypoint_idx, y_idx + 1, x_idx]
    ix1y1: Tensor = padded[instance_idx, keypoint_idx, y_idx + 1, x_idx + 1]
    ix1_y1_: Tensor = padded[instance_idx, keypoint_idx, y_idx - 1, x_idx - 1]
    ix1_: Tensor = padded[instance_idx, keypoint_idx, y_idx, x_idx - 1]
    iy1_: Tensor = padded[instance_idx, keypoint_idx, y_idx - 1, x_idx]
    derivative: Tensor = torch.stack([0.5 * (ix1 - ix1_), 0.5 * (iy1 - iy1_)], dim=-1).unsqueeze(-1)
    dxy: Tensor = 0.5 * (ix1y1 - ix1 - iy1 + center + center - ix1_ - iy1_ + ix1_y1_)
    hessian: Tensor = torch.stack([ix1 - 2.0 * center + ix1_, dxy, dxy, iy1 - 2.0 * center + iy1_], dim=-1).reshape(
        num_instances, num_keypoints, 2, 2
    )
    offsets: Tensor = torch.linalg.solve(hessian + torch.finfo(torch.float32).eps * _eye2(heatmaps.device), derivative).squeeze(-1)
    refined: Float[Tensor, "n k 2"] = torch.where(torch.isfinite(offsets).all(dim=-1, keepdim=True), keypoints - offsets, keypoints)
    rescale: Float[Tensor, "2"] = _float32_vector(
        (float(input_size[0]) / float(heatmap_size[0] - 1), float(input_size[1]) / float(heatmap_size[1] - 1)), heatmaps.device
    )
    return refined * rescale, scores


def decode_classic_heatmaps(
    heatmaps: Float[Tensor, "n k hm_h hm_w"],
    *,
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> tuple[Float[Tensor, "n k 2"], Float[Tensor, "n k"]]:
    """Decode classic top-down heatmaps (argmax + quarter-pixel gradient shift).

    The mmpose ``post_process='default'`` decoder used by ViTPose-family
    checkpoints: no UDP alignment, no DARK refinement — just a 0.25 px shift
    toward the higher neighbor, then a plain ``input/heatmap`` rescale.

    Args:
        heatmaps: Predicted heatmaps (any float dtype; upcast to float32).
        input_size: Crop input size as ``(width, height)``.
        heatmap_size: Heatmap size as ``(width, height)``.

    Returns:
        Keypoints in crop input coordinates (invalid keypoints set to ``-1``)
        and per-keypoint max scores.
    """
    heatmaps = heatmaps.float()
    num_instances: int = int(heatmaps.shape[0])
    num_keypoints: int = int(heatmaps.shape[1])
    heatmap_height: int = int(heatmaps.shape[2])
    heatmap_width: int = int(heatmaps.shape[3])
    scores: Float[Tensor, "n k"]
    flat_indices: Tensor
    scores, flat_indices = heatmaps.reshape(num_instances, num_keypoints, -1).max(dim=-1)
    x_locs: Tensor = (flat_indices % heatmap_width).long()
    y_locs: Tensor = torch.div(flat_indices, heatmap_width, rounding_mode="floor").long()
    instance_idx: Tensor = _long_arange(num_instances, heatmaps.device)[:, None]
    keypoint_idx: Tensor = _long_arange(num_keypoints, heatmaps.device)[None, :]
    x_plus: Tensor = heatmaps[instance_idx, keypoint_idx, y_locs, (x_locs + 1).clamp(max=heatmap_width - 1)]
    x_minus: Tensor = heatmaps[instance_idx, keypoint_idx, y_locs, (x_locs - 1).clamp(min=0)]
    y_plus: Tensor = heatmaps[instance_idx, keypoint_idx, (y_locs + 1).clamp(max=heatmap_height - 1), x_locs]
    y_minus: Tensor = heatmaps[instance_idx, keypoint_idx, (y_locs - 1).clamp(min=0), x_locs]
    interior_x: Tensor = (x_locs > 0) & (x_locs < heatmap_width - 1)
    interior_y: Tensor = (y_locs > 0) & (y_locs < heatmap_height - 1)
    shift_x: Tensor = torch.where(interior_x, 0.25 * torch.sign(x_plus - x_minus), torch.zeros_like(scores))
    shift_y: Tensor = torch.where(interior_y, 0.25 * torch.sign(y_plus - y_minus), torch.zeros_like(scores))
    keypoints: Float[Tensor, "n k 2"] = torch.stack([x_locs.float() + shift_x, y_locs.float() + shift_y], dim=-1)
    keypoints = torch.where((scores > 0.0)[:, :, None], keypoints, torch.full_like(keypoints, -1.0))
    rescale: Float[Tensor, "2"] = _float32_vector(
        (float(input_size[0]) / float(heatmap_size[0]), float(input_size[1]) / float(heatmap_size[1])), heatmaps.device
    )
    return keypoints * rescale, scores


def decode_simcc(
    simcc_x: Float[Tensor, "n k bins_x"],
    simcc_y: Float[Tensor, "n k bins_y"],
    *,
    simcc_split_ratio: float = 2.0,
) -> tuple[Float[Tensor, "n k 2"], Float[Tensor, "n k"]]:
    """Decode SimCC classification logits into crop-input keypoints.

    Args:
        simcc_x: Per-keypoint x-axis logits.
        simcc_y: Per-keypoint y-axis logits.
        simcc_split_ratio: Bin-per-pixel ratio the model was trained with.

    Returns:
        Keypoints in crop input coordinates (invalid keypoints set to ``-1``)
        and per-keypoint scores.
    """
    x_scores: Tensor
    x_locs: Tensor
    y_scores: Tensor
    y_locs: Tensor
    x_scores, x_locs = simcc_x.float().max(dim=-1)
    y_scores, y_locs = simcc_y.float().max(dim=-1)
    scores: Float[Tensor, "n k"] = 0.5 * (x_scores + y_scores)
    locs: Float[Tensor, "n k 2"] = torch.where(
        (scores > 0.0)[:, :, None],
        torch.stack([x_locs.float(), y_locs.float()], dim=-1),
        torch.full((*scores.shape, 2), -1.0, dtype=torch.float32, device=simcc_x.device),
    )
    return locs / float(simcc_split_ratio), scores


def _gaussian_blur_heatmaps(heatmaps: Float[Tensor, "n k hm_h hm_w"], kernel_size: int) -> Float[Tensor, "n k hm_h hm_w"]:
    # Separable zero-border blur; conv2d's own zero padding replaces an explicit
    # F.pad + border slice (identical values, one less full-size allocation).
    border: int = (kernel_size - 1) // 2
    n, k, h, w = int(heatmaps.shape[0]), int(heatmaps.shape[1]), int(heatmaps.shape[2]), int(heatmaps.shape[3])
    kernel: Float[Tensor, "kernel"] = _gaussian_kernel1d(kernel_size, device=heatmaps.device)
    blurred_rows: Tensor = F.conv2d(heatmaps.reshape(n * k, 1, h, w), kernel.reshape(1, 1, 1, kernel_size), padding=(0, border))
    return F.conv2d(blurred_rows, kernel.reshape(1, 1, kernel_size, 1), padding=(border, 0)).reshape(n, k, h, w)


@lru_cache(maxsize=8)
def _gaussian_kernel1d_np(kernel_size: int) -> Float32[ndarray, "kernel"]:
    import cv2

    return cv2.getGaussianKernel(kernel_size, 0).astype(np.float32).reshape(kernel_size)


@lru_cache(maxsize=16)
def _gaussian_kernel1d(kernel_size: int, *, device: torch.device) -> Float[Tensor, "kernel"]:
    return torch.as_tensor(_gaussian_kernel1d_np(kernel_size), dtype=torch.float32, device=device)


@lru_cache(maxsize=16)
def _long_arange(length: int, device: torch.device) -> Tensor:
    return torch.arange(length, dtype=torch.long, device=device)


@lru_cache(maxsize=8)
def _eye2(device: torch.device) -> Float[Tensor, "1 1 2 2"]:
    return torch.eye(2, dtype=torch.float32, device=device).reshape(1, 1, 2, 2)
