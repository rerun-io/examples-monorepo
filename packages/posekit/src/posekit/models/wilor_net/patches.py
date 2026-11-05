"""GPU-native image patch generation for WiLor crops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, UInt8
from torch import Tensor


def generate_rgb_image_patches_torch(
    frames_rgb: UInt8[Tensor, "batch h w 3"],
    *,
    frame_indices: Int[Tensor, "num_patches"],
    centers_xy: Float[Tensor, "num_patches 2"],
    box_sizes: Float[Tensor, "num_patches"],
    flips: Bool[Tensor, "num_patches"],
    patch_size: int = 256,
) -> Float[Tensor, "num_patches patch patch 3"]:
    """Generate WiLor RGB image patches with batched torch bilinear sampling.

    Args:
        frames_rgb: Decoded RGB frames in NHWC layout. CUDA tensors stay on CUDA.
        frame_indices: Source frame index for each requested crop.
        centers_xy: Crop centers in source image pixel coordinates.
        box_sizes: Square crop side length in source image pixels.
        flips: Whether each crop should mirror horizontally, matching the old
            OpenCV path for left-hand crops.
        patch_size: Output patch height and width.

    Returns:
        Float RGB patches in NHWC layout with values in the original 0..255 image
        range. The caller can round/cast if byte-exact integer patches are needed.

    Raises:
        ValueError: If ``patch_size`` is not positive or ``frames_rgb`` is not a
            four-dimensional RGB tensor.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if len(frames_rgb.shape) != 4 or int(frames_rgb.shape[-1]) != 3:
        raise ValueError(f"Expected RGB frames with shape [batch, h, w, 3], got {tuple(frames_rgb.shape)}.")
    if frame_indices.numel() == 0:
        return torch.empty((0, patch_size, patch_size, 3), dtype=torch.float32, device=frames_rgb.device)

    device: torch.device = frames_rgb.device
    frame_indices_device: Int[Tensor, "num_patches"] = frame_indices.to(device=device, dtype=torch.long)
    centers_device: Float[Tensor, "num_patches 2"] = centers_xy.to(device=device, dtype=torch.float32)
    box_sizes_device: Float[Tensor, "num_patches"] = box_sizes.to(device=device, dtype=torch.float32)
    flips_device: Bool[Tensor, "num_patches"] = flips.to(device=device, dtype=torch.bool)

    source_frames: Float[Tensor, "num_patches 3 h w"] = frames_rgb.index_select(0, frame_indices_device).permute(0, 3, 1, 2).float()
    image_height: int = int(frames_rgb.shape[1])
    image_width: int = int(frames_rgb.shape[2])

    coords: Float[Tensor, "patch"] = torch.arange(patch_size, dtype=torch.float32, device=device)
    grid_y: Float[Tensor, "patch patch"]
    grid_x: Float[Tensor, "patch patch"]
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    patch_center: float = float(patch_size) * 0.5
    source_scale: Float[Tensor, "num_patches 1 1"] = box_sizes_device[:, None, None] / float(patch_size)
    negative_ones: Float[Tensor, "num_patches"] = -torch.ones_like(box_sizes_device)
    positive_ones: Float[Tensor, "num_patches"] = torch.ones_like(box_sizes_device)
    flip_sign: Float[Tensor, "num_patches 1 1"] = torch.where(flips_device, negative_ones, positive_ones)[:, None, None]
    source_x: Float[Tensor, "num_patches patch patch"] = centers_device[:, 0, None, None] + flip_sign * (grid_x[None] - patch_center) * source_scale
    source_y: Float[Tensor, "num_patches patch patch"] = centers_device[:, 1, None, None] + (grid_y[None] - patch_center) * source_scale

    normalized_x: Float[Tensor, "num_patches patch patch"] = (2.0 * source_x / float(image_width - 1)) - 1.0
    normalized_y: Float[Tensor, "num_patches patch patch"] = (2.0 * source_y / float(image_height - 1)) - 1.0
    sample_grid: Float[Tensor, "num_patches patch patch 2"] = torch.stack((normalized_x, normalized_y), dim=-1)

    sampled: Float[Tensor, "num_patches 3 patch patch"] = F.grid_sample(
        source_frames,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.permute(0, 2, 3, 1).contiguous()
