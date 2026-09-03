"""CUDA reconstruction for the constrained depth PNGs stored in the catalog."""

import torch
from jaxtyping import Int32, UInt8
from torch import Tensor


def unfilter_up_cuda(filtered_hwb: UInt8[Tensor, "h row_bytes"]) -> Int32[Tensor, "h w"]:
    """Reconstruct a 16-bit grayscale PNG's all-Up rows on CUDA.

    PNG Up filtering operates on bytes. An int32 cumulative sum down the rows,
    reduced modulo 256, reconstructs those bytes. Adjacent big-endian byte pairs
    are then combined into int32 millimetres because many PyTorch uint16
    operations are unsupported.

    Args:
        filtered_hwb: CUDA uint8 filtered rows with shape
            ``(height, 1 + 2 * width)`` and filter byte 2 in every row.

    Returns:
        CUDA int32 metric depth with shape ``(height, width)``.

    Raises:
        ValueError: If the input is not a two-dimensional CUDA uint8 matrix with
            a filter byte followed by complete byte pairs.
    """
    if filtered_hwb.ndim != 2 or filtered_hwb.dtype != torch.uint8 or not filtered_hwb.is_cuda:
        raise ValueError("filtered PNG rows must be a two-dimensional CUDA uint8 tensor")
    pixel_bytes: int = int(filtered_hwb.shape[1]) - 1
    if pixel_bytes <= 0 or pixel_bytes % 2 != 0:
        raise ValueError("filtered PNG rows must contain complete 16-bit pixels")
    reconstructed_hwb: Int32[Tensor, "h pixel_bytes"] = torch.bitwise_and(
        torch.cumsum(filtered_hwb[:, 1:], dim=0, dtype=torch.int32),
        0xFF,
    )
    high_hw: Int32[Tensor, "h w"] = reconstructed_hwb[:, 0::2]
    low_hw: Int32[Tensor, "h w"] = reconstructed_hwb[:, 1::2]
    return torch.bitwise_or(torch.bitwise_left_shift(high_hw, 8), low_hw)
