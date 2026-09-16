"""NVDEC decoding through the shared SimpleCV reader, with grayscale batches for Rust."""

from collections.abc import Iterator

import torch
from einops import rearrange
from jaxtyping import Float32, UInt8
from numpy import ndarray
from simplecv.video_io import TorchCodecVideoReader
from torch import Tensor


def decode_gray_cuda(payload: bytes, resize_hw: tuple[int, int], downscale: int = 1) -> Iterator[UInt8[ndarray, "h w"]]:
    """Decode sequential batches, resize and convert RGB to gray on CUDA, then copy the small images to Rust's CPU input.

    Uses RGB luminance weights after NVDEC color conversion, so rounding and
    clipped chroma can differ from PyAV's direct YUV-to-gray path.
    The native dimensions after integer downscaling must match calibration;
    resizing must not make an incompatible source look valid.
    """
    reader: TorchCodecVideoReader = TorchCodecVideoReader(payload, device="cuda", seek_mode="exact", resize_hw=resize_hw)
    try:
        source_resize_hw: tuple[int, int] = (max(1, reader.source_height // downscale), max(1, reader.source_width // downscale))
        if source_resize_hw != resize_hw:
            raise ValueError(f"video resolution after downscale {downscale} is {source_resize_hw}, calibration says {resize_hw}")
        weights: Float32[Tensor, "1 3 1 1"] = rearrange(torch.tensor([0.299, 0.587, 0.114], device="cuda"), "c -> 1 c 1 1")
        for start in range(0, len(reader), 32):
            rgb: UInt8[Tensor, "b 3 h w"] = reader.get_frames_in_range(start, min(start + 32, len(reader)))
            gray: UInt8[Tensor, "b h w"] = (rgb.float() * weights).sum(dim=1).round().clamp(0, 255).to(dtype=torch.uint8)
            batch: UInt8[ndarray, "b h w"] = gray.cpu().numpy()
            yield from batch
    finally:
        reader.close()
