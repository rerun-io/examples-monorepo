"""area_downscale_uint8: the integer-factor avg_pool2d path equals interpolate(mode="area") (CPU)."""

import torch

from simplecv.video_io import area_downscale_uint8


def test_area_downscale_matches_area_interpolation() -> None:
    """3x through avg_pool2d equals interpolate(mode="area"); a non-integer factor keeps interpolate."""
    gen = torch.Generator().manual_seed(1)
    frames: torch.Tensor = torch.randint(0, 256, (2, 3, 216, 384), generator=gen, dtype=torch.uint8)
    for size in ((72, 128), (100, 150)):
        expected = torch.nn.functional.interpolate(frames.float(), size=size, mode="area").round_().clamp_(0, 255).to(torch.uint8)
        assert torch.equal(area_downscale_uint8(frames, size), expected), size
