"""The YUV444 -> RGB conversion behind simplecv.nvdec_444 (CPU; the NVDEC path is tested in mamma's integration lane)."""

import torch

from simplecv.nvdec_444 import limited_range_yuv_to_rgb_matrix, yuv444_planes_to_rgb


def _convert(y: int, u: int, v: int, color_space: str | None) -> list[int]:
    planes: torch.Tensor = torch.tensor([y, u, v], dtype=torch.uint8).view(3, 1, 1)
    return yuv444_planes_to_rgb(planes, limited_range_yuv_to_rgb_matrix(color_space)).flatten().tolist()


def test_limited_range_black_and_white() -> None:
    assert _convert(16, 128, 128, "bt470bg") == [0, 0, 0]
    assert _convert(235, 128, 128, "bt709") == [255, 255, 255]


def test_bt601_and_bt709_primaries() -> None:
    """Studio-swing YUV of pure red per each standard (rounded to integers) comes back within 1 of (255, 0, 0)."""
    for yuv, color_space in (((81, 90, 240), "bt470bg"), ((63, 102, 240), "bt709")):
        rgb: list[int] = _convert(*yuv, color_space)
        assert max(abs(a - b) for a, b in zip(rgb, [255, 0, 0], strict=True)) <= 1, (color_space, rgb)


def test_unknown_color_space_reads_as_bt601() -> None:
    assert _convert(81, 90, 240, None) == _convert(81, 90, 240, "smpte170m")
