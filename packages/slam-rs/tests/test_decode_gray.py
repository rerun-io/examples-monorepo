"""The RGB copy a replay colours its landmarks from comes from the same decode, and leaves the gray frames unchanged."""

import numpy as np
from fixture_types import Mp4Factory
from jaxtyping import UInt8
from numpy import ndarray

from slam_rs.catalog_feed import color_copy_size, decode_gray


def test_the_colour_copy_leaves_the_gray_frames_alone_and_keeps_each_side(synthetic_mp4: Mp4Factory) -> None:
    """98 x 66 at a quarter is 24 x 16 (each side floored on its own); red left, blue right, so a swapped axis shows."""
    picture: UInt8[ndarray, "h w 3"] = np.zeros((66, 98, 3), dtype=np.uint8)
    picture[:, :48, 0] = 220
    picture[:, 50:, 2] = 220
    video: bytes = synthetic_mp4(picture, 3)
    gray_only = list(decode_gray(video))
    paired = list(decode_gray(video, color_downscale=4))
    assert all(copy is None for _, copy in gray_only)
    assert [gray.tobytes() for gray, _ in paired] == [gray.tobytes() for gray, _ in gray_only], "tracking sees the same pixels"
    assert color_copy_size(98, 66, 4) == (24, 16)
    for _, copy in paired:
        assert copy is not None
        rgb: UInt8[ndarray, "ch cw 3"] = copy()
        assert rgb.shape == (16, 24, 3) and rgb.dtype == np.uint8
        assert rgb[:, :10, 0].min() > 150 and rgb[:, :10, 2].max() < 60, "the left of the copy is the left of the frame"
        assert rgb[:, 14:, 2].min() > 150 and rgb[:, 14:, 0].max() < 60, "the right of the copy is the right of the frame"
