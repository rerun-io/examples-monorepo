"""The NVDEC batch shrinks for large lens frames and keeps 32 frames up to 1920 px."""

import pytest

pytest.importorskip("torchcodec", reason="requires the torchcodec dependency in this environment")

from slam_rs.cuda_decode import decode_batch_frames


@pytest.mark.parametrize(("hw", "frames"), [((480, 640), 32), ((1920, 1920), 32), ((2880, 2880), 14), ((3840, 3840), 8), ((20000, 20000), 1)])
def test_frames_per_decode_batch(hw: tuple[int, int], frames: int) -> None:
    assert decode_batch_frames(hw) == frames
