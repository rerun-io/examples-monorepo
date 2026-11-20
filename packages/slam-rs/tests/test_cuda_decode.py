"""NVDEC must not hide a source/calibration resolution mismatch by resizing it."""

from io import BytesIO

import av
import numpy as np
import pytest
import torch
from av.container import OutputContainer
from av.video.stream import VideoStream
from jaxtyping import UInt8
from numpy import ndarray

pytest.importorskip("torchcodec", reason="requires the torchcodec dependency in this environment")

from slam_rs.cuda_decode import decode_gray_cuda

pytestmark = pytest.mark.integration


@pytest.fixture
def video() -> bytes:
    if not torch.cuda.is_available():
        pytest.skip("CUDA decoder test requires an NVIDIA GPU")
    pytest.importorskip("torchcodec")
    buffer: BytesIO = BytesIO()
    container: OutputContainer
    with av.open(buffer, mode="w", format="mp4") as container:
        stream: VideoStream = container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), 100, dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def test_cuda_rejects_source_resolution_mismatch(video: bytes) -> None:
    with pytest.raises(ValueError, match="resolution"):
        list(decode_gray_cuda(video, (32, 48)))


@pytest.mark.parametrize(("downscale", "height", "width"), [(1, 64, 96), (3, 21, 32)])
def test_cuda_preserves_calibrated_integer_downscale(video: bytes, downscale: int, height: int, width: int) -> None:
    frames: list[UInt8[ndarray, "h w"]] = [gray for gray, color in decode_gray_cuda(video, (height, width), downscale) if color is None]
    assert len(frames) == 1
    for frame in frames:
        assert frame.shape == (height, width)
        assert frame.dtype == np.uint8
        assert frame.flags.c_contiguous
        np.testing.assert_allclose(frame, 100, atol=2)


def test_cuda_pairs_each_gray_frame_with_a_smaller_rgb_copy_on_request(video: bytes) -> None:
    pairs = list(decode_gray_cuda(video, (64, 96), 1, color_downscale=4))
    assert len(pairs) == 1
    gray, color = pairs[0]
    assert gray.shape == (64, 96) and color is not None and color.shape == (16, 24, 3) and color.dtype == np.uint8
