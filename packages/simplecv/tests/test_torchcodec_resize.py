"""Tests for the ``resize_hw`` decode-resize support on TorchCodec readers."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import torch
from jaxtyping import UInt8

from simplecv.video_io import TorchCodecMultiVideoReader, TorchCodecVideoReader

_SOURCE_HW: tuple[int, int] = (240, 320)
_RESIZE_HW: tuple[int, int] = (120, 160)
_NUM_FRAMES: int = 8


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Encode a small synthetic H.264 clip with a per-frame brightness ramp."""
    path: Path = tmp_path_factory.mktemp("videos") / "synthetic.mp4"
    container: av.container.OutputContainer = av.open(str(path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("libx264", rate=30)
    stream.width = _SOURCE_HW[1]
    stream.height = _SOURCE_HW[0]
    stream.pix_fmt = "yuv420p"
    for i in range(_NUM_FRAMES):
        rgb: UInt8[np.ndarray, "h w 3"] = np.full((*_SOURCE_HW, 3), 20 + i * 25, dtype=np.uint8)
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def test_resize_hw_cpu_decode_time(synthetic_video: Path) -> None:
    """CPU path resizes during decoding and reports post-resize dimensions."""
    reader: TorchCodecVideoReader = TorchCodecVideoReader(synthetic_video, device="cpu", resize_hw=_RESIZE_HW)

    assert (reader.height, reader.width) == _RESIZE_HW
    assert (reader.source_height, reader.source_width) == _SOURCE_HW

    frame: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(2)
    assert tuple(frame.shape) == (3, *_RESIZE_HW)

    batch: UInt8[torch.Tensor, "b 3 h w"] = reader.get_frames_in_range(0, 4)
    assert tuple(batch.shape) == (4, 3, *_RESIZE_HW)


def test_resize_hw_preserves_content(synthetic_video: Path) -> None:
    """Resized frames keep the per-frame brightness ramp of the source."""
    reader: TorchCodecVideoReader = TorchCodecVideoReader(synthetic_video, device="cpu", resize_hw=_RESIZE_HW)
    first_mean: float = reader.get_frame(0).float().mean().item()
    last_mean: float = reader.get_frame(_NUM_FRAMES - 1).float().mean().item()
    assert last_mean > first_mean + 100.0


def test_resize_hw_none_is_native(synthetic_video: Path) -> None:
    """Default behavior (no resize) is unchanged."""
    reader: TorchCodecVideoReader = TorchCodecVideoReader(synthetic_video, device="cpu")
    assert (reader.height, reader.width) == _SOURCE_HW
    frame: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(0)
    assert tuple(frame.shape) == (3, *_SOURCE_HW)


def test_resize_hw_multiview(synthetic_video: Path) -> None:
    """Multiview reader forwards resize_hw and reports resized dims."""
    reader: TorchCodecMultiVideoReader = TorchCodecMultiVideoReader(
        [synthetic_video, synthetic_video], device="cpu", resize_hw=_RESIZE_HW
    )
    assert (reader.height, reader.width) == _RESIZE_HW
    chunk: list[UInt8[torch.Tensor, "b 3 h w"]] = next(reader.iter_chunks(chunk_size=4, max_frames=4))
    assert len(chunk) == 2
    assert tuple(chunk[0].shape) == (4, 3, *_RESIZE_HW)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU resize path")
def test_resize_hw_cuda_post_decode(synthetic_video: Path) -> None:
    """CUDA path decodes native then resizes on-GPU."""
    reader: TorchCodecVideoReader = TorchCodecVideoReader(synthetic_video, device="cuda", resize_hw=_RESIZE_HW)
    frame: UInt8[torch.Tensor, "3 h w"] = reader.get_frame(2)
    assert tuple(frame.shape) == (3, *_RESIZE_HW)
    assert frame.is_cuda
    batch: UInt8[torch.Tensor, "b 3 h w"] = reader.get_frames_in_range(0, 4)
    assert tuple(batch.shape) == (4, 3, *_RESIZE_HW)
    assert batch.is_cuda
