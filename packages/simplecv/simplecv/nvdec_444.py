"""NVDEC decode of 8-bit 4:4:4 video (HEVC/H.264 ``yuv444p``) through PyNvVideoCodec.

torchcodec's CUDA decoders hand NVDEC 4:2:0 streams only and decode 4:4:4 on the CPU (``cpu_fallback``),
an order of magnitude slower at 4K. NVDEC itself decodes 4:4:4: PyNvVideoCodec returns the planar YUV444
surface on the GPU, and the conversion to RGB happens here in one pass, with the matrix of the stream's
``color_space`` (limited range, as FFmpeg's swscale reads ``yuv444p``).

:class:`Nvdec444Decoder` offers the part of torchcodec's ``VideoDecoder`` surface that
:class:`simplecv.video_io.TorchCodecVideoReader` uses, so the reader swaps it in when torchcodec falls back.
Like every NVDEC decoder it must be created and used on one thread.
"""

import importlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float32, UInt8
from torch import Tensor

# (Kr, Kb) luma coefficients per FFmpeg color_space name; unknown or unset streams read as BT.601,
# FFmpeg's default for YUV->RGB.
_LUMA_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "bt470bg": (0.299, 0.114),
    "smpte170m": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020_ncl": (0.2627, 0.0593),
}


def limited_range_yuv_to_rgb_matrix(color_space: str | None) -> Float32[Tensor, "3 3"]:
    """Matrix taking limited-range ``[Y-16, U-128, V-128]`` to full-range RGB, for an FFmpeg ``color_space`` name."""
    kr, kb = _LUMA_COEFFICIENTS.get(color_space or "", (0.299, 0.114))
    kg: float = 1.0 - kr - kb
    y_scale: float = 255.0 / 219.0
    c_scale: float = 255.0 / 224.0
    return torch.tensor(
        [
            [y_scale, 0.0, 2.0 * (1.0 - kr) * c_scale],
            [y_scale, -2.0 * kb * (1.0 - kb) / kg * c_scale, -2.0 * kr * (1.0 - kr) / kg * c_scale],
            [y_scale, 2.0 * (1.0 - kb) * c_scale, 0.0],
        ],
        dtype=torch.float32,
    )


def yuv444_planes_to_rgb(planes: UInt8[Tensor, "3 h w"], matrix: Float32[Tensor, "3 3"]) -> UInt8[Tensor, "3 h w"]:
    """Convert one planar limited-range YUV444 frame to RGB (rounded, clamped to uint8)."""
    offsets: Float32[Tensor, "3 1 1"] = torch.tensor([16.0, 128.0, 128.0], device=planes.device).view(3, 1, 1)
    yuv: Float32[Tensor, "3 h w"] = planes.float() - offsets
    rgb: Float32[Tensor, "3 h w"] = torch.einsum("ij,jhw->ihw", matrix, yuv)
    return rgb.round_().clamp_(0.0, 255.0).to(torch.uint8)


@dataclass(frozen=True, slots=True)
class DecodedFrames:
    """``data`` holder, matching torchcodec's ``FrameBatch``/``Frame`` attribute the reader reads."""

    data: Tensor


class Nvdec444Decoder:
    """Frame-indexed NVDEC decoder of an 8-bit 4:4:4 file, yielding RGB ``NCHW`` uint8 on ``device``.

    Decodes sequentially through PyNvVideoCodec's demuxer and decoder. Not its ``SimpleDecoder``: that one
    corrupts the RASL pictures of open-GOP HEVC (the four frames before each CRA keyframe in x265's
    default output, as in the MAMMA iPhone videos) with block artefacts of up to 255 levels on moving
    regions. A request behind the read position re-opens the stream and decodes from the start; the
    streaming readers only move forward.
    """

    def __init__(self, source: Path, device: str, color_space: str | None) -> None:
        """Open ``source`` on the GPU of ``device`` (``cuda`` or ``cuda:N``); call on the decode thread.

        Raises:
            ValueError: The stream does not decode to 8-bit planar 4:4:4.
        """
        # PyNvVideoCodec binds its API from a compiled module at import time; type checkers see none of it.
        self._nvc: Any = importlib.import_module("PyNvVideoCodec")
        self._source: Path = source
        self.device: torch.device = torch.device(device)
        self._gpu_id: int = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self._matrix: Float32[Tensor, "3 3"] = limited_range_yuv_to_rgb_matrix(color_space).to(self.device)
        self._open()
        self.width: int = int(self._demuxer.Width())
        self.height: int = int(self._demuxer.Height())
        # The demuxer's ChromaFormat() reads 4:2:0 on 4:4:4 files; the first surface's shape is the check.
        self._frame(0)
        self.cpu_fallback: None = None
        """Never falls back; mirrors torchcodec's attribute."""

    def _open(self) -> None:
        nvc = self._nvc
        self._demuxer = nvc.CreateDemuxer(str(self._source))
        self._decoder = nvc.CreateDecoder(
            gpuid=self._gpu_id,
            codec=self._demuxer.GetNvCodecId(),
            # Decode on the thread's current torch stream, so the surfaces are ordered before the conversion.
            cudastream=torch.cuda.current_stream(self.device).cuda_stream,
            usedevicememory=True,
        )
        self._packets = iter(self._demuxer)
        self._ready: deque[UInt8[Tensor, "3 h w"]] = deque()
        self._next_index: int = 0
        """Display index of ``_ready[0]``."""

    def _fill(self) -> bool:
        """Decode packets until a frame is ready; ``False`` at the end of the stream."""
        while not self._ready:
            packet = next(self._packets, None)
            if packet is None:
                return False
            # Surfaces are recycled by the next Decode call: convert each frame now.
            for frame in self._decoder.Decode(packet):
                surface: Tensor = torch.from_dlpack(frame)
                if surface.shape != (3 * self.height, self.width) or surface.dtype != torch.uint8:
                    raise ValueError(f"NVDEC surface {tuple(surface.shape)} {surface.dtype} is not 8-bit planar 4:4:4")
                self._ready.append(yuv444_planes_to_rgb(surface.view(3, self.height, self.width), self._matrix))
        return True

    def _frame(self, index: int) -> UInt8[Tensor, "3 h w"]:
        if index < self._next_index:
            self._open()
        while True:
            if not self._fill():
                raise IndexError(f"{self._source}: frame {index} is past the end ({self._next_index} frames)")
            frame: UInt8[Tensor, "3 h w"] = self._ready.popleft()
            self._next_index += 1
            if self._next_index - 1 == index:
                return frame

    def get_frames_at(self, indices: list[int]) -> DecodedFrames:
        """RGB frames at ``indices`` as ``n 3 h w`` (ascending indices decode in one pass)."""
        frames: UInt8[Tensor, "n 3 h w"] = torch.empty((len(indices), 3, self.height, self.width), dtype=torch.uint8, device=self.device)
        for slot, index in enumerate(indices):
            frames[slot] = self._frame(index)
        return DecodedFrames(frames)

    def get_frames_in_range(self, start: int, stop: int) -> DecodedFrames:
        """RGB frames ``[start, stop)`` as ``n 3 h w``."""
        return self.get_frames_at(list(range(start, stop)))

    def get_frame_at(self, index: int) -> DecodedFrames:
        """One RGB frame as ``3 h w``."""
        return DecodedFrames(self.get_frames_at([index]).data[0])


__all__ = ("DecodedFrames", "Nvdec444Decoder", "limited_range_yuv_to_rgb_matrix", "yuv444_planes_to_rgb")
