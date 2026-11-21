"""NVDEC decode of 8-bit 4:4:4 video (HEVC/H.264 ``yuv444p``) through PyNvVideoCodec.

torchcodec's CUDA decoders hand NVDEC 4:2:0 streams only and decode 4:4:4 on the CPU (``cpu_fallback``),
an order of magnitude slower at 4K. NVDEC itself decodes 4:4:4: PyNvVideoCodec returns the planar YUV444
surface on the GPU, and the conversion to RGB happens here in one pass, with the matrix of the stream's
``color_space`` (limited range, as FFmpeg's swscale reads ``yuv444p``).

:class:`Nvdec444Decoder` offers the part of torchcodec's ``VideoDecoder`` surface that
:class:`simplecv.video_io.TorchCodecVideoReader` uses, so the reader swaps it in when torchcodec falls back.
NVDEC is thread-affine, so the decoder owns one thread and runs every native call (open, advance, re-open) there,
whichever thread asks for frames.
"""

import importlib
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from jaxtyping import Float32, UInt8
from torch import Tensor

T = TypeVar("T")

_BT601: tuple[float, float] = (0.299, 0.114)
# (Kr, Kb) luma coefficients per FFmpeg color_space name the conversion handles. Unset or unknown metadata reads as
# BT.601, FFmpeg's default for YUV->RGB; any other signalled matrix (smpte240m, ycgco, bt2020c, ...) is not converted
# here, so the reader leaves such streams to torchcodec's CPU decode.
_LUMA_COEFFICIENTS: dict[str | None, tuple[float, float]] = {
    None: _BT601,
    "": _BT601,
    "unknown": _BT601,
    "unspecified": _BT601,
    "bt470bg": _BT601,
    "smpte170m": _BT601,
    "bt709": (0.2126, 0.0722),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020_ncl": (0.2627, 0.0593),
}


def converts_color_space(color_space: str | None) -> bool:
    """Whether :func:`limited_range_yuv_to_rgb_matrix` reproduces FFmpeg's conversion for this ``color_space``."""
    return color_space in _LUMA_COEFFICIENTS


def limited_range_yuv_to_rgb_matrix(color_space: str | None) -> Float32[Tensor, "3 3"]:
    """Matrix taking limited-range ``[Y-16, U-128, V-128]`` to full-range RGB, for an FFmpeg ``color_space`` name.

    Raises:
        ValueError: ``color_space`` is a signalled matrix this conversion does not handle (see :func:`converts_color_space`).
    """
    if not converts_color_space(color_space):
        raise ValueError(f"no YUV->RGB matrix for color_space {color_space!r}")
    kr, kb = _LUMA_COEFFICIENTS[color_space]
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
    # Scalar offsets in place: a tensor built from a Python list would be a blocking host->device copy per frame.
    yuv: Float32[Tensor, "3 h w"] = planes.float()
    yuv[0] -= 16.0
    yuv[1:] -= 128.0
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
        """Open ``source`` on the GPU of ``device`` (``cuda`` or ``cuda:N``), on the decoder's own thread.

        Raises:
            ValueError: The stream does not decode to 8-bit planar 4:4:4, or ``color_space`` is not converted here.
        """
        # Resolve a bare "cuda" on the caller's thread: the decoder thread starts on device 0 and set_device needs an index.
        requested: torch.device = torch.device(device)
        self.device: torch.device = torch.device("cuda", torch.cuda.current_device()) if requested.type == "cuda" and requested.index is None else requested
        self._matrix: Float32[Tensor, "3 3"] = limited_range_yuv_to_rgb_matrix(color_space).to(self.device)
        self._source: Path = source
        self._thread: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvdec-444")
        self._thread_ident: int = self._thread.submit(threading.get_ident).result()
        try:
            self._run(self._start)
        except BaseException:
            self.close()
            raise

    def _run(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` on the decoder's thread (directly when already there)."""
        if threading.get_ident() == self._thread_ident:
            return fn()
        assert self._thread is not None, "decoder is closed"
        return self._thread.submit(fn).result()

    def _start(self) -> None:
        torch.cuda.set_device(self.device)  # NVDEC needs the CUDA context current on the decode thread
        # PyNvVideoCodec binds its API from a compiled module at import time; type checkers see none of it.
        self._nvc: Any = importlib.import_module("PyNvVideoCodec")
        self._gpu_id: int = torch.cuda.current_device()
        self._open()
        self.width: int = int(self._demuxer.Width())
        self.height: int = int(self._demuxer.Height())
        # The demuxer's ChromaFormat() reads 4:2:0 on 4:4:4 files; the first surface's shape is the check.
        # Frame 0 stays queued for the first read.
        if not self._fill(0):
            raise ValueError(f"{self._source}: no frames")

    def close(self) -> None:
        """Release the native decoder on its thread and stop the thread. Safe to call more than once."""
        if self._thread is None:
            return
        self._thread.submit(self._release).result()
        self._thread.shutdown(wait=True)
        self._thread = None

    def _release(self) -> None:
        # Drop the native decoder and demuxer here, on their thread.
        self._packets = iter(())
        self.__dict__.pop("_decoder", None)
        self.__dict__.pop("_demuxer", None)

    def _open(self) -> None:
        nvc = self._nvc
        self._demuxer: Any = nvc.CreateDemuxer(str(self._source))
        self._decoder: Any = nvc.CreateDecoder(
            gpuid=self._gpu_id,
            codec=self._demuxer.GetNvCodecId(),
            # Decode on the thread's current torch stream, so the surfaces are ordered before the conversion.
            cudastream=torch.cuda.current_stream(self.device).cuda_stream,
            usedevicememory=True,
        )
        self._packets = iter(self._demuxer)
        self._ready: deque[tuple[int, UInt8[Tensor, "3 h w"]]] = deque()
        """Converted frames with their display index."""
        self._decoded: int = 0
        """Display index of the next frame the decoder outputs."""
        self._next_index: int = 0
        """Display index of the next frame a read can return without re-opening."""

    def _fill(self, first_wanted: int) -> bool:
        """Decode packets until a frame at or after ``first_wanted`` is ready; ``False`` at the end of the stream.

        Earlier frames are dropped unconverted. Surfaces are recycled by the next Decode call, so the kept
        frames are converted now.
        """
        while not self._ready:
            packet = next(self._packets, None)
            if packet is None:
                return False
            for frame in self._decoder.Decode(packet):
                index: int = self._decoded
                self._decoded += 1
                surface: Tensor = torch.from_dlpack(frame)
                if surface.shape != (3 * self.height, self.width) or surface.dtype != torch.uint8:
                    raise ValueError(f"NVDEC surface {tuple(surface.shape)} {surface.dtype} is not 8-bit planar 4:4:4")
                if index >= first_wanted:
                    self._ready.append((index, yuv444_planes_to_rgb(surface.view(3, self.height, self.width), self._matrix)))
        return True

    def _frame(self, index: int) -> UInt8[Tensor, "3 h w"]:
        if index < self._next_index:
            self._open()
        while True:
            if not self._ready and not self._fill(index):
                raise IndexError(f"{self._source}: frame {index} is past the end ({self._decoded} frames)")
            ready_index, frame = self._ready.popleft()
            self._next_index = ready_index + 1
            if ready_index == index:
                return frame

    def get_frames_at(self, indices: list[int]) -> DecodedFrames:
        """RGB frames at ``indices`` as ``n 3 h w`` (ascending indices decode in one pass); any calling thread."""

        def decode() -> UInt8[Tensor, "n 3 h w"]:
            frames: UInt8[Tensor, "n 3 h w"] = torch.empty((len(indices), 3, self.height, self.width), dtype=torch.uint8, device=self.device)
            for slot, index in enumerate(indices):
                frames[slot] = self._frame(index)
            # The caller's stream may differ from this thread's: settle the frames before handing them over.
            torch.cuda.current_stream(self.device).synchronize()
            return frames

        return DecodedFrames(self._run(decode))

    def get_frames_in_range(self, start: int, stop: int) -> DecodedFrames:
        """RGB frames ``[start, stop)`` as ``n 3 h w``."""
        return self.get_frames_at(list(range(start, stop)))

    def get_frame_at(self, index: int) -> DecodedFrames:
        """One RGB frame as ``3 h w``."""
        return DecodedFrames(self.get_frames_at([index]).data[0])


__all__ = ("DecodedFrames", "Nvdec444Decoder", "converts_color_space", "limited_range_yuv_to_rgb_matrix", "yuv444_planes_to_rgb")
