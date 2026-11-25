"""NVDEC decode of catalog video streams, with the next chunk decoded on one thread per stream.

``decode_runs`` plans which samples each decoder reads, so that no sample is decoded twice at any selection of pictures.
A torchcodec CUDA decoder decodes on any thread that has a current CUDA context (``video_io.bind_cuda_device``), not
only on the one that created it; here each stream's decoder is created and used on its own thread, which binds the device
once. Two NVDEC threads decode about twice the frames per second of one thread on the RTX 5090.
"""

import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Int64, UInt8
from numpy import ndarray
from torch import Tensor

from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.video_io import bind_cuda_device


@dataclass(frozen=True, slots=True)
class DecodeRun:
    """Samples ``start .. stop - 1`` of one stream, decoded in order by one decoder, and the pictures wanted among them."""

    start: int
    """First sample: a keyframe."""
    stop: int
    """One past the last wanted picture."""
    wanted: Int64[ndarray, " w"]
    """Wanted pictures (increasing), all in ``start .. stop - 1``."""


def decode_runs(indices: Int64[ndarray, " k"], gop_starts: Int64[ndarray, " g"], *, max_gops: int | None = None) -> list[DecodeRun]:
    """Split the pictures ``indices`` (increasing) of one stream into runs that a decoder reads front to back, so no
    sample is decoded twice and none is decoded past the last picture wanted.

    A run goes on while the next picture's GOP starts at or before the sample after the last one decoded: carrying on
    decodes exactly the samples a fresh decoder from that keyframe would. It ends at a GOP gap, or at a keyframe once it
    holds ``max_gops`` GOPs, which caps the samples muxed in memory (``max_gops=1``: one decoder per GOP).

    Args:
        indices: The wanted pictures, increasing.
        gop_starts: The keyframes (first sample of each GOP), increasing.
        max_gops: GOPs after which a run ends at the next keyframe; ``None`` for no cap.

    Returns:
        The runs, in order; together their ``wanted`` are ``indices``.

    Raises:
        ValueError: If ``indices`` is not increasing, or no keyframe precedes the first picture.
    """
    if indices.size and (np.any(np.diff(indices) <= 0) or indices[0] < 0):
        raise ValueError("the wanted pictures must be increasing and not negative")
    if indices.size == 0:
        return []
    if gop_starts.size == 0 or gop_starts[0] > indices[0]:
        raise ValueError(f"no keyframe at or before picture {int(indices[0])}")
    gop: Int64[ndarray, " k"] = np.searchsorted(gop_starts, indices, side="right") - 1
    keyframes: Int64[ndarray, " k"] = gop_starts[gop]
    runs: list[DecodeRun] = []
    first: int = 0
    for k in range(1, indices.size):
        # Carry on within a GOP, or into the next one when it starts right after the last sample decoded and the run has room.
        room: bool = max_gops is None or int(gop[k - 1] - gop[first]) + 1 < max_gops
        if gop[k] == gop[k - 1] or (keyframes[k] == indices[k - 1] + 1 and room):
            continue
        runs.append(DecodeRun(start=int(keyframes[first]), stop=int(indices[k - 1]) + 1, wanted=indices[first:k]))
        first = k
    runs.append(DecodeRun(start=int(keyframes[first]), stop=int(indices[-1]) + 1, wanted=indices[first:]))
    return runs


class StreamDecoder:
    """Sequential NVDEC decode of one stream, from the keyframe at or before ``start`` up to ``end``."""

    def __init__(self, video: CatalogVideo, start: int, end: int, device: str, fps: int = 30) -> None:
        """Mux the packets ``[keyframe at or before start, end)`` into an in-memory MP4 and open a decoder."""
        from torchcodec.decoders import VideoDecoder

        bind_cuda_device(device)
        self.first: int = max(i for i in range(start + 1) if video.keyframes[i])
        data: bytes = wrap_mp4(video.samples[self.first : end], video.keyframes[self.first : end], fps, codec=video.codec)
        self.decoder = VideoDecoder(data, device=device, seek_mode="exact", num_ffmpeg_threads=0)

    def frames(self, indices: list[int]) -> UInt8[Tensor, "n h w 3"]:
        """RGB frames at ascending stream indices."""
        batch: UInt8[Tensor, "n 3 h w"] = self.decoder.get_frames_at([i - self.first for i in indices]).data
        # NVDEC frames are NHWC in memory (NCHW is a view), so this is a view, not a copy
        return rearrange(batch, "n c h w -> n h w c").contiguous()


class PrefetchedChunks:
    """Iterate the frames of each chunk, one batch per stream; the next chunk decodes while the caller computes.

    ``wait_s`` is the time the caller spent waiting for decoded frames.
    """

    def __init__(self, videos: tuple[CatalogVideo, ...], chunks: list[list[int]], start: int, end: int, device: str, fps: int = 30) -> None:
        """Start the decoders and the first chunk (decoding starts at construction).

        Each decoder muxes its packets up to the last frame the chunks want, or ``end`` when that is earlier: a caller
        that passes the end of the clip but samples only its start does not hold the whole clip in memory.
        """
        self.chunks: list[list[int]] = chunks
        self.device: str = device
        self.wait_s: float = 0.0
        self.threads: list[ThreadPoolExecutor] = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"decode-{slot}") for slot in range(len(videos))]
        last: int = min(end, max((chunk[-1] for chunk in chunks if chunk), default=start) + 1)
        self.decoders: list[Future[StreamDecoder]] = [
            thread.submit(StreamDecoder, video, start, last, device, fps) for thread, video in zip(self.threads, videos, strict=True)
        ]
        self.pending: list[Future[Tensor]] = self._submit(chunks[0]) if chunks else []

    def _decode(self, slot: int, chunk: list[int]) -> UInt8[Tensor, "n h w 3"]:
        frames: UInt8[Tensor, "n h w 3"] = self.decoders[slot].result().frames(chunk)
        # This thread's stream only: a device-wide synchronize invalidates a CUDA graph another thread captures meanwhile.
        torch.cuda.current_stream(self.device).synchronize()
        return frames

    def _submit(self, chunk: list[int]) -> list[Future[Tensor]]:
        return [thread.submit(self._decode, slot, chunk) for slot, thread in enumerate(self.threads)]

    def __iter__(self) -> Iterator[list[UInt8[Tensor, "n h w 3"]]]:
        try:
            for position in range(len(self.chunks)):
                tick: float = time.perf_counter()
                batches: list[Tensor] = [future.result() for future in self.pending]
                self.wait_s += time.perf_counter() - tick
                self.pending = self._submit(self.chunks[position + 1]) if position + 1 < len(self.chunks) else []
                yield batches
        finally:
            for thread in self.threads:
                thread.shutdown(cancel_futures=True)


__all__ = ("PrefetchedChunks", "StreamDecoder")
