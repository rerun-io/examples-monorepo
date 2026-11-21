"""NVDEC decode of catalog video streams, with the next chunk decoded on one thread per stream.

A torchcodec CUDA decoder fails on a thread other than the one that created it, so each stream's decoder is created and
used on its own thread. Two NVDEC threads decode about twice the frames per second of one thread on the RTX 5090.
"""

import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from einops import rearrange
from jaxtyping import UInt8
from torch import Tensor

from simplecv.catalog_video import CatalogVideo
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.video_io import bind_cuda_device


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
        """Start the decoders and the first chunk (decoding starts at construction)."""
        self.chunks: list[list[int]] = chunks
        self.device: str = device
        self.wait_s: float = 0.0
        self.threads: list[ThreadPoolExecutor] = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"decode-{slot}") for slot in range(len(videos))]
        self.decoders: list[Future[StreamDecoder]] = [
            thread.submit(StreamDecoder, video, start, end, device, fps) for thread, video in zip(self.threads, videos, strict=True)
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
