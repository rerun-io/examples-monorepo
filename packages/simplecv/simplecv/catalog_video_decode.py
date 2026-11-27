"""NVDEC decode of a reader's video streams, chunk by chunk, with the next batch decoded on one thread per stream.

``decode_runs`` plans which samples each decoder reads, so that no sample is decoded twice at any selection of pictures.
A torchcodec CUDA decoder decodes on any thread that has a current CUDA context (``video_io.bind_cuda_device``), not
only on the one that created it; here each stream's decoder is created and used on its own thread, which binds the device
once. Two NVDEC threads decode about twice the frames per second of one thread on the RTX 5090.
"""

import threading
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

from simplecv.catalog_video_codec import wrap_mp4
from simplecv.catalog_video_reader import MP4_FPS, CatalogVideoReader, SharedChunks, StreamPackets, _ChunkDecoder
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
    """Sequential decode of one stream of a reader, a chunk at a time, up to (not including) sample ``end``.

    One chunk is open at a time, muxed from the keyframe at or before the first frame asked of it and cut at ``end``, so
    memory is one chunk however far apart ``start`` and ``end`` are.
    """

    def __init__(
        self, reader: CatalogVideoReader, stream: int, start: int, end: int, device: str, *, packets: Iterator[StreamPackets] | None = None
    ) -> None:
        """Bind the device; nothing is fetched until frames are asked for.

        Args:
            reader: Where the stream's chunks come from.
            stream: Position of the stream in the reader.
            start: First frame that will be asked for.
            end: One past the last frame that will be asked for.
            device: ``cuda`` decodes on NVDEC.
            packets: The stream's packets of consecutive chunks, from one at or before ``start`` (``PrefetchedChunks``
                shares one read-ahead between its streams); by default the reader's cache.
        """
        bind_cuda_device(device)
        self.reader: CatalogVideoReader = reader
        self.stream: int = stream
        self.end: int = end
        self.device: str = device
        if packets is None:
            chunks: range = range(reader.chunk_of(stream, start), reader.chunk_of(stream, max(start, end - 1)) + 1)
            packets = (reader.packets(chunk).streams[stream] for chunk in chunks)
        self._packets: Iterator[StreamPackets] = packets
        self._open: tuple[StreamPackets, int, _ChunkDecoder] | None = None
        """The open chunk, the stream index of its decoder's frame 0, and the decoder."""

    def _decoder_for(self, index: int) -> tuple[StreamPackets, int, _ChunkDecoder]:
        from torchcodec.decoders import VideoDecoder

        chunk: StreamPackets | None = self._open[0] if self._open is not None else None
        while chunk is None or index >= chunk.first + len(chunk.samples):
            chunk = next(self._packets, None)
            if chunk is None:
                raise IndexError(f"{self.reader.streams[self.stream].entity}: sample {index} is past this decoder's chunks")
        local: int = index - chunk.first
        lo: int = max((position for position in range(local + 1) if chunk.keyframes[position]), default=0)
        hi: int = min(len(chunk.samples), self.end - chunk.first)
        data: bytes = wrap_mp4(chunk.samples[lo:hi], chunk.keyframes[lo:hi], MP4_FPS, codec=self.reader.streams[self.stream].codec)
        return chunk, chunk.first + lo, VideoDecoder(data, device=self.device, seek_mode="exact", num_ffmpeg_threads=0)

    def frames(self, indices: list[int]) -> UInt8[Tensor, "n h w 3"]:
        """RGB frames at ascending stream indices below ``end``."""
        parts: list[Tensor] = []
        done: int = 0
        while done < len(indices):
            if self._open is None or indices[done] >= self._open[0].first + len(self._open[0].samples):
                self._open = self._decoder_for(indices[done])
            chunk, offset, decoder = self._open
            group: list[int] = [index for index in indices[done:] if index < chunk.first + len(chunk.samples)]
            parts.append(decoder.get_frames_at([index - offset for index in group]).data)
            done += len(group)
        if not parts:
            return torch.empty((0, 0, 0, 3), dtype=torch.uint8, device=self.device)
        batch: UInt8[Tensor, "n 3 h w"] = parts[0] if len(parts) == 1 else torch.cat(parts)
        # NVDEC frames are NHWC in memory (NCHW is a view), so this is a view, not a copy
        return rearrange(batch, "n c h w -> n h w c").contiguous()


def _locked(lock: threading.Lock, packets: Iterator[StreamPackets]) -> Iterator[StreamPackets]:
    """``SharedChunks`` is not thread-safe; its streams' walks are taken on their decode threads."""
    while True:
        with lock:
            taken: StreamPackets | None = next(packets, None)
        if taken is None:
            return
        yield taken


class PrefetchedChunks:
    """Iterate the frames of each batch, one tensor per stream of the reader; the next batch decodes while the caller computes.

    The streams share one read-ahead of the reader's chunks: one query per chunk for every stream. ``wait_s`` is the time
    the caller spent waiting for decoded frames.
    """

    def __init__(self, reader: CatalogVideoReader, chunks: list[list[int]], start: int, end: int, device: str) -> None:
        """Start the decoders and the first batch (decoding starts at construction).

        The decoders read the chunks up to the last frame the batches want, or ``end`` when that is earlier: a caller
        that passes the end of the clip but samples only its start fetches only the start.

        Args:
            reader: One decode thread per stream.
            chunks: The batches of ascending sample indices, the same for every stream.
            start: First sample of the first batch, or earlier.
            end: One past the last sample a batch may want.
            device: ``cuda`` decodes on NVDEC.
        """
        self.chunks: list[list[int]] = chunks
        self.device: str = device
        self.wait_s: float = 0.0
        streams: range = range(len(reader.streams))
        self.threads: list[ThreadPoolExecutor] = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"decode-{slot}") for slot in streams]
        last: int = min(end, max((chunk[-1] for chunk in chunks if chunk), default=start) + 1)

        def chunk_at(slot: int, sample: int) -> int:
            return reader.chunk_of(slot, min(sample, len(reader.streams[slot].times) - 1))

        # Within a batch one stream may decode all of it before another starts, so it may hold the chunks a batch spans.
        span: int = max((chunk_at(slot, chunk[-1]) - chunk_at(slot, chunk[0]) + 1 for chunk in chunks if chunk for slot in streams), default=1)
        read: range = range(min(chunk_at(slot, start) for slot in streams), max(chunk_at(slot, max(start, last - 1)) for slot in streams) + 1)
        shared: SharedChunks = SharedChunks(reader, read, max_held=span + 2)
        lock: threading.Lock = threading.Lock()
        self.decoders: list[Future[StreamDecoder]] = [
            thread.submit(StreamDecoder, reader, slot, start, last, device, packets=_locked(lock, shared.stream(slot)))
            for slot, thread in zip(streams, self.threads, strict=True)
        ]
        self.pending: list[Future[Tensor]] = self._submit(chunks[0]) if chunks else []

    def _decode(self, slot: int, chunk: list[int]) -> UInt8[Tensor, "n h w 3"]:
        frames: UInt8[Tensor, "n h w 3"] = self.decoders[slot].result().frames(chunk)
        if torch.device(self.device).type == "cuda":
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
