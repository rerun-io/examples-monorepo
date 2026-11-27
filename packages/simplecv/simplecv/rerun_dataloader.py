"""The Rerun catalog dataloader's NVDEC column decoder, over the chunked reader (:mod:`simplecv.catalog_video_reader`), and keyed
decoders (NVDEC and the dataloader's CPU decoder) whose values name the segment and index they answer.

Import this module explicitly: it needs ``rerun-sdk``'s ``dataloader`` extra, which only the catalog lanes install, so
it must never be re-exported from ``simplecv/__init__.py``.

Upstream's video decoder (rerun-io/reality PR #2893) wraps each sample's keyframe window into its own in-memory MP4
and is slower than dav1d, with no decode-session reuse across calls. ``SegmentNvdecDecoder`` instead decodes from
GOP-aligned chunks of the segment on one GPU decoder per chunk.

TODO(rerun#upstream): delete this module once the dataloader decodes video fast enough on its own. Holding a
``DatasetEntry`` makes it unpicklable (so ``num_workers`` must stay 0), and the cached CUDA decoder does not survive
``decode_threads > 1``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Generic, Literal, TypeAlias, TypeVar

import numpy as np
import torch
from jaxtyping import Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import ColumnDecoder, DecodeRequest, FieldBatch, VideoFrameDecoder, Yuv420Frame
from torch import Tensor

from simplecv.catalog_video_reader import ChunkedVideoDecoder, open_decoders

TimedeltaNs: TypeAlias = Shaped[ndarray, " n_samples"]
"""Catalog index values in timeline order, preserving the timeline dtype."""
FrameRgbChw: TypeAlias = UInt8[Tensor, "3 h w"]
"""One decoded frame, channels-first RGB on the decoder's device."""
RECOMMENDED_FETCH_BLOCK_SIZE: int = 1024
"""Recommended samples per query when a decoder ignores the shipped payloads."""
DecodedT = TypeVar("DecodedT")


class SegmentNvdecDecoder(ColumnDecoder[Tensor]):
    """Serve a fetched request block from the segment's chunked decoder (:mod:`simplecv.catalog_video_reader`).

    The base-class defaults (no ``prior_keyframe_path``) make the dataloader ship each grid slot as a point read; the
    shipped bytes are ignored, the frames come from GOP-aligned chunks of the segment instead, so memory is a few
    chunks whatever the segment's length.
    """

    def __init__(
        self,
        dataset: DatasetEntry,
        entity: str,
        timeline: str,
        device: torch.device,
        *,
        max_frames_per_call: int | None = None,
        transform: Callable[[UInt8[Tensor, "n 3 h w"]], Tensor] | None = None,
    ) -> None:
        """Bind the decoder to one dataset's video column.

        Args:
            dataset: Rerun catalog dataset entry the segments come from.
            entity: Entity path of the ``VideoStream`` column.
            timeline: A temporal or a sequence timeline.
            device: ``cuda`` decodes on NVDEC; take it from :func:`simplecv.catalog_video_reader.select_decode_device`.
            max_frames_per_call: Decode a fetched block at most this many frames at a time (None: one call per segment run).
                A 1,024-frame block of 1080p RGB is 6.4 GB on the device; slices bound that peak.
            transform: Applied on the device to each decoded slice (``n 3 h w`` uint8) before results are kept, e.g. to reduce
                frames to what the consumer needs; its output's first dimension must stay ``n``.
        """
        if max_frames_per_call is not None and max_frames_per_call < 1:
            raise ValueError("max_frames_per_call must be positive")
        self._max_frames: int | None = max_frames_per_call
        self._transform: Callable[[UInt8[Tensor, "n 3 h w"]], Tensor] | None = transform
        self._dataset = dataset
        self._entity = entity
        self._timeline = timeline
        self._device = device
        self._segment_id: str | None = None
        self.decoder: ChunkedVideoDecoder | None = None
        """The current segment's decoder; its ``reader`` relays the stored samples (``relay_video``)."""
        self.query_seconds: float = 0.0
        """Index query and reader setup, per segment."""
        self.decode_seconds: float = 0.0
        """Chunk fetches and GPU decode."""
        self.query_count: int = 0
        self.times: TimedeltaNs = np.empty(0, dtype="timedelta64[ns]")
        """The current segment's sample index values, in the timeline's dtype."""

    def _ensure_segment(self, segment_id: str) -> ChunkedVideoDecoder:
        if segment_id != self._segment_id or self.decoder is None:
            started: float = perf_counter()
            self.decoder = open_decoders(self._dataset, segment_id, [self._entity], self._timeline, self._device)[0]
            self.times = self.decoder.index.times
            self.query_seconds += perf_counter() - started
            self.query_count += 1
            self._segment_id = segment_id
        return self.decoder

    def decode_at(self, index_value: int | np.datetime64 | np.timedelta64, segment_id: str) -> FrameRgbChw | None:
        """Return a segment's latest frame at or before an index value."""
        decoder: ChunkedVideoDecoder = self._ensure_segment(segment_id)
        frame_index: int = int(np.searchsorted(self.times, index_value, side="right")) - 1
        if frame_index < 0:
            return None
        started: float = perf_counter()
        frame_chw: FrameRgbChw = decoder.get_frame_at(frame_index)
        self.decode_seconds += perf_counter() - started
        return frame_chw

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> Sequence[Tensor | None]:
        """Decode one aligned result per request in a fetched field block: an RGB ``FrameRgbChw``, or its ``transform``.

        Contiguous same-segment requests share one ``get_frames_at`` call, so a block costs one codec pass per segment
        instead of one seek per request.
        """
        del batch  # decoded from the segment's chunks, not the fetched point reads
        decoded: list[Tensor | None] = [None] * len(requests)
        start: int = 0
        while start < len(requests):
            segment_id: str = requests[start].segment_id
            stop: int = start
            while stop < len(requests) and requests[stop].segment_id == segment_id:
                stop += 1
            decoder: ChunkedVideoDecoder = self._ensure_segment(segment_id)
            positions: list[int] = []
            frame_indices: list[int] = []
            for position in range(start, stop):
                frame_index: int = int(np.searchsorted(self.times, requests[position].index_value, side="right")) - 1
                if frame_index >= 0:
                    positions.append(position)
                    frame_indices.append(frame_index)
            step: int = self._max_frames or max(len(frame_indices), 1)
            for first in range(0, len(frame_indices), step):
                started: float = perf_counter()
                frames_nchw: Tensor = decoder.get_frames_at(frame_indices[first : first + step])
                if self._transform is not None:
                    frames_nchw = self._transform(frames_nchw)
                self.decode_seconds += perf_counter() - started
                for row, position in enumerate(positions[first : first + step]):
                    decoded[position] = frames_nchw[row]
            start = stop
        return decoded


@dataclass(slots=True, frozen=True)
class TimedFrame:
    """One decoded camera frame and the grid timestamp (ns on the index timeline) it answers."""

    t_ns: int
    """Grid timestamp the frame was asked for, in nanoseconds on the index timeline."""
    rgb: FrameRgbChw
    """The segment's latest frame at or before ``t_ns``, channels-first RGB on the decoder's device."""


class TimedNvdecDecoder(SegmentNvdecDecoder):
    """``SegmentNvdecDecoder`` whose values also carry the grid timestamp; the dataloader sample has none.

    The index values must come from a duration timeline (or a plain nanosecond
    count); a timestamp timeline would need its own epoch handling.
    """

    # pyrefly: ignore  # bad-override — the base decodes to Tensor; the dataloader only moves the value into the sample dict.
    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[TimedFrame | None]:
        """Pair each decoded frame with the grid timestamp its request asked for; slots before the first sample stay None."""
        frames: Sequence[FrameRgbChw | None] = super().decode(batch, requests)
        return [
            None if frame is None else TimedFrame(t_ns=int(np.asarray(request.index_value).astype("timedelta64[ns]").astype(np.int64)), rgb=frame)
            for request, frame in zip(requests, frames, strict=True)
        ]


@dataclass(slots=True, frozen=True)
class KeyedFrame(Generic[DecodedT]):
    """One decoded value with the catalog key it answers; dataloader samples carry no key of their own."""

    segment_id: str
    index: int
    """The request's index value on an integer timeline (e.g. ``frame_index``)."""
    image: DecodedT
    """The decoded value: a ``Tensor`` from ``KeyedNvdecDecoder`` (the frame, or its ``transform``, on the decoder's device), a
    ``Yuv420Frame`` from ``KeyedVideoFrameDecoder``."""


class KeyedNvdecDecoder(SegmentNvdecDecoder):
    """``SegmentNvdecDecoder`` whose values also name their segment and integer index, so a consumer can join per-frame labels."""

    # pyrefly: ignore  # bad-override — the base decodes to Tensor; the dataloader only moves the value into the sample dict.
    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[KeyedFrame[Tensor] | None]:
        """Pair each decoded frame with its request's segment and index; slots before the first sample stay None."""
        frames: Sequence[Tensor | None] = super().decode(batch, requests)
        return [
            None if frame is None else KeyedFrame(segment_id=request.segment_id, index=int(request.index_value), image=frame)
            for request, frame in zip(requests, frames, strict=True)
        ]


class KeyedVideoFrameDecoder(VideoFrameDecoder[Literal["yuv420p"]]):
    """The dataloader's CPU ``VideoFrameDecoder`` (``yuv420p``) whose values also name their segment and integer index."""

    # pyrefly: ignore  # bad-override — the base decodes to Yuv420Frame; the dataloader only moves the value into the sample dict.
    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> list[KeyedFrame[Yuv420Frame] | None]:
        """Pair each decoded frame with its request's segment and index; samples the decoder cannot answer stay None."""
        frames: Sequence[Yuv420Frame | None] = super().decode(batch, requests)
        return [
            None if frame is None else KeyedFrame(segment_id=request.segment_id, index=int(request.index_value), image=frame)
            for request, frame in zip(requests, frames, strict=True)
        ]
