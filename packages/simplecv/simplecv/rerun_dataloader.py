"""GPU video decoding for the Rerun catalog dataloader.

Import this module explicitly: it needs ``rerun-sdk``'s ``dataloader`` extra,
which only the catalog lanes install, so it must never be re-exported from
``simplecv/__init__.py``.

Upstream's video decoder (rerun-io/reality PR #2893) wraps each sample's
keyframe window into its own in-memory MP4 and is slower than dav1d — no
decode-session reuse across calls. ``SegmentNvdecDecoder`` instead bulk-fetches
a segment's packets once, wraps them into a single MP4 (same ``add_mux_stream``
technique), and serves every sample from one GPU decoder.

TODO(rerun#upstream): delete this module once the dataloader decodes video fast
enough on its own — this is a stopgap, and it costs real flexibility: holding a
``DatasetEntry`` makes it unpicklable (so ``num_workers`` must stay 0), and the
cached CUDA decoder does not survive ``decode_threads > 1``. Upstream's batch
decoder now reuses one codec context across a GOP, but its PyAV CPU path remains
slower than this segment-wide torchcodec/NVDEC path.
"""

from collections.abc import Sequence
from fractions import Fraction
from io import BytesIO
from typing import TypeAlias

import av
import numpy as np
import torch
from jaxtyping import Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView
from rerun.experimental.dataloader import ColumnDecoder, DecodeRequest, FieldBatch
from torch import Tensor
from torchcodec.decoders import VideoDecoder

TimedeltaNs: TypeAlias = Shaped[ndarray, " n_samples"]
"""Sample timestamps in timeline order, dtype ``timedelta64[ns]`` (jaxtyping has no timedelta dtype)."""
FrameRgbChw: TypeAlias = UInt8[Tensor, "3 h w"]
"""One decoded frame, channels-first RGB on the decoder's device."""
RECOMMENDED_FETCH_BLOCK_SIZE: int = 1024
"""Recommended samples per query when a decoder ignores the shipped payloads."""


def open_segment_decoder(
    dataset: DatasetEntry, segment_id: str, entity: str, timeline: str, device: torch.device, fps: int
) -> tuple[TimedeltaNs, list[bytes], list[bool], VideoDecoder]:
    """Fetch one segment's video packets (one query), wrap them, and open a GPU decoder.

    Args:
        dataset: Rerun catalog dataset entry holding the segment.
        segment_id: Segment whose packets are fetched.
        entity: Entity path of the ``VideoStream`` column, without a leading slash.
        timeline: Index timeline the packets are read on. The reader's row order
            is trusted; the ordering guard below fails loudly if it breaks.
        device: Device the decoder outputs frames on (``cuda`` uses NVDEC).
        fps: Nominal frame rate written into the wrapping MP4 track.

    Returns:
        The sample timestamps (timedelta64[ns], timeline order), the raw video
        samples with their keyframe flags (relayable as a Rerun VideoStream),
        and the decoder over the whole segment.
    """
    view: DatasetView = dataset.filter_segments(segment_id).filter_contents(entity)
    # No .sort(timeline): the reader already yields (segment, index)-ordered rows, and a
    # client-side SortExec re-materializes the blob columns (~4x the query wall time).
    # The ordering is an implicit server contract, so the guard below fails loudly if it
    # ever breaks instead of silently corrupting packet order.
    table = (
        view.reader(index=timeline)
        .select(timeline, f"/{entity}:VideoStream:sample", f"/{entity}:VideoStream:is_keyframe")
        .to_arrow_table()
    )
    times: TimedeltaNs = table[0].combine_chunks().to_numpy(zero_copy_only=False)
    if np.any(times[1:] < times[:-1]):
        raise ValueError(f"segment {segment_id}: reader returned rows out of timeline order; the no-sort fast path assumes index order")
    blobs = table[1].combine_chunks().flatten()
    data = memoryview(blobs.flatten().buffers()[1])
    offsets: list[int] = blobs.offsets.to_pylist()
    samples: list[bytes] = [bytes(data[start:end]) for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    keyframes: list[bool] = [bool(flag) for flag in table[2].combine_chunks().flatten().to_pylist()]
    decoder: VideoDecoder = VideoDecoder(wrap_mp4(samples, keyframes, fps), device=device, seek_mode="exact", num_ffmpeg_threads=0)
    return times, samples, keyframes, decoder


def wrap_mp4(samples: list[bytes], keyframes: list[bool], fps: int, codec: str = "av1") -> bytes:
    """Mux pre-encoded samples into an in-memory MP4 with positional pts (no re-encode).

    ``add_mux_stream`` muxes without instantiating an encoder. The nominal 16x16
    stream dimensions are irrelevant: decoders read the real dimensions from the
    bitstream (parameter sets travel in-band).

    Args:
        samples: Encoded video samples in decode order.
        keyframes: Keyframe flag per sample.
        fps: Frame rate for the muxed track and its time base.
        codec: Codec name of the pre-encoded samples.

    Returns:
        The complete MP4 file as bytes.
    """
    buffer: BytesIO = BytesIO()
    # Pin the mp4 track timescale to fps: the muxer otherwise picks its own (15360)
    # without rescaling our positional pts, and the track then claims a ~0.1s
    # duration. Index-seeking decoders never notice; timestamp readers do.
    with av.open(buffer, "w", format="mp4", options={"video_track_timescale": str(fps)}) as container:
        stream = container.add_mux_stream(codec, rate=fps, width=16, height=16)
        stream.time_base = Fraction(1, fps)
        for sample_index, (sample, is_keyframe) in enumerate(zip(samples, keyframes, strict=True)):
            packet: av.Packet = av.Packet(sample)
            packet.pts = packet.dts = sample_index
            packet.duration = 1
            packet.time_base = stream.time_base
            packet.stream = stream
            packet.is_keyframe = is_keyframe
            container.mux(packet)
    return buffer.getvalue()


class SegmentNvdecDecoder(ColumnDecoder[FrameRgbChw]):
    """Serve a fetched request block from the cached whole-segment GPU decoder.

    The base-class defaults (no ``prior_keyframe_path``) make the dataloader ship each
    grid slot as a point read. The shipped bytes are ignored because frames come from
    the cached segment-wide NVDEC decoder instead.
    """

    def __init__(self, dataset: DatasetEntry, entity: str, timeline: str, device: torch.device, fps: int) -> None:
        """Bind the decoder to one dataset's video column.

        Args:
            dataset: Rerun catalog dataset entry the segments come from.
            entity: Entity path of the ``VideoStream`` column, without a leading slash.
            timeline: Index timeline the packets are read on (row order per
                ``open_segment_decoder``'s ordering guard).
            device: Device decoded frames land on (``cuda`` uses NVDEC).
            fps: Nominal frame rate of the stored video.
        """
        self._dataset = dataset
        self._entity = entity
        self._timeline = timeline
        self._device = device
        self._fps = fps
        self._segment_id: str | None = None
        self._decoder: VideoDecoder | None = None
        self.times: TimedeltaNs = np.empty(0, dtype="timedelta64[ns]")
        """The current segment's sample timestamps (timedelta64[ns], timeline order)."""
        self.samples: list[bytes] | None = None
        """The current segment's raw video samples, relayable as a Rerun VideoStream."""
        self.keyframes: list[bool] | None = None
        """Keyframe flag per raw sample."""

    def _ensure_segment(self, segment_id: str) -> None:
        if segment_id != self._segment_id:
            self.times, self.samples, self.keyframes, self._decoder = open_segment_decoder(
                self._dataset, segment_id, self._entity, self._timeline, self._device, self._fps
            )
            self._segment_id = segment_id

    def decode_at(self, index_value: int | np.datetime64 | np.timedelta64, segment_id: str) -> FrameRgbChw | None:
        """Return a segment's latest frame at or before an index value."""
        self._ensure_segment(segment_id)
        frame_index: int = int(np.searchsorted(self.times, index_value, side="right")) - 1
        if frame_index < 0:
            return None
        assert self._decoder is not None
        frame_chw: FrameRgbChw = self._decoder.get_frame_at(frame_index).data
        return frame_chw

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> Sequence[FrameRgbChw | None]:
        """Decode one aligned result per request in a fetched field block.

        Contiguous same-segment requests share one ``get_frames_at`` call, so a
        block costs one codec pass per segment instead of one seek per request.
        """
        del batch  # decoded from the segment-wide cache, not the fetched point reads
        decoded: list[FrameRgbChw | None] = [None] * len(requests)
        start: int = 0
        while start < len(requests):
            segment_id: str = requests[start].segment_id
            stop: int = start
            while stop < len(requests) and requests[stop].segment_id == segment_id:
                stop += 1
            self._ensure_segment(segment_id)
            positions: list[int] = []
            frame_indices: list[int] = []
            for position in range(start, stop):
                frame_index: int = int(np.searchsorted(self.times, requests[position].index_value, side="right")) - 1
                if frame_index >= 0:
                    positions.append(position)
                    frame_indices.append(frame_index)
            if frame_indices:
                assert self._decoder is not None
                frames_nchw: UInt8[Tensor, "n 3 h w"] = self._decoder.get_frames_at(frame_indices).data
                for row, position in enumerate(positions):
                    decoded[position] = frames_nchw[row]
            start = stop
        return decoded
