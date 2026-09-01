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
cached CUDA decoder does not survive ``decode_threads > 1``. Two upstream PRs
converge on it: reality#2893 adds a torchcodec decoder (slower than the av path
today, by its author's own measurement), and reality#3083 replaces per-sample
``decode`` with a batch API whose ``_DecoderSession`` reuses one codec context
across a GOP — the session reuse this module hand-rolls. #3083 also changes the
``ColumnDecoder.decode`` signature, so this class needs a port when it lands.
"""

from fractions import Fraction
from io import BytesIO
from typing import TypeAlias

import av
import numpy as np
import pyarrow as pa
import torch
from jaxtyping import Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView
from rerun.experimental.dataloader import ColumnDecoder
from torch import Tensor
from torchcodec.decoders import VideoDecoder

TimedeltaNs: TypeAlias = Shaped[ndarray, " n_samples"]
"""Sample timestamps in timeline order, dtype ``timedelta64[ns]`` (jaxtyping has no timedelta dtype)."""
FrameRgbChw: TypeAlias = UInt8[Tensor, "3 h w"]
"""One decoded frame, channels-first RGB on the decoder's device."""


def open_segment_decoder(
    dataset: DatasetEntry, segment_id: str, entity: str, timeline: str, device: torch.device, fps: int
) -> tuple[TimedeltaNs, list[bytes], list[bool], VideoDecoder]:
    """Fetch one segment's video packets (one query), wrap them, and open a GPU decoder.

    Args:
        dataset: Rerun catalog dataset entry holding the segment.
        segment_id: Segment whose packets are fetched.
        entity: Entity path of the ``VideoStream`` column, without a leading slash.
        timeline: Index timeline the packets are read and sorted on.
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
    times: TimedeltaNs = np.array([t.value for t in table[0]], dtype="timedelta64[ns]")
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


class SegmentNvdecDecoder(ColumnDecoder):
    """Serve the dataloader's per-sample decode calls from one whole-segment GPU decoder.

    The base-class defaults (no ``prior_keyframe_path``, no ``context_range``) make the
    dataloader ship each grid slot as a point read; the shipped bytes are ignored and the
    frame comes from the cached segment-wide NVDEC decoder instead.
    """

    def __init__(self, dataset: DatasetEntry, entity: str, timeline: str, device: torch.device, fps: int) -> None:
        """Bind the decoder to one dataset's video column.

        Args:
            dataset: Rerun catalog dataset entry the segments come from.
            entity: Entity path of the ``VideoStream`` column, without a leading slash.
            timeline: Index timeline the packets are read and sorted on.
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

    def decode(self, raw: pa.ChunkedArray, index_value: int | np.datetime64 | np.timedelta64, segment_id: str) -> FrameRgbChw | None:
        """Return the segment's latest frame at or before *index_value*, or None before the first sample."""
        del raw  # decoded from the segment-wide cache, not the shipped point read
        if segment_id != self._segment_id:
            self.times, self.samples, self.keyframes, self._decoder = open_segment_decoder(
                self._dataset, segment_id, self._entity, self._timeline, self._device, self._fps
            )
            self._segment_id = segment_id
        frame_index: int = int(np.searchsorted(self.times, index_value, side="right")) - 1
        if frame_index < 0:
            return None
        assert self._decoder is not None
        frame_chw: FrameRgbChw = self._decoder.get_frame_at(frame_index).data
        return frame_chw
