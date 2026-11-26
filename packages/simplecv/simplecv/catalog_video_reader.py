"""Bounded-memory reads of a segment's ``VideoStream``s from a Rerun catalog: an index first, packets per time chunk.

:func:`simplecv.catalog_video.read_catalog_videos` materializes whole columns: a 4 h, 6-camera RoboCap session is
~42 GB of packets, held up to three times over (the Arrow result, per-camera copies, a whole-stream MP4). This module
reads the same packets in pieces whose size does not depend on the session's length:

1. :func:`read_video_index`: every sample's timestamp and keyframe flag per stream, in time windows. The server reads
   the blobs of a window to answer it, so a window bounds what it holds; none are shipped.
2. :class:`CatalogVideoReader`: the streams cut into chunks of about ``chunk_s`` seconds. A chunk of a stream is a run
   of whole GOPs (it starts on a keyframe, and the next chunk starts on the next one), so each decodes alone and the
   chunks of a stream partition its samples: decoding chunk by chunk yields exactly the frames of one decoder over the
   whole stream. One query fetches a chunk of every stream.
3. :class:`ChunkedVideoDecoder` for random access, :func:`iter_chunks` and :class:`SharedChunks` for sequential reads
   with the next chunk fetched on a thread.

Two facts about the server decide the queries (rerun 0.42, ``rr.server.Server`` and ``rerun server``):

* The time filter must compare the index column with a literal of the column's own type. ``col(t).cast(int64)``
  is not pushed down: the server scans the whole recording for every query (a 30 s window of a 44 GB RoboCap base:
  56-130 s and 45 GB RSS, against 0.2-0.9 s and < 1 GB uncast).
* ``VideoStream:is_keyframe`` may be logged on keyframes only (dataforge's RoboCap converter does), and the codec is a
  static, so without its sample column a row cannot be attributed to a stream. The index therefore takes one query
  per stream (run concurrently); a chunk takes one query for all streams, split by the non-null sample column.
"""

import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import torch
from datafusion import DataFrame, Expr, col, lit
from jaxtyping import Bool, Int64, Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torch import Tensor

from simplecv.catalog_video import catalog_codec, catalog_keyframes, packet_views
from simplecv.catalog_video_codec import CatalogCodecName, wrap_mp4

INDEX_WINDOW_S: float = 300.0
"""Seconds of every stream one index query covers. The server reads that window's blobs to answer it: 300 s of six
RoboCap cameras is ~0.9 GB; a 4 h session takes 49 windows (~1 min on a local server)."""
CHUNK_S: float = 30.0
"""Seconds of every stream in a chunk: about 90 MB for six RoboCap cameras (1080p fisheye, ~0.5 MB/s each)."""
CACHE_BYTES: int = 1 << 30
"""Whole chunks, least recently used out."""
MP4_FPS: int = 30
"""Frame rate written into a chunk's MP4 wrapper; frames are addressed by index, so it only has to be valid."""


@dataclass(frozen=True, slots=True)
class StreamIndex:
    """One stream's sample timestamps and keyframe flags, without the sample bytes."""

    entity: str
    """With a leading slash."""
    times: Shaped[ndarray, " n"]
    """Strictly increasing, in the timeline's own dtype."""
    keyframes: Bool[ndarray, " n"]
    """A null flag counts as false: RoboCap logs ``is_keyframe`` on keyframes only."""
    codec: CatalogCodecName
    """From ``VideoStream:codec``, logged as a static or on every sample."""

    @property
    def t_ns(self) -> Int64[ndarray, " n"]:
        """The times as nanoseconds, without a copy."""
        return self.times.view(np.int64)


@dataclass(frozen=True, slots=True)
class StreamPackets:
    """One stream's samples in one chunk: whole GOPs, the first a keyframe (unless the stream starts mid-GOP)."""

    first: int
    """Stream sample index of ``samples[0]``."""
    samples: list[UInt8[ndarray, " n_bytes"]]
    """Views into the chunk's Arrow buffers, which they keep alive."""
    keyframes: list[bool]
    """One per sample."""

    def mp4(self, codec: CatalogCodecName) -> bytes:
        """An in-memory MP4 (one copy of the chunk's bytes) with sample ``i`` at frame ``i``."""
        return wrap_mp4(self.samples, self.keyframes, MP4_FPS, codec=codec)


@dataclass(frozen=True, slots=True)
class VideoChunk:
    """Every stream's samples in one chunk, from one query."""

    index: int
    """Position in :attr:`CatalogVideoReader.starts`."""
    streams: tuple[StreamPackets, ...]
    """In the reader's stream order."""
    nbytes: int
    """The whole query result, which the packet views keep alive: it also holds the neighbouring chunks' samples that
    the union of the streams' spans reaches."""


def time_window(timeline: str, timeline_type: pa.DataType, first: int, last: int) -> Expr:
    """``first <= timeline <= last`` with literals of the column's own type: a cast column is not pushed down."""
    return (col(timeline) >= lit(pa.scalar(first, type=timeline_type))) & (col(timeline) <= lit(pa.scalar(last, type=timeline_type)))


def _entity_path(entity: str) -> str:
    return f"/{entity.strip('/')}"


def _require_nanoseconds(timeline: str, timeline_type: pa.DataType) -> None:
    """Chunk and window sizes are seconds: a sequence timeline (frame numbers) has none, and would read in one piece."""
    if not (pa.types.is_duration(timeline_type) or pa.types.is_timestamp(timeline_type)) or timeline_type.unit != "ns":
        raise ValueError(f"{timeline} is {timeline_type}: chunks are cut in seconds, which needs a nanosecond duration or timestamp timeline")


def read_video_index(
    dataset: DatasetEntry, segment_id: str, entities: Sequence[str], timeline: str, *, window_s: float = INDEX_WINDOW_S
) -> tuple[StreamIndex, ...]:
    """Index every stream: one query per stream and window, the streams' queries run concurrently.

    Args:
        dataset: Dataset holding the segment.
        segment_id: Segment to read.
        entities: ``VideoStream`` entity paths, with or without a leading slash.
        timeline: A nanosecond duration or timestamp timeline.
        window_s: Seconds per query; the server holds that window's blobs to answer it.

    Returns:
        One index per entity, in order.

    Raises:
        ValueError: If a stream has no samples, no codec, or timestamps that do not strictly increase.
    """
    paths: list[str] = [_entity_path(entity) for entity in entities]
    selection = dataset.filter_segments(segment_id)
    ranges: pa.Table = selection.get_index_ranges().to_arrow_table()
    if ranges.num_rows != 1:
        raise ValueError(f"{segment_id}: expected one segment in the index ranges, got {ranges.num_rows}")
    start: int = int(ranges[f"{timeline}:start"].combine_chunks().cast(pa.int64())[0].as_py())
    end: int = int(ranges[f"{timeline}:end"].combine_chunks().cast(pa.int64())[0].as_py())
    window_ns: int = max(1, int(window_s * 1e9))

    def read_stream(path: str) -> StreamIndex:
        reader: DataFrame = selection.filter_contents([path]).reader(index=timeline)
        schema: pa.Schema = reader.schema()
        columns: list[str] = [f"{path}:VideoStream:{component}" for component in ("is_keyframe", "codec")]
        if f"{path}:VideoStream:sample" not in schema.names or columns[0] not in schema.names:
            raise ValueError(f"{segment_id} {path}: expected samples with keyframe flags on {timeline}, the recording has none")
        if columns[1] not in schema.names:
            raise ValueError(f"{segment_id} {path}: the video stream carries no codec, so its samples cannot be decoded")
        timeline_type: pa.DataType = schema.field(timeline).type
        _require_nanoseconds(timeline, timeline_type)
        tables: list[pa.Table] = []
        for first in range(start, end + 1, window_ns):
            window: Expr = time_window(timeline, timeline_type, first, min(first + window_ns - 1, end))
            # The codec is a static on most recordings (then on every row) and a temporal on some; either way one uint32 a row.
            tables.append(reader.filter(window).select(timeline, *columns).to_arrow_table())
        table: pa.Table = pa.concat_tables(tables)
        times: Shaped[ndarray, " n"] = table[timeline].combine_chunks().to_numpy(zero_copy_only=False)
        if not len(times) or not np.all(np.diff(times.view(np.int64)) > 0):
            raise ValueError(f"{segment_id} {path}: expected samples with strictly increasing timestamps")
        keyframes: Bool[ndarray, " n"] = np.asarray(catalog_keyframes(table[columns[0]]), dtype=bool)
        return StreamIndex(entity=path, times=times, keyframes=keyframes, codec=catalog_codec(table[columns[1]], f"{segment_id} {path}"))

    with ThreadPoolExecutor(max_workers=len(paths), thread_name_prefix="video-index") as pool:
        futures: list[Future[StreamIndex]] = [pool.submit(read_stream, path) for path in paths]
        return tuple(future.result() for future in futures)


def chunk_starts(streams: Sequence[StreamIndex], chunk_ns: int) -> Int64[ndarray, "n_bounds n_streams"]:
    """Chunk ``k`` of stream ``s`` is its samples ``[starts[k, s], starts[k + 1, s])``.

    Chunk ``k`` covers the time grid ``[t0 + k * chunk_ns, t0 + (k + 1) * chunk_ns)``, moved back per stream to the
    keyframe at or before each edge, so every chunk of a stream starts on a keyframe and the chunks partition it. The
    first chunk starts at sample 0 whatever it is; a chunk with no keyframe in its span is empty and the GOP stays
    whole in an earlier chunk, so a stream's GOP length bounds its chunks from below.

    Args:
        streams: The streams' indexes.
        chunk_ns: Grid step.

    Returns:
        ``n_chunks + 1`` rows: row 0 is all zeros, the last row each stream's sample count.

    Raises:
        ValueError: If a stream has no keyframe: nothing of it decodes, and it would be one chunk of the whole session.
    """
    t0: int = min(int(stream.t_ns[0]) for stream in streams)
    t_end: int = max(int(stream.t_ns[-1]) for stream in streams)
    n_chunks: int = max(1, -(-(t_end - t0 + 1) // chunk_ns))
    edges: Int64[ndarray, " n_inner"] = t0 + chunk_ns * np.arange(1, n_chunks, dtype=np.int64)
    columns: list[Int64[ndarray, " n_bounds"]] = []
    for stream in streams:
        keyframe_at: Int64[ndarray, " n_keyframes"] = np.flatnonzero(stream.keyframes).astype(np.int64)
        if not len(keyframe_at):
            raise ValueError(f"{stream.entity}: the stream has no keyframe, so none of it decodes")
        # Sample 0 for edges before the first keyframe; no running maximum needed, keyframe positions increase.
        position: Int64[ndarray, " n_inner"] = np.searchsorted(stream.t_ns[keyframe_at], edges, side="right") - 1
        inner: Int64[ndarray, " n_inner"] = np.where(position >= 0, keyframe_at[np.maximum(position, 0)], 0)
        columns.append(np.concatenate([[0], inner, [len(stream.t_ns)]]).astype(np.int64))
    return np.stack(columns, axis=1)


class CatalogVideoReader:
    """Memory is a few chunks, whatever the segment's length. :meth:`packets` and :meth:`fetch` are thread-safe."""

    def __init__(
        self,
        dataset: DatasetEntry,
        segment_id: str,
        streams: Sequence[StreamIndex],
        timeline: str,
        *,
        chunk_s: float = CHUNK_S,
        cache_bytes: int = CACHE_BYTES,
    ) -> None:
        """Plan the chunks; nothing is fetched until a chunk is asked for.

        Args:
            dataset: Dataset holding the segment.
            segment_id: Segment to read.
            streams: From :func:`read_video_index`; chunks list the streams in this order.
            timeline: A nanosecond duration or timestamp timeline.
            chunk_s: Seconds of every stream per chunk.
            cache_bytes: What :meth:`packets` keeps; the most recent chunk is kept even when larger.
        """
        if not streams:
            raise ValueError("a reader needs at least one stream")
        self.segment_id: str = segment_id
        self.streams: tuple[StreamIndex, ...] = tuple(streams)
        self.timeline: str = timeline
        self.starts: Int64[ndarray, "n_bounds n_streams"] = chunk_starts(self.streams, max(1, int(chunk_s * 1e9)))
        self.cache_bytes: int = cache_bytes
        # The dataset, not a DataFrame built from it: a DataFrame that outlives its client's session fails to plan.
        self._dataset: DatasetEntry = dataset
        self._timeline_type: pa.DataType = self._query().schema().field(timeline).type
        _require_nanoseconds(timeline, self._timeline_type)
        self._cache: OrderedDict[int, VideoChunk] = OrderedDict()
        self._cached_bytes: int = 0
        self._lock: threading.Lock = threading.Lock()
        self.fetched_bytes: int = 0
        """Refetches of evicted chunks included."""
        self.fetch_count: int = 0

    def _query(self) -> DataFrame:
        return self._dataset.filter_segments(self.segment_id).filter_contents([stream.entity for stream in self.streams]).reader(index=self.timeline)

    @property
    def num_chunks(self) -> int:
        """The same for every stream; a stream's chunk may be empty."""
        return len(self.starts) - 1

    def chunk_of(self, stream: int, sample: int) -> int:
        """The chunk holding a stream's sample index."""
        return int(np.searchsorted(self.starts[:, stream], sample, side="right")) - 1

    def fetch(self, chunk: int) -> VideoChunk:
        """One chunk of every stream in one query, not cached.

        Raises:
            ValueError: If a stream's fetched timestamps are not the ones its index lists for the chunk.
        """
        bounds: list[tuple[int, int]] = [(int(self.starts[chunk, s]), int(self.starts[chunk + 1, s])) for s in range(len(self.streams))]
        spans: list[tuple[int, int]] = [
            (int(stream.t_ns[first]), int(stream.t_ns[stop - 1])) for stream, (first, stop) in zip(self.streams, bounds, strict=True) if stop > first
        ]
        if not spans:
            return VideoChunk(index=chunk, streams=tuple(StreamPackets(first=first, samples=[], keyframes=[]) for first, _ in bounds), nbytes=0)
        columns: list[str] = [f"{stream.entity}:VideoStream:sample" for stream in self.streams]
        window: Expr = time_window(self.timeline, self._timeline_type, min(span[0] for span in spans), max(span[1] for span in spans))
        table: pa.Table = self._query().filter(window).select(self.timeline, *columns).to_arrow_table()
        times_ns: Int64[ndarray, " n_rows"] = table[self.timeline].combine_chunks().to_numpy(zero_copy_only=False).view(np.int64)
        packets: list[StreamPackets] = []
        for stream, (first, stop), column in zip(self.streams, bounds, columns, strict=True):
            samples: pa.ChunkedArray = table[column]
            valid: Bool[ndarray, " n_rows"] = samples.is_valid().to_numpy(zero_copy_only=False)
            own_ns: Int64[ndarray, " n_own"] = times_ns[valid]
            # The query spans every stream's chunk, so a stream's rows reach into its neighbouring chunks.
            lo: int = int(np.searchsorted(own_ns, stream.t_ns[first], side="left")) if stop > first else 0
            hi: int = lo + (stop - first)
            if not np.array_equal(own_ns[lo:hi], stream.t_ns[first:stop]):
                raise ValueError(
                    f"{self.segment_id} {stream.entity}: chunk {chunk} returned other samples than the index lists for [{first}, {stop})"
                )
            views: list[UInt8[ndarray, " n_bytes"]] = packet_views(samples)[lo:hi] if stop > first else []
            packets.append(StreamPackets(first=first, samples=views, keyframes=stream.keyframes[first:stop].tolist()))
        fetched: VideoChunk = VideoChunk(index=chunk, streams=tuple(packets), nbytes=table.nbytes)
        with self._lock:
            self.fetched_bytes += fetched.nbytes
            self.fetch_count += 1
        return fetched

    def packets(self, chunk: int) -> VideoChunk:
        """:meth:`fetch` through the byte-bounded cache, for random access."""
        with self._lock:
            if chunk in self._cache:
                self._cache.move_to_end(chunk)
                return self._cache[chunk]
        fetched: VideoChunk = self.fetch(chunk)
        with self._lock:
            if chunk not in self._cache:
                self._cache[chunk] = fetched
                self._cached_bytes += fetched.nbytes
            while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= evicted.nbytes
        return fetched


def iter_chunks(reader: CatalogVideoReader, chunks: Sequence[int]) -> Iterator[VideoChunk]:
    """The chunks in order, the next one fetched on a thread while the caller works on this one.

    Holds at most two fetched chunks besides the caller's; bypasses the reader's cache.
    """
    if not chunks:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-read-ahead") as pool:
        pending: Future[VideoChunk] = pool.submit(reader.fetch, chunks[0])
        for following in chunks[1:]:
            current: VideoChunk = pending.result()
            pending = pool.submit(reader.fetch, following)
            yield current
        yield pending.result()


class SharedChunks:
    """Each stream walks its chunks at its own pace from one read-ahead (:func:`iter_chunks`) of every stream.

    A chunk is held until every stream has taken its part, so memory is the chunks between the slowest and the fastest
    stream, bounded by ``max_held``. Not thread-safe.
    """

    def __init__(self, reader: CatalogVideoReader, chunks: Sequence[int], *, max_held: int = 4) -> None:
        """Start the read-ahead.

        Args:
            reader: Where the chunks come from.
            chunks: Increasing.
            max_held: How far a stream may run ahead of the slowest one; further raises rather than hold the session.
        """
        self._chunks: list[int] = list(chunks)
        self._max_held: int = max_held
        self._source: Iterator[VideoChunk] = iter_chunks(reader, self._chunks)
        self._streams: int = len(reader.streams)
        self._held: dict[int, tuple[VideoChunk, set[int]]] = {}

    @property
    def held(self) -> int:
        """Fetched chunks that some stream has not taken yet."""
        return len(self._held)

    def _take(self, chunk: int, stream: int) -> StreamPackets:
        while chunk not in self._held:
            if len(self._held) >= self._max_held:
                raise ValueError(f"stream {stream} is {self._max_held} chunks ahead of the slowest stream; walk the streams together")
            fetched: VideoChunk = next(self._source)
            self._held[fetched.index] = (fetched, set())
        held: tuple[VideoChunk, set[int]] = self._held[chunk]
        held[1].add(stream)
        if len(held[1]) == self._streams:
            del self._held[chunk]
        return held[0].streams[stream]

    def stream(self, stream: int) -> Iterator[StreamPackets]:
        """One stream's packets of each chunk, in order."""
        for chunk in self._chunks:
            yield self._take(chunk, stream)


@runtime_checkable
class _FrameBatch(Protocol):
    """``torchcodec``'s ``FrameBatch``, which is imported lazily."""

    data: Tensor


@runtime_checkable
class _ChunkDecoder(Protocol):
    """``torchcodec``'s ``VideoDecoder``: a CPU lane that only reads packets has no torchcodec."""

    def get_frames_at(self, indices: list[int]) -> _FrameBatch: ...


class ChunkedVideoDecoder:
    """One stream's frames by sample index, decoded from the chunks that hold them.

    The same frames as a whole-stream ``VideoDecoder(seek_mode="exact")``, because every chunk starts on a keyframe.
    ``max_open`` chunk decoders, each holding its chunk's MP4, stay open, so requests in sample order decode each chunk
    once.
    """

    def __init__(self, reader: CatalogVideoReader, stream: int, device: torch.device, *, max_open: int = 2) -> None:
        """Nothing is fetched until frames are asked for.

        Args:
            reader: Share one between a rig's decoders: one query then serves every stream's chunk.
            stream: Position of the stream in the reader.
            device: ``cuda`` decodes on NVDEC.
            max_open: Chunk decoders kept open.
        """
        self.reader: CatalogVideoReader = reader
        self.stream: int = stream
        self.device: torch.device = device
        self.max_open: int = max_open
        self._open: OrderedDict[int, _ChunkDecoder] = OrderedDict()

    def _decoder(self, chunk: int) -> _ChunkDecoder:
        from torchcodec.decoders import VideoDecoder

        if chunk in self._open:
            self._open.move_to_end(chunk)
            return self._open[chunk]
        packets: StreamPackets = self.reader.packets(chunk).streams[self.stream]
        decoder: _ChunkDecoder = VideoDecoder(
            packets.mp4(self.reader.streams[self.stream].codec), device=str(self.device), seek_mode="exact", num_ffmpeg_threads=0
        )
        self._open[chunk] = decoder
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return decoder

    def get_frames_at(self, indices: Sequence[int]) -> UInt8[Tensor, "n 3 h w"]:
        """In the order asked, repeats allowed.

        Raises:
            IndexError: If an index is outside the stream.
        """
        wanted: Int64[ndarray, " n"] = np.asarray(indices, dtype=np.int64)
        count: int = len(self.reader.streams[self.stream].t_ns)
        if wanted.size and (wanted.min() < 0 or wanted.max() >= count):
            raise IndexError(f"{self.reader.streams[self.stream].entity}: sample indices must be in [0, {count})")
        unique: Int64[ndarray, " u"] = np.unique(wanted)
        chunk_of: Int64[ndarray, " u"] = np.searchsorted(self.reader.starts[:, self.stream], unique, side="right") - 1
        parts: list[Tensor] = []
        for chunk in np.unique(chunk_of).tolist():
            local: list[int] = (unique[chunk_of == chunk] - int(self.reader.starts[chunk, self.stream])).tolist()
            parts.append(self._decoder(int(chunk)).get_frames_at(local).data)
        if not parts:
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=self.device)
        # colsfm's usual request, sorted within one chunk, needs no copy.
        frames: Tensor = parts[0] if len(parts) == 1 else torch.cat(parts)
        if np.array_equal(unique, wanted):
            return frames
        return frames[torch.from_numpy(np.searchsorted(unique, wanted)).to(frames.device)]

    def close(self) -> None:
        """Release the open chunk decoders and their MP4s."""
        self._open.clear()


__all__ = (
    "CatalogVideoReader",
    "ChunkedVideoDecoder",
    "SharedChunks",
    "StreamIndex",
    "StreamPackets",
    "VideoChunk",
    "chunk_starts",
    "iter_chunks",
    "read_video_index",
    "time_window",
)
