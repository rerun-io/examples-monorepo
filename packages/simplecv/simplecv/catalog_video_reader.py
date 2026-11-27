"""Bounded-memory reads of a segment's ``VideoStream``s from a Rerun catalog: an index first, packets per time chunk.

A whole-column read of a 4 h, 6-camera RoboCap session is ~42 GB of packets, and decoding it used to hold them up to
three times over (the Arrow result, per-camera copies, a whole-stream MP4). This module reads the same packets in
pieces whose size does not depend on the session's length:

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
from functools import cache
from importlib.util import find_spec
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from datafusion import DataFrame, Expr, col, lit
from jaxtyping import Bool, Int64, Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from torch import Tensor

from simplecv.catalog_video import catalog_codec, catalog_keyframes, packet_views
from simplecv.catalog_video_codec import CatalogCodecName, video_codec, wrap_mp4

INDEX_WINDOW_S: float = 300.0
"""Seconds of every stream one index query covers. The server reads that window's blobs to answer it: 300 s of six
RoboCap cameras is ~0.9 GB; a 4 h session takes 49 windows (~1 min on a local server)."""
CHUNK_S: float = 30.0
"""Seconds of every stream in a chunk: about 90 MB for six RoboCap cameras (1080p fisheye, ~0.5 MB/s each)."""
CACHE_BYTES: int = 1 << 30
"""Whole chunks, least recently used out."""
MP4_FPS: int = 30
"""Frame rate written into a chunk's MP4 wrapper; frames are addressed by index, so it only has to be valid.

Also how a sequence timeline turns ``chunk_s`` and ``window_s`` into index units: it counts frames at this rate."""

DecodeRequest: TypeAlias = Literal["auto", "cuda", "cpu"]
DecodeDevice: TypeAlias = Literal["cpu", "cuda"]


@cache
def nvdec_unavailable_reason() -> str | None:
    """None when NVDEC can decode in this process; discovered once."""
    if find_spec("torchcodec") is None:
        return "this environment has no torchcodec"
    if not torch.cuda.is_available():
        return "torch sees no CUDA device"
    return None


_CPU_NOTICE_PRINTED: list[bool] = []
_CPU_NOTICE_LOCK: threading.Lock = threading.Lock()


def select_decode_device(requested: DecodeRequest = "auto") -> DecodeDevice:
    """NVDEC whenever CUDA is available; the CPU only without it, said once per process, never silently.

    Raises:
        ValueError: If ``cuda`` is requested and NVDEC is unavailable.
    """
    if requested == "cpu":
        return "cpu"
    reason: str | None = nvdec_unavailable_reason()
    if reason is None:
        return "cuda"
    if requested == "cuda":
        raise ValueError(f"NVDEC decoding is unavailable: {reason}")
    with _CPU_NOTICE_LOCK:
        if not _CPU_NOTICE_PRINTED:
            _CPU_NOTICE_PRINTED.append(True)
            print(f"simplecv: decoding on the CPU because NVDEC is unavailable: {reason}", flush=True)
    return "cpu"


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

    @property
    def lead(self) -> int:
        """Samples before the first keyframe: non-zero only in a stream's first chunk when the stream starts mid-GOP."""
        return next((position for position, key in enumerate(self.keyframes) if key), len(self.keyframes))

    def mp4(self, codec: CatalogCodecName) -> bytes:
        """An in-memory MP4 (one copy of the chunk's bytes) from the first keyframe: frame ``i`` is sample ``lead + i``.

        Samples before a keyframe cannot be decoded, and an MP4 that opens on one carries no parameter sets to open with.
        """
        return wrap_mp4(self.samples[self.lead :], self.keyframes[self.lead :], MP4_FPS, codec=codec)


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


def _units_per_second(timeline: str, timeline_type: pa.DataType) -> float:
    """Index units in a second: a temporal timeline's own unit, or :data:`MP4_FPS` frames on a sequence timeline."""
    if pa.types.is_duration(timeline_type) or pa.types.is_timestamp(timeline_type):
        return {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[timeline_type.unit]
    if pa.types.is_integer(timeline_type):
        return float(MP4_FPS)
    raise ValueError(f"{timeline} is {timeline_type}, neither a temporal nor a sequence timeline")


def read_video_index(
    dataset: DatasetEntry, segment_id: str, entities: Sequence[str], timeline: str, *, window_s: float = INDEX_WINDOW_S
) -> tuple[StreamIndex, ...]:
    """Index every stream: one query per stream and window, the streams' queries run concurrently.

    Args:
        dataset: Dataset holding the segment.
        segment_id: Segment to read.
        entities: ``VideoStream`` entity paths, with or without a leading slash.
        timeline: A temporal or a sequence timeline.
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

    def read_stream(path: str) -> StreamIndex:
        reader: DataFrame = selection.filter_contents([path]).reader(index=timeline)
        schema: pa.Schema = reader.schema()
        columns: list[str] = [f"{path}:VideoStream:{component}" for component in ("is_keyframe", "codec")]
        if f"{path}:VideoStream:sample" not in schema.names or columns[0] not in schema.names:
            raise ValueError(f"{segment_id} {path}: expected samples with keyframe flags on {timeline}, the recording has none")
        if columns[1] not in schema.names:
            raise ValueError(f"{segment_id} {path}: the video stream carries no codec, so its samples cannot be decoded")
        timeline_type: pa.DataType = schema.field(timeline).type
        window_units: int = max(1, int(window_s * _units_per_second(timeline, timeline_type)))
        tables: list[pa.Table] = []
        for first in range(start, end + 1, window_units):
            window: Expr = time_window(timeline, timeline_type, first, min(first + window_units - 1, end))
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
            timeline: A temporal or a sequence timeline.
            chunk_s: Seconds of every stream per chunk.
            cache_bytes: What :meth:`packets` keeps; the most recent chunk is kept even when larger.
        """
        if not streams:
            raise ValueError("a reader needs at least one stream")
        self.segment_id: str = segment_id
        self.streams: tuple[StreamIndex, ...] = tuple(streams)
        self.timeline: str = timeline
        self.cache_bytes: int = cache_bytes
        # The dataset, not a DataFrame built from it: a DataFrame that outlives its client's session fails to plan.
        self._dataset: DatasetEntry = dataset
        self.timeline_type: pa.DataType = self._query().schema().field(timeline).type
        units: float = _units_per_second(timeline, self.timeline_type)
        self.starts: Int64[ndarray, "n_bounds n_streams"] = chunk_starts(self.streams, max(1, int(chunk_s * units)))
        self._cache: OrderedDict[int, VideoChunk] = OrderedDict()
        self._inflight: dict[int, Future[VideoChunk]] = {}
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
        window: Expr = time_window(self.timeline, self.timeline_type, min(span[0] for span in spans), max(span[1] for span in spans))
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
        """:meth:`fetch` through the byte-bounded cache, for random access; threads that miss one chunk at once share its fetch."""
        with self._lock:
            if chunk in self._cache:
                self._cache.move_to_end(chunk)
                return self._cache[chunk]
            owner: bool = chunk not in self._inflight
            if owner:
                self._inflight[chunk] = Future()
            pending: Future[VideoChunk] = self._inflight[chunk]
        if not owner:
            return pending.result()
        try:
            fetched: VideoChunk = self.fetch(chunk)
        except BaseException as error:
            with self._lock:
                del self._inflight[chunk]
            pending.set_exception(error)
            raise
        with self._lock:
            del self._inflight[chunk]
            self._cache[chunk] = fetched
            self._cached_bytes += fetched.nbytes
            while self._cached_bytes > self.cache_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._cached_bytes -= evicted.nbytes
        pending.set_result(fetched)
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

    @property
    def cpu_fallback(self) -> object:
        """Truthy once NVDEC has declined the stream and the CPU decodes it."""
        ...

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
        self._open: OrderedDict[int, tuple[int, _ChunkDecoder]] = OrderedDict()
        """Per open chunk: the stream sample index of the decoder's frame 0, and the decoder."""
        self._fallback_said: bool = False

    @property
    def index(self) -> StreamIndex:
        """The stream's sample times and flags; sample ``i`` is ``index.times[i]``."""
        return self.reader.streams[self.stream]

    def __len__(self) -> int:
        return len(self.index.times)

    def _decoder(self, chunk: int) -> tuple[int, _ChunkDecoder]:
        from torchcodec.decoders import VideoDecoder

        if chunk in self._open:
            self._open.move_to_end(chunk)
            return self._open[chunk]
        packets: StreamPackets = self.reader.packets(chunk).streams[self.stream]
        if packets.lead == len(packets.samples):
            raise ValueError(f"{self.index.entity}: chunk {chunk} holds only samples before the stream's first keyframe")
        decoder: _ChunkDecoder = VideoDecoder(
            packets.mp4(self.reader.streams[self.stream].codec), device=str(self.device), seek_mode="exact", num_ffmpeg_threads=0
        )
        self._open[chunk] = (packets.first + packets.lead, decoder)
        while len(self._open) > self.max_open:
            self._open.popitem(last=False)
        return self._open[chunk]

    def get_frames_at(self, indices: Sequence[int]) -> UInt8[Tensor, "n 3 h w"]:
        """In the order asked, repeats allowed.

        Raises:
            IndexError: If an index is outside the stream.
        """
        wanted: Int64[ndarray, " n"] = np.asarray(indices, dtype=np.int64)
        count: int = len(self.reader.streams[self.stream].t_ns)
        if wanted.size and (wanted.min() < 0 or wanted.max() >= count):
            raise IndexError(f"{self.reader.streams[self.stream].entity}: sample indices must be in [0, {count})")
        if self.device.type == "cuda":
            # NVDEC creates and advances a decoder only on a thread whose CUDA context is current; a fresh worker has none.
            from simplecv.video_io import bind_cuda_device

            bind_cuda_device(str(self.device))
        unique: Int64[ndarray, " u"] = np.unique(wanted)
        chunk_of: Int64[ndarray, " u"] = np.searchsorted(self.reader.starts[:, self.stream], unique, side="right") - 1
        parts: list[Tensor] = []
        for chunk in np.unique(chunk_of).tolist():
            frame_zero, decoder = self._decoder(int(chunk))
            local: list[int] = (unique[chunk_of == chunk] - frame_zero).tolist()
            if local[0] < 0:
                raise ValueError(f"{self.index.entity}: sample {int(unique[chunk_of == chunk][0])} is before the stream's first keyframe")
            parts.append(decoder.get_frames_at(local).data)
            # torchcodec decodes on the CPU by itself when NVDEC cannot (no libnvcuvid, an unsupported stream): never silently.
            if decoder.cpu_fallback and not self._fallback_said:
                self._fallback_said = True
                print(f"simplecv: {self.index.entity} decodes on the CPU although {self.device} was asked for: {decoder.cpu_fallback}", flush=True)
        if not parts:
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8, device=self.device)
        # colsfm's usual request, sorted within one chunk, needs no copy.
        frames: Tensor = parts[0] if len(parts) == 1 else torch.cat(parts)
        if np.array_equal(unique, wanted):
            return frames
        return frames[torch.from_numpy(np.searchsorted(unique, wanted)).to(frames.device)]

    def get_frame_at(self, index: int) -> UInt8[Tensor, "3 h w"]:
        """One frame; a run of single reads in sample order decodes each chunk once."""
        return self.get_frames_at([index])[0]

    def close(self) -> None:
        """Release the open chunk decoders and their MP4s."""
        self._open.clear()


def open_decoders(
    dataset: DatasetEntry, segment_id: str, entities: Sequence[str], timeline: str, device: torch.device, *, chunk_s: float = CHUNK_S
) -> tuple[ChunkedVideoDecoder, ...]:
    """Index the streams and open one decoder each over a shared reader, so one query serves every stream's chunk."""
    reader: CatalogVideoReader = CatalogVideoReader(
        dataset, segment_id, read_video_index(dataset, segment_id, entities, timeline), timeline, chunk_s=chunk_s
    )
    return tuple(ChunkedVideoDecoder(reader, position, device) for position in range(len(reader.streams)))


def relay_video(reader: CatalogVideoReader, targets: Sequence[str], first: int, last: int, *, recording: rr.RecordingStream | None = None) -> int:
    """Re-log each stream's stored samples at its target over ``[first, last]``, never decoded, a chunk at a time.

    Each stream's window opens on its keyframe at or before ``first`` (the stream's first keyframe when none is): a
    decoder cannot start mid-GOP. Index values keep the timeline's kind (duration, timestamp or sequence).

    Returns:
        Samples logged, over every stream.
    """
    ranges: list[tuple[int, int]] = []
    for target, stream in zip(targets, reader.streams, strict=True):
        keyframe_at: Int64[ndarray, " k"] = np.flatnonzero(stream.keyframes)
        anchors: Int64[ndarray, " a"] = keyframe_at[stream.t_ns[keyframe_at] <= first]
        ranges.append((int(anchors[-1]) if len(anchors) else int(keyframe_at[0]), int(np.searchsorted(stream.t_ns, last, side="right"))))
        rr.log(target, rr.VideoStream(codec=video_codec(stream.codec)), static=True, recording=recording)
    wanted: list[int] = [position for position, (start, stop) in enumerate(ranges) if stop > start]
    if not wanted:
        return 0
    chunks: range = range(
        min(reader.chunk_of(position, ranges[position][0]) for position in wanted),
        max(reader.chunk_of(position, ranges[position][1] - 1) for position in wanted) + 1,
    )
    kind: str = (
        "duration" if pa.types.is_duration(reader.timeline_type) else "timestamp" if pa.types.is_timestamp(reader.timeline_type) else "sequence"
    )
    logged: int = 0
    for chunk in iter_chunks(reader, chunks):
        for target, stream, packets, (start, stop) in zip(targets, reader.streams, chunk.streams, ranges, strict=True):
            lo: int = max(start, packets.first) - packets.first
            hi: int = min(stop, packets.first + len(packets.samples)) - packets.first
            if hi <= lo:
                continue
            times: Shaped[ndarray, " m"] = stream.times[packets.first + lo : packets.first + hi]
            rr.send_columns(
                target,
                indexes=[rr.TimeColumn(reader.timeline, **{kind: times})],
                columns=rr.VideoStream.columns(sample=packets.samples[lo:hi], is_keyframe=packets.keyframes[lo:hi]),
                recording=recording,
            )
            logged += hi - lo
    return logged


__all__ = (
    "CatalogVideoReader",
    "ChunkedVideoDecoder",
    "SharedChunks",
    "StreamIndex",
    "StreamPackets",
    "VideoChunk",
    "chunk_starts",
    "iter_chunks",
    "open_decoders",
    "read_video_index",
    "relay_video",
    "select_decode_device",
    "time_window",
)
