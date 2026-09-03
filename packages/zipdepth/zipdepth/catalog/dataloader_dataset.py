"""Rerun-0.37 dataloader path for exact chosen-frame ZipDepth samples.

The stock temporal sample index is fixed-rate, while the PromptDA layer is an
irregular subset of real ``video_time`` packet timestamps. This module queries
those timestamps once, then lets ``RerunIterableDataset`` own block fetching,
shuffling, DDP partitioning, field alignment, and batch decoder dispatch.
"""

from collections import defaultdict
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import TypeAlias

import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from arkitscenes_download.ingest.depth import decode_depth_png, inflate_depth_png_rows
from arkitscenes_download.ingest.paths import DEPTH_PROMPTDA, TIMELINE, VIDEO_WIDE
from jaxtyping import Int32, Int64, Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import (
    BlockShuffle,
    ColumnDecoder,
    DataSource,
    DecodeRequest,
    Field,
    FieldBatch,
    FixedRateSampling,
    IndexValue,
    NumericDecoder,
    RerunIterableDataset,
    SampleIndex,
    SegmentMetadata,
    ShuffleStrategy,
    Yuv420Frame,
)
from simplecv.rerun_dataloader import RECOMMENDED_FETCH_BLOCK_SIZE, SegmentNvdecDecoder
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from zipdepth.catalog.builders import CudaSampleBuilder
from zipdepth.catalog.dataset import CONFIDENCE_COLUMN, NATIVE_VIDEO_FPS, PROMPT_BLOB_COLUMN, PROMPTDA_BLOB_COLUMN
from zipdepth.catalog.png import unfilter_up_cuda_batch
from zipdepth.catalog.segments import SegmentRow
from zipdepth.catalog.stats import BuilderStats, CatalogDatasetStats

RawValue: TypeAlias = Tensor | Yuv420Frame | None
_EXACT_INDEX_SEGMENTS_PER_QUERY: int = 8
"""Segments per chosen-timestamp query; the measured catalog scan-cost knee."""
_EXACT_INDEX_QUERY_THREADS: int = 4
"""Concurrent timestamp queries; the local catalog showed no gain beyond four."""


@dataclass(frozen=True, slots=True)
class ExactIndexQueryStats:
    """One-time exact PromptDA timestamp query measurements."""

    plan_s: float
    """Seconds spent constructing the lazy query plan."""
    materialize_s: float
    """Seconds spent in the terminal Arrow materialization."""
    arrow_bytes: int
    """Bytes in the returned timestamp/segment-id Arrow table."""
    round_trips: int = 1
    """Terminal catalog materializations performed by this query."""


@dataclass(frozen=True, slots=True)
class _SampleMetadata:
    """Natural-order identity needed by the existing augmentation seed policy."""

    segment_seed_index: int
    """Position of the segment in the caller's unsharded input list."""
    sample_index: int
    """Position of the strided chosen timestamp within its segment."""
    quarter_turns: int
    """Counter-clockwise turns from stored orientation to landscape."""


class _ExactTimestampSampleIndex(SampleIndex):
    """SampleIndex whose local positions resolve to irregular duration values."""

    def __init__(self, timestamps_by_segment: dict[str, Int64[ndarray, "n"]]) -> None:
        """Build segment offsets over exact nanosecond timestamps."""
        segments: list[SegmentMetadata] = []
        segment_id: str
        timestamps_ns: Int64[ndarray, "n"]
        for segment_id, timestamps_ns in timestamps_by_segment.items():
            if timestamps_ns.size == 0:
                continue
            segments.append(
                SegmentMetadata(
                    segment_id=segment_id,
                    index_start=int(timestamps_ns[0]),
                    index_end=int(timestamps_ns[-1]),
                    num_samples=int(timestamps_ns.size),
                )
            )
        # A non-None step preserves duration typing in the stock query builders.
        # ``resolve_local_index`` below supplies the real, irregular values.
        super().__init__(segments, ns_per_sample=1, ns_dtype="timedelta64[ns]")
        self._timestamps_by_segment: dict[str, Int64[ndarray, "n"]] = timestamps_by_segment

    def resolve_local_index(self, seg: SegmentMetadata, pos: int) -> IndexValue:
        """Return the exact chosen packet timestamp at one local position."""
        timestamps_ns: Int64[ndarray, "n"] = self._timestamps_by_segment[seg.segment_id]
        if not 0 <= pos < timestamps_ns.size:
            raise IndexError(f"segment {seg.segment_id!r} sample {pos} is out of range")
        return np.timedelta64(int(timestamps_ns[pos]), "ns")


class _ExactTimestampRerunDataset(RerunIterableDataset):
    """Rerun iterable with the exact PromptDA index installed after setup."""

    def __init__(
        self,
        source: DataSource,
        fields: dict[str, Field],
        sample_index: _ExactTimestampSampleIndex,
        *,
        fetch_block_size: int,
        shuffle_strategy: ShuffleStrategy,
    ) -> None:
        """Build the stock connection/query pipeline and replace its fixed grid."""
        super().__init__(
            source,
            index=TIMELINE,
            fields=fields,
            # Required by 0.37 for duration timelines. The public constructor has
            # no exact-row temporal strategy, so the resulting grid is replaced.
            timeline_sampling=FixedRateSampling(rate_hz=10.0),
            fetch_block_size=fetch_block_size,
            shuffle_strategy=shuffle_strategy,
            decode_threads=1,
        )
        self._sample_index = sample_index


def _component_views(column: pa.Array, column_name: str) -> list[UInt8[ndarray, "n_bytes"] | None]:
    """Expose one zero-copy uint8 component instance per Arrow row.

    Args:
        column: One combined Rerun ``list<list<uint8>>`` component array.
        column_name: Fully-qualified name used in validation errors.

    Returns:
        A zero-copy byte view per populated row and None for a null row.
    """
    if not isinstance(column, pa.ListArray):
        raise ValueError(f"{column_name}: expected list rows, got {column.type}")
    outer_offsets: Shaped[ndarray, "n_rows_plus_one"] = column.offsets.to_numpy()
    inner: pa.Array = column.values
    if not isinstance(inner, pa.ListArray):
        raise ValueError(f"{column_name}: expected list component instances, got {inner.type}")
    inner_offsets: Shaped[ndarray, "n_instances_plus_one"] = inner.offsets.to_numpy()
    data: pa.Array = inner.values
    if not pa.types.is_uint8(data.type):
        raise ValueError(f"{column_name}: expected uint8 component values, got {data.type}")
    if data.null_count != 0:
        raise ValueError(f"{column_name}: component values must not be null")
    data_buffer: pa.Buffer | None = data.buffers()[1]
    if data_buffer is None:
        raise ValueError(f"{column_name}: component values have no Arrow data buffer")
    all_bytes: UInt8[ndarray, "all_bytes"] = np.frombuffer(
        data_buffer,
        dtype=np.uint8,
        offset=data.offset,
        count=len(data),
    )
    null_rows: list[bool] = column.is_null().to_pylist()
    views: list[UInt8[ndarray, "n_bytes"] | None] = []
    row: int
    for row in range(len(column)):
        if null_rows[row]:
            views.append(None)
            continue
        instance_start: int = int(outer_offsets[row])
        instance_stop: int = int(outer_offsets[row + 1])
        if instance_stop - instance_start != 1:
            raise ValueError(f"{column_name}: each row must contain exactly one component instance")
        if inner[instance_start].as_py() is None:
            views.append(None)
            continue
        byte_start: int = int(inner_offsets[instance_start]) - data.offset
        byte_stop: int = int(inner_offsets[instance_start + 1]) - data.offset
        views.append(all_bytes[byte_start:byte_stop])
    return views


class CudaDepthPngDecoder(ColumnDecoder[Tensor]):
    """Decode every requested depth PNG in a FieldBatch with batched CUDA unfiltering."""

    def __init__(
        self,
        column_name: str,
        device: torch.device,
        metadata_by_timestamp: dict[tuple[str, int], _SampleMetadata] | None = None,
        cuda_tile_size: int = 32,
        inflate_threads: int = 8,
    ) -> None:
        """Bind one depth column and optional target-sample metadata."""
        if device.type != "cuda":
            raise ValueError("CudaDepthPngDecoder requires a CUDA device")
        if cuda_tile_size <= 0 or inflate_threads <= 0:
            raise ValueError("cuda_tile_size and inflate_threads must be positive")
        self._column_name: str = column_name
        self._device: torch.device = device
        self._cuda_tile_size: int = cuda_tile_size
        self._inflate_threads: int = inflate_threads
        self._metadata_by_timestamp: dict[tuple[str, int], _SampleMetadata] | None = metadata_by_timestamp
        self._metadata_by_tensor: dict[int, _SampleMetadata] = {}
        self.decode_s: float = 0.0
        """Synchronized Arrow extraction, PNG inflation, transfer, and CUDA unfilter time."""
        self.png_fallbacks: int = 0
        """Rows that required the general PNG decoder."""

    def reset_metadata(self) -> None:
        """Drop metadata retained for samples abandoned by a stopped iterator."""
        self._metadata_by_tensor.clear()

    def take_metadata(self, target_mm: Tensor) -> _SampleMetadata:
        """Remove and return the identity associated with one decoded target."""
        try:
            return self._metadata_by_tensor.pop(id(target_mm))
        except KeyError:
            raise RuntimeError("decoded target metadata was already consumed or is absent") from None

    def decode(self, batch: FieldBatch, requests: Sequence[DecodeRequest]) -> Sequence[Tensor | None]:
        """Decode all resolved output rows with one CUDA pass per stored shape."""
        started: float = perf_counter()
        views: list[UInt8[ndarray, "n_bytes"] | None] = _component_views(batch.column, self._column_name)
        decoded: list[Tensor | None] = [None] * len(requests)
        filtered_groups: dict[tuple[int, int], list[tuple[int, UInt8[ndarray, "h row_bytes"]]]] = defaultdict(list)
        fallback_groups: dict[tuple[int, int], list[tuple[int, Int32[ndarray, "h w"]]]] = defaultdict(list)

        requested_blobs: list[UInt8[ndarray, "n_bytes"] | None] = []
        request: DecodeRequest
        for request in requests:
            if len(request.output_row_indices) != 1:
                raise ValueError("ZipDepth depth fields must be unwindowed")
            blob: UInt8[ndarray, "n_bytes"] | None = views[request.output_row_indices[0]]
            requested_blobs.append(blob)
        present_blobs: list[UInt8[ndarray, "n_bytes"]] = [blob for blob in requested_blobs if blob is not None]
        with ThreadPoolExecutor(max_workers=self._inflate_threads, thread_name_prefix="depth-png-inflate") as pool:
            inflated_present: list[tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None] = list(
                pool.map(inflate_depth_png_rows, present_blobs)
            )
        inflated_iterator: Iterator[tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None] = iter(inflated_present)

        position: int
        blob: UInt8[ndarray, "n_bytes"] | None
        for position, blob in enumerate(requested_blobs):
            if blob is None:
                continue
            inflated: tuple[UInt8[ndarray, "h row_bytes"], tuple[int, int]] | None = next(inflated_iterator)
            if inflated is None:
                depth_mm: Int32[ndarray, "h w"] = decode_depth_png(blob).astype(np.int32)
                fallback_groups[depth_mm.shape].append((position, depth_mm))
                self.png_fallbacks += 1
            else:
                filtered_rows: UInt8[ndarray, "h row_bytes"] = inflated[0]
                filtered_groups[filtered_rows.shape].append((position, filtered_rows))

        group: list[tuple[int, UInt8[ndarray, "h row_bytes"]]]
        for group in filtered_groups.values():
            tile_start: int
            for tile_start in range(0, len(group), self._cuda_tile_size):
                tile: list[tuple[int, UInt8[ndarray, "h row_bytes"]]] = group[
                    tile_start : tile_start + self._cuda_tile_size
                ]
                filtered_batch: UInt8[ndarray, "n h row_bytes"] = np.stack([item[1] for item in tile])
                filtered_tensor: UInt8[Tensor, "n h row_bytes"] = torch.from_numpy(filtered_batch).to(
                    device=self._device,
                    non_blocking=True,
                )
                depths_mm: Int32[Tensor, "n h w"] = unfilter_up_cuda_batch(filtered_tensor)
                row: int
                for row, (position, _) in enumerate(tile):
                    decoded[position] = depths_mm[row]

        fallback_group: list[tuple[int, Int32[ndarray, "h w"]]]
        for fallback_group in fallback_groups.values():
            for tile_start in range(0, len(fallback_group), self._cuda_tile_size):
                fallback_tile: list[tuple[int, Int32[ndarray, "h w"]]] = fallback_group[
                    tile_start : tile_start + self._cuda_tile_size
                ]
                fallback_depths_mm: Int32[ndarray, "n h w"] = np.stack([item[1] for item in fallback_tile])
                fallback_tensors: Int32[Tensor, "n h w"] = torch.from_numpy(fallback_depths_mm).to(
                    device=self._device,
                    non_blocking=True,
                )
                row = 0
                for row, (position, _) in enumerate(fallback_tile):
                    decoded[position] = fallback_tensors[row]
        torch.cuda.synchronize(self._device)
        self.decode_s += perf_counter() - started

        if self._metadata_by_timestamp is not None:
            decoded_depth: Tensor | None
            for request, decoded_depth in zip(requests, decoded, strict=True):
                if decoded_depth is None:
                    continue
                timestamp_key: tuple[str, int] = (request.segment_id, int(request.index_value))
                try:
                    self._metadata_by_tensor[id(decoded_depth)] = self._metadata_by_timestamp[timestamp_key]
                except KeyError:
                    raise RuntimeError(f"no chosen-row metadata for {timestamp_key}") from None
        return decoded

    def __repr__(self) -> str:
        return f"CudaDepthPngDecoder(column_name={self._column_name!r})"


def _query_exact_timestamps(dataset: DatasetEntry, segment_ids: list[str]) -> pa.Table:
    """Materialize chosen PromptDA entity timestamps for one segment batch."""
    return (
        dataset.filter_segments(segment_ids)
        .filter_contents([f"/{DEPTH_PROMPTDA}"])
        .reader(index=TIMELINE)
        .select("rerun_segment_id", TIMELINE)
        .to_arrow_table()
    )


def _load_exact_sample_index(
    dataset: DatasetEntry,
    segment_ids: list[str],
    row_by_id: dict[str, SegmentRow],
    frame_stride: int,
) -> tuple[_ExactTimestampSampleIndex, dict[tuple[str, int], _SampleMetadata], ExactIndexQueryStats]:
    """Collect exact PromptDA timestamps through bounded cross-segment queries."""
    plan_started: float = perf_counter()
    segment_batches: list[list[str]] = [
        segment_ids[start : start + _EXACT_INDEX_SEGMENTS_PER_QUERY]
        for start in range(0, len(segment_ids), _EXACT_INDEX_SEGMENTS_PER_QUERY)
    ]
    plan_s: float = perf_counter() - plan_started
    materialize_started: float = perf_counter()
    with ThreadPoolExecutor(max_workers=_EXACT_INDEX_QUERY_THREADS, thread_name_prefix="exact-index-query") as pool:
        tables: list[pa.Table] = list(pool.map(partial(_query_exact_timestamps, dataset), segment_batches))
    table: pa.Table = pa.concat_tables(tables)
    materialize_s: float = perf_counter() - materialize_started

    segment_values: list[str] = table.column("rerun_segment_id").to_pylist()
    timestamps_ns: Int64[ndarray, "n_rows"] = table.column(TIMELINE).to_numpy(zero_copy_only=False).astype(np.int64)
    grouped: dict[str, list[int]] = {segment_id: [] for segment_id in segment_ids}
    segment_value: str
    timestamp_ns: np.int64
    for segment_value, timestamp_ns in zip(segment_values, timestamps_ns, strict=True):
        if segment_value in grouped:
            grouped[segment_value].append(int(timestamp_ns))

    exact_by_segment: dict[str, Int64[ndarray, "n"]] = {}
    metadata_by_timestamp: dict[tuple[str, int], _SampleMetadata] = {}
    segment_seed_index: int
    segment_id: str
    for segment_seed_index, segment_id in enumerate(segment_ids):
        selected_ns: Int64[ndarray, "n"] = np.asarray(sorted(grouped[segment_id]), dtype=np.int64)[::frame_stride]
        if selected_ns.size == 0:
            continue
        if np.any(selected_ns[1:] <= selected_ns[:-1]):
            raise ValueError(f"segment {segment_id}: PromptDA timestamps must be strictly increasing")
        exact_by_segment[segment_id] = selected_ns
        segment_row: SegmentRow = row_by_id[segment_id]
        quarter_turns: int = 0 if segment_row.orientation == "landscape" else (-segment_row.orientation_quarter_turns_ccw) % 4
        sample_index: int
        timestamp: np.int64
        for sample_index, timestamp in enumerate(selected_ns):
            metadata_by_timestamp[(segment_id, int(timestamp))] = _SampleMetadata(
                segment_seed_index=segment_seed_index,
                sample_index=sample_index,
                quarter_turns=quarter_turns,
            )
    if not exact_by_segment:
        raise RuntimeError("selected segments contain no PromptDA target rows")
    return (
        _ExactTimestampSampleIndex(exact_by_segment),
        metadata_by_timestamp,
        ExactIndexQueryStats(
            plan_s=plan_s,
            materialize_s=materialize_s,
            arrow_bytes=table.nbytes,
            round_trips=len(segment_batches),
        ),
    )


class RerunPromptDepthDataset(IterableDataset[dict[str, Tensor]]):
    """Yield existing-contract ZipDepth samples through Rerun's 0.37 loader."""

    def __init__(
        self,
        dataset_entry: DatasetEntry,
        segment_ids: list[str],
        row_by_id: dict[str, SegmentRow],
        *,
        device: torch.device,
        builder_factory: Callable[[], CudaSampleBuilder],
        shuffle_strategy: ShuffleStrategy | None = None,
        seed: int = 0,
        frame_stride: int = 1,
        fetch_block_size: int = RECOMMENDED_FETCH_BLOCK_SIZE,
        build_batch_size: int = 8,
        load_confidence: bool = True,
    ) -> None:
        """Build the exact timestamp index, batch decoders, and Rerun iterable."""
        super().__init__()
        if not segment_ids:
            raise ValueError("segment_ids must not be empty")
        if device.type != "cuda":
            raise ValueError("RerunPromptDepthDataset requires a CUDA device")
        if frame_stride <= 0 or fetch_block_size <= 0 or build_batch_size <= 0:
            raise ValueError("frame_stride, fetch_block_size, and build_batch_size must be positive")
        self._device: torch.device = device
        self._segment_ids: list[str] = list(segment_ids)
        self._seed: int = seed
        self._epoch: int = 0
        self._build_batch_size: int = build_batch_size
        self._load_confidence: bool = load_confidence
        self._pipeline_wait_s: float = 0.0
        setup_started: float = perf_counter()
        exact_result: tuple[
            _ExactTimestampSampleIndex,
            dict[tuple[str, int], _SampleMetadata],
            ExactIndexQueryStats,
        ] = _load_exact_sample_index(dataset_entry, self._segment_ids, row_by_id, frame_stride)
        sample_index: _ExactTimestampSampleIndex = exact_result[0]
        metadata_by_timestamp: dict[tuple[str, int], _SampleMetadata] = exact_result[1]
        self.index_query_stats: ExactIndexQueryStats = exact_result[2]
        """Measurements for the one cross-segment chosen-timestamp query."""

        self._video_decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(
            dataset_entry,
            VIDEO_WIDE,
            TIMELINE,
            device,
            NATIVE_VIDEO_FPS,
        )
        self._target_decoder: CudaDepthPngDecoder = CudaDepthPngDecoder(
            PROMPTDA_BLOB_COLUMN,
            device,
            metadata_by_timestamp,
        )
        self._prompt_decoder: CudaDepthPngDecoder = CudaDepthPngDecoder(PROMPT_BLOB_COLUMN, device)
        fields: dict[str, Field] = {
            "video": Field(f"/{VIDEO_WIDE}:VideoStream:sample", decode=self._video_decoder),
            "target": Field(PROMPTDA_BLOB_COLUMN, decode=self._target_decoder),
            "prompt": Field(PROMPT_BLOB_COLUMN, decode=self._prompt_decoder),
        }
        if load_confidence:
            fields["confidence"] = Field(CONFIDENCE_COLUMN, decode=NumericDecoder())
        self._samples: _ExactTimestampRerunDataset = _ExactTimestampRerunDataset(
            DataSource(dataset=dataset_entry, segments=self._segment_ids),
            fields,
            sample_index,
            fetch_block_size=fetch_block_size,
            shuffle_strategy=shuffle_strategy if shuffle_strategy is not None else BlockShuffle(),
        )
        self._builder: CudaSampleBuilder = builder_factory()
        self.setup_s: float = perf_counter() - setup_started
        """Total synchronous construction time, including both index queries."""

    @property
    def stats(self) -> CatalogDatasetStats:
        """Return cumulative critical-path and decoder/build counters."""
        builder_stats: BuilderStats = self._builder.stats
        decoder_s: float = self._target_decoder.decode_s + self._prompt_decoder.decode_s
        video_query_s: float = self._video_decoder.query_seconds
        video_decode_s: float = self._video_decoder.decode_seconds
        uncovered_fetch_s: float = max(self._pipeline_wait_s - decoder_s - video_query_s - video_decode_s, 0.0)
        return CatalogDatasetStats(
            segment_query=self.index_query_stats.materialize_s + video_query_s + uncovered_fetch_s,
            video_decode=video_decode_s,
            blob_decode=decoder_s,
            augment=builder_stats.augment,
            samples_built=builder_stats.samples_built,
            png_fallbacks=self._target_decoder.png_fallbacks + self._prompt_decoder.png_fallbacks,
            skipped_frames=0,
            skipped_flat_frames=builder_stats.skipped_flat_frames,
        )

    @property
    def skipped_frames(self) -> int:
        """Return video rows skipped by recoverable decoder failures."""
        return self.stats.skipped_frames

    @property
    def skipped_flat_frames(self) -> int:
        """Return rows rejected by the target depth-span filter."""
        return self.stats.skipped_flat_frames

    def set_epoch(self, epoch: int) -> None:
        """Seed Rerun shuffling and the existing sample augmentation policy."""
        self._epoch = epoch
        self._samples.set_epoch(self._seed + epoch)

    def __len__(self) -> int:
        """Return the unsharded exact chosen-sample count."""
        return len(self._samples)

    def _build(self, raw_batch: list[dict[str, RawValue]], rank: int) -> Generator[dict[str, Tensor], None, None]:
        """Build one raw Rerun batch with the existing CUDA sample contract."""
        frames: list[UInt8[Tensor, "3 h w"]] = []
        targets_mm: list[Int32[Tensor, "h w"]] = []
        prompts_mm: list[Int32[Tensor, "prompt_h prompt_w"]] = []
        confidences: list[UInt8[Tensor, "confidence_n"] | None] = []
        quarter_turns: list[int] = []
        sample_seeds: list[int] = []
        raw: dict[str, RawValue]
        for raw in raw_batch:
            frame: RawValue = raw["video"]
            target: RawValue = raw["target"]
            prompt: RawValue = raw["prompt"]
            if not isinstance(frame, Tensor) or not isinstance(target, Tensor) or not isinstance(prompt, Tensor):
                continue
            if frame.dtype != torch.uint8 or target.dtype != torch.int32 or prompt.dtype != torch.int32:
                raise TypeError("Rerun ZipDepth decoders returned an unexpected tensor dtype")
            decoded_frame: UInt8[Tensor, "3 h w"] = frame
            target_mm: Int32[Tensor, "h w"] = target
            prompt_mm: Int32[Tensor, "prompt_h prompt_w"] = prompt
            raw_confidence: RawValue = raw.get("confidence")
            if raw_confidence is not None and not isinstance(raw_confidence, Tensor):
                raise TypeError("Rerun confidence decoder returned a non-tensor value")
            confidence: Tensor | None = raw_confidence
            if confidence is not None and confidence.dtype != torch.uint8:
                raise TypeError("Rerun confidence decoder returned a non-uint8 tensor")
            confidence_tensor: UInt8[Tensor, "confidence_n"] | None = confidence
            metadata: _SampleMetadata = self._target_decoder.take_metadata(target_mm)
            frames.append(decoded_frame)
            targets_mm.append(target_mm)
            prompts_mm.append(prompt_mm)
            confidences.append(confidence_tensor)
            quarter_turns.append(metadata.quarter_turns)
            sample_seeds.append(
                (
                    self._seed
                    + self._epoch * 1_000_003
                    + rank * 10_000_019
                    + metadata.segment_seed_index * 100_000_007
                    + metadata.sample_index
                )
                % (2**63 - 1)
            )
        built: list[dict[str, Tensor] | None] = self._builder.build_decoded_batch(
            frames,
            targets_mm,
            prompts_mm,
            confidences,
            quarter_turns,
            sample_seeds,
        )
        sample: dict[str, Tensor] | None
        for sample in built:
            if sample is not None:
                yield sample

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Decode, build in small batches, and yield one finite Rerun pass."""
        if get_worker_info() is not None:
            raise RuntimeError("RerunPromptDepthDataset requires DataLoader(num_workers=0) for CUDA decoding")
        distributed: bool = dist.is_available() and dist.is_initialized()
        rank: int = dist.get_rank() if distributed else 0
        raw_iterator: Iterator[dict[str, RawValue]] = iter(self._samples)
        self._target_decoder.reset_metadata()
        raw_batch: list[dict[str, RawValue]] = []
        try:
            while True:
                started: float = perf_counter()
                try:
                    raw: dict[str, RawValue] = next(raw_iterator)
                except StopIteration:
                    self._pipeline_wait_s += perf_counter() - started
                    break
                self._pipeline_wait_s += perf_counter() - started
                raw_batch.append(raw)
                if len(raw_batch) == self._build_batch_size:
                    yield from self._build(raw_batch, rank)
                    raw_batch.clear()
            if raw_batch:
                yield from self._build(raw_batch, rank)
        finally:
            if isinstance(raw_iterator, Generator):
                raw_iterator.close()
            self._target_decoder.reset_metadata()
