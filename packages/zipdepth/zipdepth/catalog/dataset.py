"""Stream chosen-frame PromptDA targets from a Rerun catalog."""

from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from queue import Empty, Queue
from random import Random
from threading import Thread, current_thread
from time import perf_counter
from typing import Generic, TypeVar
from warnings import warn

import numpy as np
import pyarrow as pa
import torch
from arkitscenes_download.ingest.paths import (
    CONFIDENCE,
    DEPTH,
    DEPTH_PROMPTDA,
    DEPTH_ULTRAWIDE_RECT,
    RGB_ULTRAWIDE_RECT,
    TIMELINE,
    VIDEO_WIDE,
)
from arkitscenes_download.ingest.timestamps import match_exact_timestamps, table_timestamps
from beartype.roar import BeartypeException
from datafusion import col
from jaxtyping import Bool, Int64, Shaped, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rerun_dataloader import open_segment_decoder
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchcodec.decoders import VideoDecoder
from torchvision.io import ImageReadMode, decode_jpeg

from zipdepth.catalog.builders import SampleBuilder
from zipdepth.catalog.segments import SegmentRow
from zipdepth.catalog.stats import CatalogDatasetStats, total
from zipdepth.catalog.threaded import iter_threaded
from zipdepth.catalog.ultrawide import ULTRAWIDE_LAYER, CameraSelection, interleave, subsample_indices

PROMPTDA_BLOB_COLUMN: str = f"/{DEPTH_PROMPTDA}:EncodedDepthImage:blob"
"""Catalog column containing uint16 PromptDA PNG blobs."""
PROMPT_BLOB_COLUMN: str = f"/{DEPTH}:EncodedDepthImage:blob"
"""Catalog column containing uint16 ARKit LiDAR prompt PNG blobs."""
CONFIDENCE_COLUMN: str = f"/{CONFIDENCE}:SegmentationImage:buffer"
"""Catalog column containing flattened uint8 ARKit depth confidence."""
ULTRAWIDE_DEPTH_BLOB_COLUMN: str = f"/{DEPTH_ULTRAWIDE_RECT}:EncodedDepthImage:blob"
"""Catalog column containing uint16 millimetre rectified-ultrawide depth PNG blobs."""
ULTRAWIDE_RGB_BLOB_COLUMN: str = f"/{RGB_ULTRAWIDE_RECT}:EncodedImage:blob"
"""Catalog column containing rectified-ultrawide JPEG blobs."""
NATIVE_VIDEO_FPS: int = 60
"""Nominal frame rate used when wrapping the segment's encoded AV1 packets."""
ULTRAWIDE_SEED_OFFSET: int = 500_000_011
"""Separates the ultrawide augmentation seed stream from the wide one within a segment."""
ItemT = TypeVar("ItemT")


class ShuffleBuffer(Generic[ItemT]):  # noqa: UP046 — beartype does not support PEP 695 generics
    """Seeded streaming shuffle with bounded memory."""

    def __init__(self, size: int, seed: int) -> None:
        """Create an empty buffer."""
        if size <= 0:
            raise ValueError("shuffle buffer size must be positive")
        self._size: int = size
        self._rng: Random = Random(seed)
        self._samples: list[ItemT] = []

    def push(self, sample: ItemT) -> ItemT | None:
        """Add one sample and return a random eviction when the buffer is full."""
        if len(self._samples) < self._size:
            self._samples.append(sample)
            return None
        index: int = self._rng.randrange(self._size)
        evicted: ItemT = self._samples[index]
        self._samples[index] = sample
        return evicted

    def flush(self) -> Generator[ItemT, None, None]:
        """Yield all remaining samples in seeded random order and empty the buffer."""
        remaining: list[ItemT] = list(self._samples)
        self._samples.clear()
        self._rng.shuffle(remaining)
        yield from remaining


@dataclass(slots=True)
class _ProducerSlot:
    """Long-lived builder state owned by at most one producer thread."""

    builder: SampleBuilder
    """Sample builder reused across dataset passes."""
    stats: CatalogDatasetStats
    """Dataset counters accumulated across passes."""
    owner: Thread | None = None
    """Producer thread currently driving this builder; None when idle."""


@dataclass(slots=True)
class _WideSegment:
    """Wide-camera inputs for one segment after its query and decoder open."""

    quarter_turns: int
    """Counter-clockwise turns that bring the stored frame to landscape."""
    segment_seed: int
    """Base augmentation seed for this segment, pass, and rank."""
    target_blobs: list[UInt8[ndarray, "target_n_bytes"]]
    """Chosen-frame PromptDA depth PNG blobs."""
    prompt_blobs: list[UInt8[ndarray, "prompt_n_bytes"]]
    """Latest-at ARKit LiDAR depth PNG blobs aligned with the chosen frames."""
    confidences: list[UInt8[ndarray, "confidence_n_values"] | None]
    """Flattened ARKit confidence per chosen frame, or None when not fetched."""
    packet_indices: Int64[ndarray, "n_chosen"]
    """Video packet index per chosen frame."""
    decoder: VideoDecoder
    """Whole-segment video decoder owned by the calling producer."""


@dataclass(slots=True)
class _UltrawideSegment:
    """Rectified ultrawide inputs for one segment after its catalog query."""

    quarter_turns: int
    """Counter-clockwise turns that bring the stored frame to landscape."""
    segment_seed: int
    """Base augmentation seed for this segment, pass, and rank."""
    rgb_blobs: list[UInt8[ndarray, "rgb_n_bytes"]]
    """Rectified ultrawide JPEG blobs."""
    target_blobs: list[UInt8[ndarray, "target_n_bytes"]]
    """Raycast ultrawide depth PNG blobs aligned with the JPEG blobs."""
    prompt_blobs: list[UInt8[ndarray, "prompt_n_bytes"]]
    """Latest-at ARKit LiDAR depth PNG blobs aligned with the chosen frames."""
    confidences: list[UInt8[ndarray, "confidence_n_values"] | None]
    """Flattened ARKit confidence per chosen frame, or None when not fetched."""


def component_bytes_views(column: pa.ChunkedArray, column_name: str) -> list[UInt8[ndarray, "n_bytes"]]:
    """Return byte views into each Arrow chunk for one uint8 component instance per row.

    Args:
        column: Rerun ``list<list<uint8>>`` component column. Each outer row
            must contain exactly one component instance.
        column_name: Fully-qualified column name used in validation errors.

    Returns:
        One zero-copy uint8 view into its source chunk's values buffer per row.
    """
    views: list[UInt8[ndarray, "n_bytes"]] = []
    chunk: pa.Array
    for chunk in column.chunks:
        if not isinstance(chunk, pa.ListArray):
            raise ValueError(f"{column_name}: expected list rows, got {chunk.type}")
        if chunk.offset != 0:
            raise ValueError(f"{column_name}: outer chunk offset must be zero, got {chunk.offset}")
        if chunk.null_count != 0:
            raise ValueError(f"{column_name}: outer rows must not be null")
        outer_offsets: Shaped[ndarray, "n_rows_plus_one"] = chunk.offsets.to_numpy()
        if not bool(np.all(np.diff(outer_offsets) == 1)):
            raise ValueError(f"{column_name}: each row must contain exactly one component instance")

        inner: pa.Array = chunk.values
        if not isinstance(inner, pa.ListArray):
            raise ValueError(f"{column_name}: expected list component instances, got {inner.type}")
        if inner.offset != 0:
            raise ValueError(f"{column_name}: inner array offset must be zero, got {inner.offset}")
        if inner.null_count != 0:
            raise ValueError(f"{column_name}: component instances must not be null")
        data: pa.Array = inner.values
        if not pa.types.is_uint8(data.type):
            raise ValueError(f"{column_name}: expected uint8 component values, got {data.type}")
        if data.null_count != 0:
            raise ValueError(f"{column_name}: component values must not be null")
        data_buffer: pa.Buffer | None = data.buffers()[1]
        if data_buffer is None:
            raise ValueError(f"{column_name}: component values have no Arrow data buffer")
        all_bytes_n: UInt8[ndarray, "all_bytes"] = np.frombuffer(data_buffer, dtype=np.uint8, offset=data.offset, count=len(data))
        inner_offsets: Shaped[ndarray, "n_instances_plus_one"] = inner.offsets.to_numpy()
        row: int
        for row in range(len(chunk)):
            instance: int = int(outer_offsets[row])
            start: int = int(inner_offsets[instance]) - data.offset
            end: int = int(inner_offsets[instance + 1]) - data.offset
            views.append(all_bytes_n[start:end])
    return views


def _drop_null_payload_rows(table: pa.Table, payload_columns: list[str]) -> pa.Table:
    """Keep only rows whose every payload column carries a value.

    Args:
        table: Query result whose rows may be missing a payload column.
        payload_columns: Columns that must be non-null on a usable row.

    Returns:
        The input table when nothing is null, otherwise a filtered copy.
    """
    if all(table.column(column_name).null_count == 0 for column_name in payload_columns):
        return table
    keep_n: Bool[ndarray, "n_rows"] = np.ones(table.num_rows, dtype=bool)
    column_name: str
    for column_name in payload_columns:
        keep_n &= table.column(column_name).is_valid().to_numpy(zero_copy_only=False).astype(bool)
    return table.filter(pa.array(keep_n))


def promptda_blob_views(column: pa.ChunkedArray) -> list[UInt8[ndarray, "n_bytes"]]:
    """Return zero-copy PromptDA blob views, retaining the established public helper."""
    return component_bytes_views(column, PROMPTDA_BLOB_COLUMN)


class CatalogPromptDepthDataset(IterableDataset[dict[str, Tensor]]):
    """Yield augmented ZipDepth samples from catalog PromptDA rows.

    Use this dataset only with ``DataLoader(num_workers=0)`` so CUDA decoding
    stays in the training process. Producer threads claim whole segments, and
    each producer owns one sample builder and one video decoder at a time. A
    bounded queue limits decoded-sample memory. One producer uses the same path
    and preserves segment order for evaluation and fixed probes.
    """

    def __init__(
        self,
        dataset_entry: DatasetEntry,
        segment_ids: list[str],
        row_by_id: dict[str, SegmentRow],
        *,
        device: torch.device,
        builder_factory: Callable[[], SampleBuilder],
        shuffle_buffer_size: int = 256,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        frame_stride: int = 1,
        num_producers: int = 6,
        prefetch_samples: int = 256,
        load_confidence: bool = True,
        cameras: CameraSelection = "wide",
    ) -> None:
        """Configure catalog streaming without opening the server or decoder.

        Args:
            dataset_entry: Rerun catalog dataset containing ARKitScenes segments.
            segment_ids: Segment identifiers eligible for this dataset.
            row_by_id: Segment-table rows keyed by identifier, used for orientation.
            device: Device used by each whole-segment video decoder.
            builder_factory: Construct one sample builder per producer thread.
            shuffle_buffer_size: Maximum samples held for bounded random shuffling.
            seed: Base seed for segment and sample ordering.
            rank: Distributed rank used for segment sharding.
            world_size: Number of distributed ranks sharing the segment list.
            frame_stride: Keep every Nth chosen PromptDA row within each segment.
            num_producers: Whole-segment producer threads. One preserves order.
            prefetch_samples: Maximum completed samples queued by producers.
            load_confidence: Fetch the ARKit confidence column. Training ignores it --
                the student takes the raw prompt and the model range-gates internally --
                so leaving it out drops a column and ~48 KB/frame off every segment
                query. Evaluation keeps it for the affine-reference and prompt-upsample
                diagnostics, which do consume ``prompt_valid``.
            cameras: Cameras streamed per segment. ``wide`` is the historical
                lane. ``ultrawide`` streams the rectified ultrawide chosen frames.
                ``both`` streams the wide frames plus a fresh seeded ultrawide
                subsample of equal size, interleaved, so the shuffle buffer mixes
                them into roughly balanced batches. The builder must accept the
                ultrawide camera when this selects it.

        Raises:
            ValueError: If sizes, camera selection, or distributed shard settings
                are invalid.
        """
        if cameras not in ("wide", "ultrawide", "both"):
            raise ValueError(f"unknown camera selection {cameras!r}")
        if shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} must be in [0, {world_size})")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if num_producers <= 0:
            raise ValueError("num_producers must be positive")
        if prefetch_samples <= 0:
            raise ValueError("prefetch_samples must be positive")
        self._dataset_entry: DatasetEntry = dataset_entry
        self._segment_ids: list[str] = list(segment_ids)
        self._row_by_id: dict[str, SegmentRow] = row_by_id
        self._device: torch.device = device
        self._builder_factory: Callable[[], SampleBuilder] = builder_factory
        self._load_confidence: bool = load_confidence
        self._cameras: CameraSelection = cameras
        self._missing_ultrawide_reported: bool = False
        self._shuffle_buffer_size: int = shuffle_buffer_size
        self._seed: int = seed
        self._rank: int = rank
        self._world_size: int = world_size
        self._epoch: int = 0
        self._frame_stride: int = frame_stride
        self._num_producers: int = num_producers
        self._prefetch_samples: int = prefetch_samples
        self._segment_seed_index: dict[str, int] = {segment_id: index for index, segment_id in enumerate(self._segment_ids)}
        self._slots: list[_ProducerSlot] = []
        """One builder, its counters, and its current owner per producer slot."""

    @property
    def stats(self) -> CatalogDatasetStats:
        """Return a snapshot summed across every producer slot's dataset and builder counters."""
        return total([slot.stats + CatalogDatasetStats.from_builder(slot.builder.stats) for slot in self._slots])

    @property
    def skipped_frames(self) -> int:
        """Return the number of video frames skipped after decode errors."""
        return self.stats.skipped_frames

    @property
    def skipped_flat_frames(self) -> int:
        """Return the number of frames rejected by the depth-span filter."""
        return self.stats.skipped_flat_frames

    @property
    def skipped_low_valid_frames(self) -> int:
        """Return the number of ultrawide frames rejected for sparse valid depth."""
        return self.stats.skipped_low_valid_frames

    def set_epoch(self, epoch: int) -> None:
        """Set the pass index used to seed segment and sample shuffling."""
        self._epoch = epoch

    def _segment_seed(self, segment_id: str) -> int:
        """Return the augmentation seed base for one segment, pass, and rank."""
        return (
            self._seed
            + self._epoch * 1_000_003
            + self._rank * 10_000_019
            + self._segment_seed_index[segment_id] * 100_000_007
        )

    def _require_ordered_rows(self, table: pa.Table, segment_id: str) -> Shaped[ndarray, "n_rows"]:
        """Return the reader's index column, rejecting any out-of-order response."""
        all_times_n: Shaped[ndarray, "n_rows"] = table_timestamps(table)
        if np.any(all_times_n[1:] < all_times_n[:-1]):
            raise ValueError(f"segment {segment_id}: reader returned rows out of {TIMELINE} order; the no-sort fast path assumes index order")
        return all_times_n

    def _prepare_wide(self, segment_id: str, stats: CatalogDatasetStats) -> _WideSegment | None:
        """Query one segment's chosen PromptDA rows and open its video decoder."""
        started: float = perf_counter()
        contents: list[str] = [f"/{DEPTH_PROMPTDA}", f"/{DEPTH}"]
        columns: list[str] = [TIMELINE, PROMPTDA_BLOB_COLUMN, PROMPT_BLOB_COLUMN]
        if self._load_confidence:
            contents.append(f"/{CONFIDENCE}")
            columns.append(CONFIDENCE_COLUMN)
        # The depth query is independent of the video query inside open_segment_decoder;
        # both block in native code with the GIL released, so overlapping them takes
        # ~30% off the per-segment stall. The depth query goes to the worker thread (pure
        # datafusion/gRPC); the decoder stays on this thread because NVDEC creation needs
        # the producer's CUDA context. Neither query sorts client-side — the reader's row
        # order is trusted; the rationale lives in simplecv's open_segment_decoder, and
        # the guard below fails loudly if the server ordering contract ever breaks.
        def _query_depth_table() -> pa.Table:
            return (
                self._dataset_entry.filter_segments(segment_id)
                .filter_contents(contents)
                .reader(index=TIMELINE, fill_latest_at=True)
                .filter(col(f'"{PROMPTDA_BLOB_COLUMN}"').is_not_null())
                .select(*columns)
                .to_arrow_table()
            )

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="segment-depth-query") as pool:
            table_future: Future[pa.Table] = pool.submit(_query_depth_table)
            decoder_result: tuple[Shaped[ndarray, "n_packets"], list[bytes], list[bool], VideoDecoder] = open_segment_decoder(
                self._dataset_entry,
                segment_id,
                VIDEO_WIDE,
                TIMELINE,
                self._device,
                NATIVE_VIDEO_FPS,
            )
            table: pa.Table = table_future.result()
        stats.segment_query += perf_counter() - started
        if table.num_rows == 0:
            return None

        all_times_n: Shaped[ndarray, "n_rows"] = self._require_ordered_rows(table, segment_id)
        chosen_times_n: Shaped[ndarray, "n_chosen"] = all_times_n[:: self._frame_stride]
        target_blobs: list[UInt8[ndarray, "target_n_bytes"]] = component_bytes_views(table.column(PROMPTDA_BLOB_COLUMN), PROMPTDA_BLOB_COLUMN)[
            :: self._frame_stride
        ]
        prompt_blobs: list[UInt8[ndarray, "prompt_n_bytes"]] = component_bytes_views(table.column(PROMPT_BLOB_COLUMN), PROMPT_BLOB_COLUMN)[
            :: self._frame_stride
        ]
        confidences: list[UInt8[ndarray, "confidence_n_values"] | None]
        if self._load_confidence:
            confidences = list(component_bytes_views(table.column(CONFIDENCE_COLUMN), CONFIDENCE_COLUMN)[:: self._frame_stride])
        else:
            # Not fetched: the builder stands in an all-trusted mask once it knows the
            # decoded prompt shape. The model range-gates the prompt itself.
            confidences = [None] * len(prompt_blobs)

        packet_times_n: Shaped[ndarray, "n_packets"] = decoder_result[0]
        decoder: VideoDecoder = decoder_result[3]
        del decoder_result
        packet_indices_n: Int64[ndarray, "n_chosen"] = match_exact_timestamps(packet_times_n, chosen_times_n)
        return _WideSegment(
            quarter_turns=self._quarter_turns(segment_id),
            segment_seed=self._segment_seed(segment_id),
            target_blobs=target_blobs,
            prompt_blobs=prompt_blobs,
            confidences=confidences,
            packet_indices=packet_indices_n,
            decoder=decoder,
        )

    def _quarter_turns(self, segment_id: str) -> int:
        """Return the counter-clockwise turns that bring stored frames to landscape."""
        segment_row: SegmentRow = self._row_by_id[segment_id]
        return 0 if segment_row.orientation == "landscape" else (-segment_row.orientation_quarter_turns_ccw) % 4

    def _build_wide_samples(
        self,
        segment_id: str,
        prepared: _WideSegment,
        builder: SampleBuilder,
        stats: CatalogDatasetStats,
    ) -> Generator[dict[str, Tensor], None, None]:
        """Decode video frames and build one wide sample per chosen frame."""
        target_blob_bytes: UInt8[ndarray, "target_n_bytes"]
        prompt_blob_bytes: UInt8[ndarray, "prompt_n_bytes"]
        confidence: UInt8[ndarray, "confidence_n_values"] | None
        packet_index: np.int64
        first_decode_error_reported: bool = False
        sample_inputs = zip(prepared.target_blobs, prepared.prompt_blobs, prepared.confidences, prepared.packet_indices, strict=True)
        for sample_index, (target_blob_bytes, prompt_blob_bytes, confidence, packet_index) in enumerate(sample_inputs):
            started: float = perf_counter()
            try:
                frame_chw: UInt8[Tensor, "3 stored_h stored_w"] = prepared.decoder.get_frame_at(int(packet_index)).data
                stats.video_decode += perf_counter() - started
            except BeartypeException:
                raise
            except Exception as error:
                stats.video_decode += perf_counter() - started
                stats.skipped_frames += 1
                if not first_decode_error_reported:
                    warn(f"{segment_id}: first video decode exception: {error!r}", RuntimeWarning, stacklevel=2)
                    first_decode_error_reported = True
                continue
            sample: dict[str, Tensor] | None = builder(
                frame_chw,
                target_blob_bytes,
                prompt_blob_bytes,
                confidence,
                prepared.quarter_turns,
                (prepared.segment_seed + sample_index) % (2**63 - 1),
            )
            if sample is not None:
                yield sample

    def _prepare_ultrawide(self, segment_id: str, stats: CatalogDatasetStats, frame_limit: int | None) -> _UltrawideSegment | None:
        """Query one segment's rectified-ultrawide chosen rows and their latest-at prompts.

        The ultrawide rows are logged at their own 10 Hz selection, disjoint from
        the wide chosen frames, so the ARKit prompt is taken latest-at-or-before
        each ultrawide timestamp. The ARKit stream runs at 60 Hz, which bounds the
        staleness at one frame interval.

        Args:
            segment_id: Segment to query.
            stats: Producer counters that accumulate the query time.
            frame_limit: Keep a seeded random subset of this size; None keeps all
                chosen frames.

        Returns:
            The prepared inputs, or None when the segment has no ultrawide rows.
        """
        started: float = perf_counter()
        contents: list[str] = [f"/{DEPTH_ULTRAWIDE_RECT}", f"/{RGB_ULTRAWIDE_RECT}", f"/{DEPTH}"]
        columns: list[str] = [TIMELINE, ULTRAWIDE_DEPTH_BLOB_COLUMN, ULTRAWIDE_RGB_BLOB_COLUMN, PROMPT_BLOB_COLUMN]
        if self._load_confidence:
            contents.append(f"/{CONFIDENCE}")
            columns.append(CONFIDENCE_COLUMN)
        # Every payload column must be present on a kept row. The ultrawide track
        # has its own drop_leading and its own 10 Hz selection, so a chosen frame
        # can precede the first ARKit lowres row; latest-at cannot fill what has
        # not been logged yet, and component_bytes_views rejects a null outer row.
        # A null row carries no payload bytes, so dropping these client-side costs
        # the same transfer as a server-side filter and still yields the count.
        chosen: pa.Table = (
            self._dataset_entry.filter_segments(segment_id)
            .filter_contents(contents)
            .reader(index=TIMELINE, fill_latest_at=True)
            .filter(col(f'"{ULTRAWIDE_DEPTH_BLOB_COLUMN}"').is_not_null())
            .select(*columns)
            .to_arrow_table()
        )
        stats.segment_query += perf_counter() - started
        table: pa.Table = _drop_null_payload_rows(chosen, columns[1:])
        stats.skipped_missing_payload_frames += chosen.num_rows - table.num_rows
        if table.num_rows == 0:
            return None
        self._require_ordered_rows(table, segment_id)

        rgb_blobs: list[UInt8[ndarray, "rgb_n_bytes"]] = component_bytes_views(
            table.column(ULTRAWIDE_RGB_BLOB_COLUMN), ULTRAWIDE_RGB_BLOB_COLUMN
        )[:: self._frame_stride]
        target_blobs: list[UInt8[ndarray, "target_n_bytes"]] = component_bytes_views(
            table.column(ULTRAWIDE_DEPTH_BLOB_COLUMN), ULTRAWIDE_DEPTH_BLOB_COLUMN
        )[:: self._frame_stride]
        prompt_blobs: list[UInt8[ndarray, "prompt_n_bytes"]] = component_bytes_views(table.column(PROMPT_BLOB_COLUMN), PROMPT_BLOB_COLUMN)[
            :: self._frame_stride
        ]
        confidences: list[UInt8[ndarray, "confidence_n_values"] | None]
        if self._load_confidence:
            confidences = list(component_bytes_views(table.column(CONFIDENCE_COLUMN), CONFIDENCE_COLUMN)[:: self._frame_stride])
        else:
            confidences = [None] * len(prompt_blobs)

        segment_seed: int = self._segment_seed(segment_id)
        if frame_limit is not None:
            kept: list[int] = subsample_indices(len(target_blobs), frame_limit, segment_seed)
            rgb_blobs = [rgb_blobs[index] for index in kept]
            target_blobs = [target_blobs[index] for index in kept]
            prompt_blobs = [prompt_blobs[index] for index in kept]
            confidences = [confidences[index] for index in kept]
        return _UltrawideSegment(
            quarter_turns=self._quarter_turns(segment_id),
            segment_seed=segment_seed,
            rgb_blobs=rgb_blobs,
            target_blobs=target_blobs,
            prompt_blobs=prompt_blobs,
            confidences=confidences,
        )

    def _build_ultrawide_samples(
        self,
        prepared: _UltrawideSegment,
        builder: SampleBuilder,
        stats: CatalogDatasetStats,
    ) -> Generator[dict[str, Tensor], None, None]:
        """Decode rectified JPEG frames and build one ultrawide sample per chosen frame."""
        rgb_blob_bytes: UInt8[ndarray, "rgb_n_bytes"]
        target_blob_bytes: UInt8[ndarray, "target_n_bytes"]
        prompt_blob_bytes: UInt8[ndarray, "prompt_n_bytes"]
        confidence: UInt8[ndarray, "confidence_n_values"] | None
        sample_inputs = zip(prepared.rgb_blobs, prepared.target_blobs, prepared.prompt_blobs, prepared.confidences, strict=True)
        for sample_index, (rgb_blob_bytes, target_blob_bytes, prompt_blob_bytes, confidence) in enumerate(sample_inputs):
            started: float = perf_counter()
            # The Arrow view is read-only, and decode_jpeg needs an owned uint8
            # tensor. Copying one ~50 KB blob costs far less than the decode.
            encoded_n: UInt8[Tensor, "rgb_n_bytes"] = torch.frombuffer(bytearray(rgb_blob_bytes), dtype=torch.uint8)
            decoded: Tensor | list[Tensor] = decode_jpeg(encoded_n, mode=ImageReadMode.RGB)
            if not isinstance(decoded, Tensor):
                raise TypeError("decode_jpeg returned a batch for one encoded ultrawide frame")
            frame_chw: UInt8[Tensor, "3 stored_h stored_w"] = decoded.to(device=self._device, non_blocking=True)
            stats.jpeg_decode += perf_counter() - started
            sample: dict[str, Tensor] | None = builder(
                frame_chw,
                target_blob_bytes,
                prompt_blob_bytes,
                confidence,
                prepared.quarter_turns,
                (prepared.segment_seed + ULTRAWIDE_SEED_OFFSET + sample_index) % (2**63 - 1),
                camera="ultrawide",
            )
            if sample is not None:
                yield sample

    def _iter_segment(
        self,
        segment_id: str,
        builder: SampleBuilder,
        stats: CatalogDatasetStats,
    ) -> Generator[dict[str, Tensor], None, None]:
        """Stream the configured cameras for one segment, interleaved when both are on."""
        wants_wide: bool = self._cameras in ("wide", "both")
        wants_ultrawide: bool = self._cameras in ("ultrawide", "both")
        if wants_ultrawide and ULTRAWIDE_LAYER not in self._row_by_id[segment_id].layer_names:
            if not self._missing_ultrawide_reported:
                self._missing_ultrawide_reported = True
                warn(
                    f"skipping ultrawide frames for segments without the {ULTRAWIDE_LAYER!r} layer, starting with {segment_id}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            wants_ultrawide = False
        wide_prepared: _WideSegment | None = self._prepare_wide(segment_id, stats) if wants_wide else None
        # Match the ultrawide count to the wide count so a mixed pass is about
        # half of each camera after the shuffle buffer. Without wide frames there
        # is nothing to balance against, and streaming the whole ultrawide track
        # (up to 1,840 frames) would silently skew the mix, so skip the segment.
        frame_limit: int | None = None
        if self._cameras == "both":
            if wide_prepared is None:
                warn(f"{segment_id}: no wide chosen frames, skipping its ultrawide frames too", RuntimeWarning, stacklevel=2)
                wants_ultrawide = False
            else:
                frame_limit = len(wide_prepared.target_blobs)
        ultrawide_prepared: _UltrawideSegment | None = (
            self._prepare_ultrawide(segment_id, stats, frame_limit) if wants_ultrawide else None
        )
        streams: list[Iterable[dict[str, Tensor]]] = []
        if wide_prepared is not None:
            streams.append(self._build_wide_samples(segment_id, wide_prepared, builder, stats))
        if ultrawide_prepared is not None:
            streams.append(self._build_ultrawide_samples(ultrawide_prepared, builder, stats))
        if len(streams) == 1:
            yield from streams[0]
        elif streams:
            yield from interleave(streams)

    def _iter_parallel(self, segment_ids: list[str]) -> Generator[dict[str, Tensor], None, None]:
        """Yield complete samples from independent whole-segment worker threads."""
        for index, slot in enumerate(self._slots):
            if slot.owner is not None and slot.owner.is_alive():
                raise RuntimeError(
                    f"producer thread {slot.owner.name!r} from a previous pass still owns builder slot {index}; refusing to reuse it"
                )
        if not self._slots:
            # One long-lived builder (and, on CUDA, one stream) per producer thread, never more
            # threads than segments: reused by every pass so counters stay cumulative and nothing leaks.
            self._slots = [
                _ProducerSlot(self._builder_factory(), CatalogDatasetStats())
                for _ in range(min(self._num_producers, len(segment_ids)))
            ]
        segment_queue: Queue[str] = Queue()
        segment_id: str
        for segment_id in segment_ids:
            segment_queue.put(segment_id)

        def produce(slot: _ProducerSlot) -> Generator[dict[str, Tensor], None, None]:
            """Claim whole segments with one builder until the queue is empty."""
            slot.owner = current_thread()
            try:
                while True:
                    try:
                        claimed_segment_id: str = segment_queue.get_nowait()
                    except Empty:
                        return
                    yield from self._iter_segment(claimed_segment_id, slot.builder, slot.stats)
            finally:
                slot.owner = None

        workers: list[Callable[[], Generator[dict[str, Tensor], None, None]]] = [partial(produce, slot) for slot in self._slots]
        yield from iter_threaded(workers, maxsize=self._prefetch_samples)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Decode, build, and bounded-shuffle this rank's segment shard."""
        if get_worker_info() is not None:
            raise RuntimeError("CatalogPromptDepthDataset requires DataLoader(num_workers=0) for its in-process CUDA decoder")

        segment_ids: list[str] = list(self._segment_ids[self._rank :: self._world_size])
        Random(self._seed + self._epoch).shuffle(segment_ids)
        shuffle_buffer: ShuffleBuffer[dict[str, Tensor]] = ShuffleBuffer(
            size=self._shuffle_buffer_size,
            seed=self._seed + self._epoch * self._world_size + self._rank,
        )
        parallel_samples: Generator[dict[str, Tensor], None, None] = self._iter_parallel(segment_ids)
        try:
            sample: dict[str, Tensor]
            for sample in parallel_samples:
                evicted: dict[str, Tensor] | None = shuffle_buffer.push(sample)
                if evicted is not None:
                    yield evicted
        finally:
            parallel_samples.close()
        yield from shuffle_buffer.flush()
