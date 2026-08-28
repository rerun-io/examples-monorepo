"""Behavioral and construction tests for catalog training data."""

from threading import Event, Thread
from typing import Any

import numpy as np
import pytest
import torch
from jaxtyping import UInt8, UInt16
from numpy import ndarray
from numpy.testing import assert_array_equal
from torch.utils.data import IterableDataset

pytest.importorskip("rerun.catalog", reason="Rerun catalog dependencies live in the zipdepth catalog lane")
pytest.importorskip("arkitscenes_download", reason="ARKitScenes catalog dependencies live in the zipdepth catalog lane")
pa = pytest.importorskip("pyarrow", reason="Arrow catalog dependencies live in the zipdepth catalog lane")

from arkitscenes_download.ingest.depth import encode_depth_png  # noqa: E402
from rerun.catalog import DatasetEntry  # noqa: E402

from zipdepth.apis.catalog_throughput import CatalogThroughputConfig  # noqa: E402
from zipdepth.catalog.builders import CpuSampleBuilder  # noqa: E402
from zipdepth.catalog.dataset import CatalogPromptDepthDataset, ShuffleBuffer, _ProducerSlot, promptda_blob_views  # noqa: E402
from zipdepth.catalog.stats import CatalogDatasetStats  # noqa: E402
from zipdepth.catalog.targets import build_eval_transform  # noqa: E402


def test_shuffle_buffer_is_seeded_and_preserves_every_sample() -> None:
    """Produce one deterministic streaming permutation without drops or duplicates."""

    def shuffled(seed: int) -> list[int]:
        """Shuffle one fixed range through the streaming buffer."""
        buffer: ShuffleBuffer[int] = ShuffleBuffer(size=3, seed=seed)
        output: list[int] = []
        value: int
        for value in range(10):
            evicted: int | None = buffer.push(value)
            if evicted is not None:
                output.append(evicted)
        output.extend(buffer.flush())
        return output

    first: list[int] = shuffled(17)
    second: list[int] = shuffled(17)

    assert first == second
    assert sorted(first) == list(range(10))


def test_catalog_dataset_constructs_without_contacting_server() -> None:
    """Store segment metadata and initialize mutable counters without catalog IO."""
    dataset_entry: DatasetEntry = object.__new__(DatasetEntry)
    row_by_id: dict[str, dict[str, Any]] = {
        "segment-a": {
            "rerun_segment_id": "segment-a",
            "property:capture:orientation": ["portrait"],
            "property:capture:orientation_quarter_turns_ccw": [1],
        }
    }

    dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
        dataset_entry,
        ["segment-a"],
        row_by_id,
        device=torch.device("cpu"),
        builder_factory=lambda: CpuSampleBuilder(build_eval_transform(8, 8), min_depth_span=0.0),
    )
    dataset.set_epoch(2)

    assert isinstance(dataset, IterableDataset)
    assert dataset.stats == CatalogDatasetStats()
    assert dataset.skipped_frames == 0


def test_iter_parallel_refuses_a_slot_owned_by_a_live_thread() -> None:
    """Reject builder reuse while a producer from the previous pass is still alive."""
    dataset_entry: DatasetEntry = object.__new__(DatasetEntry)
    row_by_id: dict[str, dict[str, Any]] = {
        "segment-a": {
            "rerun_segment_id": "segment-a",
            "property:capture:orientation": ["portrait"],
            "property:capture:orientation_quarter_turns_ccw": [1],
        }
    }
    builder: CpuSampleBuilder = CpuSampleBuilder(build_eval_transform(8, 8), min_depth_span=0.0)
    dataset: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
        dataset_entry,
        ["segment-a"],
        row_by_id,
        device=torch.device("cpu"),
        builder_factory=lambda: CpuSampleBuilder(build_eval_transform(8, 8), min_depth_span=0.0),
    )
    release: Event = Event()
    owner: Thread = Thread(target=release.wait, name="test-stale-zipdepth-producer")
    owner.start()
    dataset._slots = [_ProducerSlot(builder, CatalogDatasetStats(), owner=owner)]

    try:
        with pytest.raises(RuntimeError, match="still owns builder slot"):
            next(dataset._iter_parallel(["segment-a"]))
    finally:
        release.set()
        owner.join(timeout=2.0)


def test_cpu_sample_builder_decodes_and_reports_local_stats() -> None:
    """Build one CPU sample through the typed callable and retain lock-free counters."""
    rgb_chw: UInt8[torch.Tensor, "3 h w"] = torch.arange(3 * 6 * 8, dtype=torch.uint8).reshape(3, 6, 8)
    depth_mm_hw: UInt16[ndarray, "h w"] = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8) + 100
    blob_u8: UInt8[ndarray, "n"] = np.frombuffer(encode_depth_png(depth_mm_hw), dtype=np.uint8)
    builder: CpuSampleBuilder = CpuSampleBuilder(build_eval_transform(6, 8), min_depth_span=0.0)

    sample: dict[str, torch.Tensor] | None = builder(rgb_chw, blob_u8, quarter_turns=0, sample_seed=17)

    assert sample is not None
    assert sample["image"].shape == (3, 6, 8)
    assert builder.stats.samples_built == 1
    assert builder.stats.png_fallbacks == 0
    assert builder.stats.blob_decode > 0.0
    assert builder.stats.augment > 0.0


@pytest.mark.parametrize(("num_producers", "prefetch_samples"), [(0, 64), (1, 0)])
def test_catalog_dataset_rejects_invalid_producer_settings(num_producers: int, prefetch_samples: int) -> None:
    """Reject producer counts and queue capacities that cannot make progress."""
    dataset_entry: DatasetEntry = object.__new__(DatasetEntry)

    with pytest.raises(ValueError, match="positive"):
        CatalogPromptDepthDataset(
            dataset_entry,
            [],
            {},
            device=torch.device("cpu"),
            builder_factory=lambda: CpuSampleBuilder(build_eval_transform(8, 8), min_depth_span=0.0),
            num_producers=num_producers,
            prefetch_samples=prefetch_samples,
        )


def test_promptda_blob_views_extract_one_instance_per_row_without_python_lists() -> None:
    """Expose each list-of-list uint8 row as the same bytes as Arrow's Python conversion."""
    expected: list[list[list[int]]] = [[[1, 2, 3]], [[10, 20]], [[255, 0, 17, 4]]]
    column = pa.chunked_array(
        [pa.array(expected[:2], type=pa.list_(pa.list_(pa.uint8()))), pa.array(expected[2:], type=pa.list_(pa.list_(pa.uint8())))]
    )
    assert column.num_chunks == 2

    views = promptda_blob_views(column)

    assert len(views) == len(expected)
    for view, python_row in zip(views, column.to_pylist(), strict=True):
        assert not view.flags.owndata
        assert_array_equal(view, python_row[0])


def test_promptda_blob_views_rejects_sliced_outer_arrays() -> None:
    """Reject nonzero Arrow offsets before applying zero-based row arithmetic."""
    source = pa.array([[[1]], [[2, 3]]], type=pa.list_(pa.list_(pa.uint8())))
    column = pa.chunked_array([source]).slice(1, 1)

    with pytest.raises(ValueError, match="EncodedDepthImage:blob.*offset"):
        promptda_blob_views(column)


@pytest.mark.parametrize(
    "values",
    [
        [[[1]], [[2], [3]]],
        [[[1]], None],
        [[[1]], [None]],
    ],
)
def test_promptda_blob_views_rejects_invalid_component_instances(values: list[object]) -> None:
    """Reject multi-instance and null rows with a column-specific error."""
    column = pa.chunked_array([pa.array(values, type=pa.list_(pa.list_(pa.uint8())))])

    with pytest.raises(ValueError, match="EncodedDepthImage:blob"):
        promptda_blob_views(column)


def test_catalog_throughput_config_has_safe_smoke_defaults() -> None:
    """Keep the throughput probe small unless the caller requests more data."""
    config: CatalogThroughputConfig = CatalogThroughputConfig()

    assert config.dataset_name == "arkitscenes-v2"
    assert config.num_segments == 2
    assert config.batch_size == 8
    assert config.builder == "cuda"
    assert config.num_producers == 6
    assert config.prefetch_samples == 256
    assert config.segment_ids is None
