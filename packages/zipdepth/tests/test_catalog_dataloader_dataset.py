"""Live equivalence proof for the Rerun-0.37 ZipDepth dataloader path."""

import os
from itertools import zip_longest

import numpy as np
import pytest
import torch
from jaxtyping import Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import NoShuffle

from zipdepth.catalog.builders import CudaSampleBuilder
from zipdepth.catalog.dataloader_dataset import RerunPromptDepthDataset, _ExactTimestampSampleIndex
from zipdepth.catalog.dataset import CatalogPromptDepthDataset
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, load_promptda_catalog
from zipdepth.catalog.targets import AugmentPolicy


def test_exact_timestamp_index_preserves_irregular_duration_values() -> None:
    """Resolve global sample positions to the stored packet timestamps, not a fixed grid."""
    first_timestamps: Int64[ndarray, "n"] = np.asarray([3, 11], dtype=np.int64)
    second_timestamps: Int64[ndarray, "n"] = np.asarray([100], dtype=np.int64)
    sample_index: _ExactTimestampSampleIndex = _ExactTimestampSampleIndex(
        {"first": first_timestamps, "second": second_timestamps}
    )

    assert sample_index.total_samples == 3
    assert sample_index.global_to_local(0)[1] == np.timedelta64(3, "ns")
    assert sample_index.global_to_local(1)[1] == np.timedelta64(11, "ns")
    assert sample_index.global_to_local(2)[1] == np.timedelta64(100, "ns")


def test_rerun_dataset_rejects_an_empty_segment_selection_before_querying() -> None:
    """Do not let an empty catalog filter widen into an accidental full-dataset query."""
    dataset_entry: DatasetEntry = object.__new__(DatasetEntry)

    with pytest.raises(ValueError, match="segment_ids must not be empty"):
        RerunPromptDepthDataset(
            dataset_entry,
            [],
            {},
            device=torch.device("cuda"),
            builder_factory=lambda: CudaSampleBuilder((192, 256), AugmentPolicy(), 0.0, torch.device("cuda")),
        )


@pytest.mark.skipif(
    os.environ.get("ZIPDEPTH_RUN_CATALOG_EQUIVALENCE") != "1",
    reason="requires the local ARKitScenes catalog and a CUDA decoder",
)
def test_rerun_dataloader_matches_every_sample_from_one_segment() -> None:
    """Yield the same frames and bit-identical CUDA-built tensors in natural order."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    catalog: PromptDACatalog = load_promptda_catalog(DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME)
    segment_id: str = catalog.segment_ids[0]
    device: torch.device = torch.device("cuda")
    policy: AugmentPolicy = AugmentPolicy()

    current: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
        catalog.dataset_entry,
        [segment_id],
        catalog.row_by_id,
        device=device,
        builder_factory=lambda: CudaSampleBuilder((192, 256), policy, 0.0, device, target_mode="metric"),
        shuffle_buffer_size=1,
        seed=0,
        frame_stride=1,
        num_producers=1,
        prefetch_samples=8,
        load_confidence=True,
    )
    idiomatic: RerunPromptDepthDataset = RerunPromptDepthDataset(
        catalog.dataset_entry,
        [segment_id],
        catalog.row_by_id,
        device=device,
        builder_factory=lambda: CudaSampleBuilder((192, 256), policy, 0.0, device, target_mode="metric"),
        shuffle_strategy=NoShuffle(),
        seed=0,
        frame_stride=1,
        fetch_block_size=128,
        build_batch_size=8,
        load_confidence=True,
    )

    sentinel: object = object()
    sample_count: int = 0
    current_sample: dict[str, torch.Tensor] | object
    idiomatic_sample: dict[str, torch.Tensor] | object
    for current_sample, idiomatic_sample in zip_longest(current, idiomatic, fillvalue=sentinel):
        assert current_sample is not sentinel
        assert idiomatic_sample is not sentinel
        assert isinstance(current_sample, dict)
        assert isinstance(idiomatic_sample, dict)
        assert current_sample.keys() == idiomatic_sample.keys()
        key: str
        for key in current_sample:
            assert current_sample[key].dtype == idiomatic_sample[key].dtype
            assert current_sample[key].shape == idiomatic_sample[key].shape
            assert torch.equal(current_sample[key], idiomatic_sample[key]), f"sample {sample_count}, field {key}"
        sample_count += 1

    assert sample_count > 0
