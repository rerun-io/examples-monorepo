"""Live equivalence proof for the Rerun-0.37 ZipDepth dataloader path."""

import os
from hashlib import sha256
from typing import TypeAlias

import numpy as np
import pytest
import torch
from jaxtyping import Int64, Shaped
from numpy import ndarray

pytestmark = pytest.mark.integration
pytest.importorskip("rerun.catalog", reason="Rerun catalog dependencies live in the zipdepth catalog lane")
pytest.importorskip("rerun.experimental.dataloader", reason="rerun.experimental.dataloader is required by this test module")
pytest.importorskip("torchcodec", reason="torchcodec is required by this test module")
pytest.importorskip("depth_png", reason="depth_png is required by this test module")

pytest.importorskip("arkitscenes_download", reason="ARKitScenes catalog dependencies live in the zipdepth catalog lane")
import depth_png  # noqa: E402
import pyarrow as pa  # noqa: E402
from rerun.catalog import DatasetEntry  # noqa: E402
from rerun.experimental.dataloader import NoShuffle  # noqa: E402
from rerun.experimental.dataloader.decoders._base import DecodeRequest, FieldBatch  # noqa: E402

from zipdepth.catalog.builders import CudaSampleBuilder  # noqa: E402
from zipdepth.catalog.dataloader_dataset import CudaDepthPngDecoder, RerunPromptDepthDataset, _ExactTimestampSampleIndex  # noqa: E402
from zipdepth.catalog.dataset import CatalogPromptDepthDataset  # noqa: E402
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, load_promptda_catalog  # noqa: E402
from zipdepth.catalog.targets import AugmentPolicy  # noqa: E402

TensorDigest: TypeAlias = tuple[torch.dtype, tuple[int, ...], bytes]
SampleDigest: TypeAlias = dict[str, TensorDigest]


def _sample_digest(sample: dict[str, torch.Tensor]) -> SampleDigest:
    """Describe every tensor with its dtype, shape, and exact contiguous bytes."""
    digest_by_field: SampleDigest = {}
    field: str
    value: Shaped[torch.Tensor, "..."]
    for field, value in sample.items():
        shape: tuple[int, ...] = tuple(value.shape)
        digest: bytes = sha256(value.detach().cpu().contiguous().numpy().tobytes()).digest()
        digest_by_field[field] = (value.dtype, shape, digest)
    return digest_by_field


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

    # Production selects one loader. Run them sequentially so beartype's lazy
    # PEP 526 checker cache is never initialized concurrently by both pipelines.
    current_digests: list[SampleDigest] = [_sample_digest(sample) for sample in current]
    idiomatic_digests: list[SampleDigest] = [_sample_digest(sample) for sample in idiomatic]

    assert current_digests
    assert idiomatic_digests == current_digests


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the depth column decoder copies to CUDA")
def test_depth_column_decoder_decodes_every_row_into_cuda_int32() -> None:
    """Rows of two stored shapes, a null row and one-frame tiles: each row decodes to its own millimetres."""
    rng: np.random.Generator = np.random.default_rng(5)
    depths: list[ndarray | None] = [
        rng.integers(0, 65536, size=(30, 40), dtype=np.uint16),
        None,
        rng.integers(0, 65536, size=(10, 20), dtype=np.uint16),
        rng.integers(0, 65536, size=(30, 40), dtype=np.uint16),
    ]
    blobs: list[list[bytes] | None] = [None if depth is None else [depth_png.encode(depth)] for depth in depths]
    column: pa.Array = pa.array(blobs, type=pa.list_(pa.list_(pa.uint8())))
    requests: list[DecodeRequest] = [
        DecodeRequest(sample_position=row, segment_id="segment", index_value=row, decode_row_indices=(row,), output_row_indices=(row,), starts_at_keyframe=True)
        for row in range(len(depths))
    ]
    decoder: CudaDepthPngDecoder = CudaDepthPngDecoder("depth:blob", torch.device("cuda"), cuda_tile_size=1, decode_threads=2)

    decoded: list[torch.Tensor | None] = list(decoder.decode(FieldBatch(column=column), requests))

    assert decoded[1] is None
    for depth, result in zip(depths, decoded, strict=True):
        if depth is not None:
            assert result is not None and result.dtype == torch.int32 and result.device.type == "cuda"
            np.testing.assert_array_equal(result.cpu().numpy(), depth.astype(np.int32))
