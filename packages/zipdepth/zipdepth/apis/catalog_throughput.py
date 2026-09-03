"""Measure catalog-to-batch throughput for ZipDepth training data."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from zipdepth.catalog.builders import CpuSampleBuilder, CudaSampleBuilder, SampleBuilder
from zipdepth.catalog.dataset import CatalogPromptDepthDataset
from zipdepth.catalog.segments import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, PromptDACatalog, load_promptda_catalog
from zipdepth.catalog.stats import CatalogDatasetStats
from zipdepth.catalog.targets import AugmentPolicy, build_train_transform


@dataclass(slots=True)
class CatalogThroughputConfig:
    """Configuration for the catalog throughput probe."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """URL of the local Rerun catalog server."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset containing ARKitScenes and PromptDA layers."""
    num_segments: int = 2
    """Number of leading PromptDA segments to use when ``segment_ids`` is absent."""
    segment_ids: list[str] | None = None
    """Explicit PromptDA segment identifiers; overrides ``num_segments`` when set."""
    height: int = 768
    """Augmented training image height."""
    width: int = 1024
    """Augmented training image width."""
    batch_size: int = 8
    """Samples per DataLoader batch."""
    builder: Literal["cuda", "cpu"] = "cuda"
    """Sample builder implementation used by the A/B throughput probe."""
    num_producers: int = 6
    """Whole-segment producer threads, each with its own video decoder."""
    prefetch_samples: int = 256
    """Maximum completed samples buffered across segment producers."""
    load_confidence: bool = True
    """Fetch the ARKit confidence column; training skips it."""
    min_depth_span: float = 1.25
    """Minimum valid-depth p95/p5 ratio; zero disables flat-frame filtering."""
    max_batches: int | None = None
    """Optional limit on yielded batches, including shuffle-buffer warmup cost."""


def main(config: CatalogThroughputConfig) -> None:
    """Read catalog samples and print overall and per-stage throughput.

    Args:
        config: Catalog, resize, batching, and stopping settings.

    Raises:
        RuntimeError: If no eligible segment or sample is available.
        ValueError: If numeric limits are not positive or an explicit segment lacks PromptDA.
    """
    if config.segment_ids is None and config.num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.num_producers <= 0 or config.prefetch_samples <= 0:
        raise ValueError("num_producers and prefetch_samples must be positive")
    if config.max_batches is not None and config.max_batches <= 0:
        raise ValueError("max_batches must be positive when set")

    catalog: PromptDACatalog = load_promptda_catalog(config.catalog_url, config.dataset_name)
    if config.segment_ids is None:
        segment_ids: list[str] = catalog.segment_ids[: config.num_segments]
    else:
        segment_ids = list(config.segment_ids)
        catalog.require_segments(segment_ids)
    if not segment_ids:
        raise RuntimeError(f"dataset {config.dataset_name!r} has no selected PromptDA segments")

    device: torch.device = torch.device("cuda")
    policy: AugmentPolicy = AugmentPolicy()
    builder_factory: Callable[[], SampleBuilder]
    if config.builder == "cuda":
        builder_factory = lambda: CudaSampleBuilder((config.height, config.width), policy, config.min_depth_span, device)  # noqa: E731
    else:
        builder_factory = lambda: CpuSampleBuilder(  # noqa: E731
            build_train_transform(config.height, config.width, policy), config.min_depth_span
        )
    samples: CatalogPromptDepthDataset = CatalogPromptDepthDataset(
        catalog.dataset_entry,
        segment_ids,
        catalog.row_by_id,
        device=device,
        builder_factory=builder_factory,
        num_producers=config.num_producers,
        prefetch_samples=config.prefetch_samples,
        load_confidence=config.load_confidence,
    )
    loader: DataLoader[dict[str, Tensor]] = DataLoader(samples, batch_size=config.batch_size, num_workers=0)
    started: float = perf_counter()
    frames: int = 0
    batches: int = 0
    batch: dict[str, Tensor]
    for batch in loader:
        if config.builder == "cuda":
            torch.cuda.synchronize()
        frames += int(batch["image"].shape[0])
        batches += 1
        if config.max_batches is not None and batches >= config.max_batches:
            break
    elapsed: float = perf_counter() - started
    if frames == 0 or samples.stats.samples_built == 0:
        raise RuntimeError("catalog dataset yielded no training samples")

    stats: CatalogDatasetStats = samples.stats
    built_frames: int = stats.samples_built
    decoded_frames: int = built_frames + stats.skipped_flat_frames
    video_attempts: int = decoded_frames + stats.skipped_frames
    print(
        f"config: builder={config.builder}, producers={config.num_producers}, "
        f"prefetch_samples={config.prefetch_samples}, batch_size={config.batch_size}, out={config.width}x{config.height}"
    )
    print(f"overall: {frames / elapsed:.2f} frames/s ({frames} yielded, {built_frames} built, {batches} batches, {elapsed:.2f}s)")
    print(f"segment_query: {stats.segment_query * 1000.0 / video_attempts:.3f} ms/frame")
    print(f"video_decode: {stats.video_decode * 1000.0 / video_attempts:.3f} ms/frame")
    print(f"blob_decode: {stats.blob_decode * 1000.0 / decoded_frames:.3f} ms/frame")
    print(f"augment: {stats.augment * 1000.0 / decoded_frames:.3f} ms/frame")
    print(f"png fallbacks: {stats.png_fallbacks}")
    print(f"skipped frames: {samples.skipped_frames}")
    print(f"skipped flat frames: {samples.skipped_flat_frames}")
