"""Behavioral tests for catalog pipeline counters."""

from dataclasses import fields

from zipdepth.catalog.stats import STAGE_FIELDS, BuilderStats, CatalogDatasetStats, total


def test_stage_fields_name_every_float_timer_of_the_dataset_stats() -> None:
    """Keep the instrumentation's stage-timer list in step with the dataclass."""
    float_fields: tuple[str, ...] = tuple(field.name for field in fields(CatalogDatasetStats) if field.type is float)

    assert float_fields == STAGE_FIELDS


def test_builder_stats_are_a_named_subset_of_the_dataset_stats() -> None:
    """Let a dataset snapshot absorb every builder counter by field name."""
    builder_names: set[str] = {field.name for field in fields(BuilderStats)}
    dataset_names: set[str] = {field.name for field in fields(CatalogDatasetStats)}

    assert builder_names == {"blob_decode", "augment", "samples_built", "png_fallbacks", "skipped_flat_frames"}
    assert builder_names <= dataset_names
    assert BuilderStats() == BuilderStats(blob_decode=0.0, augment=0.0, samples_built=0, png_fallbacks=0, skipped_flat_frames=0)


def test_total_sums_field_wise_without_mutating_inputs() -> None:
    """Fold any number of snapshots into one, leaving the operands untouched."""
    first: CatalogDatasetStats = CatalogDatasetStats(segment_query=1.0, samples_built=2)
    second: CatalogDatasetStats = CatalogDatasetStats(segment_query=0.5, video_decode=2.0, skipped_frames=3)

    combined: CatalogDatasetStats = total([first, second])

    assert combined == CatalogDatasetStats(segment_query=1.5, video_decode=2.0, samples_built=2, skipped_frames=3)
    assert first == CatalogDatasetStats(segment_query=1.0, samples_built=2)
    assert total([]) == CatalogDatasetStats()
