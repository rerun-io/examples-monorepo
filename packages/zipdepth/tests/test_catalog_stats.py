"""Behavioral tests for catalog pipeline counters."""

from dataclasses import fields

from zipdepth.catalog.stats import STAGE_FIELDS, BuilderStats, CatalogDatasetStats, total


def _every_field(scale: int) -> CatalogDatasetStats:
    """Return stats with every field set to a distinct multiple of ``scale``."""
    return CatalogDatasetStats(
        segment_query=1.0 * scale,
        video_decode=2.0 * scale,
        blob_decode=3.0 * scale,
        augment=4.0 * scale,
        samples_built=5 * scale,
        png_fallbacks=6 * scale,
        skipped_frames=7 * scale,
        skipped_flat_frames=8 * scale,
    )


def test_stage_fields_name_every_float_timer_of_the_dataset_stats() -> None:
    """Keep the instrumentation's stage-timer list in step with the dataclass."""
    float_fields: tuple[str, ...] = tuple(field.name for field in fields(CatalogDatasetStats) if field.type is float)

    assert float_fields == STAGE_FIELDS


def test_builder_stats_lift_into_the_dataset_schema_by_name() -> None:
    """Project builder-local counters onto the dataset counters; dataset-only fields stay zero."""
    builder_names: set[str] = {field.name for field in fields(BuilderStats)}
    builder_stats: BuilderStats = BuilderStats(blob_decode=1.5, augment=0.5, samples_built=3, png_fallbacks=1, skipped_flat_frames=2)

    lifted: CatalogDatasetStats = CatalogDatasetStats.from_builder(builder_stats)

    assert builder_names <= {field.name for field in fields(CatalogDatasetStats)}
    assert lifted == CatalogDatasetStats(blob_decode=1.5, augment=0.5, samples_built=3, png_fallbacks=1, skipped_flat_frames=2)


def test_total_sums_every_field_without_mutating_inputs() -> None:
    """Fold any number of snapshots field-wise into one, leaving the operands untouched."""
    first: CatalogDatasetStats = _every_field(1)
    second: CatalogDatasetStats = _every_field(10)

    combined: CatalogDatasetStats = total([first, second])

    assert all(getattr(first, field.name) != 0 for field in fields(CatalogDatasetStats)), "helper must set every field"
    assert combined == _every_field(11)
    assert first == _every_field(1)
    assert total([]) == CatalogDatasetStats()
