"""Dependency-free catalog pipeline statistics."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass

STAGE_FIELDS: tuple[str, ...] = ("segment_query", "video_decode", "jpeg_decode", "blob_decode", "augment")
"""Floating-point stage timers reported by catalog instrumentation."""


@dataclass(slots=True)
class BuilderStats:
    """Mutable counters owned by one sample builder."""

    blob_decode: float = 0.0
    """PromptDA PNG blob decode time in seconds."""
    augment: float = 0.0
    """Orientation, target construction, and augmentation time in seconds."""
    samples_built: int = 0
    """Samples successfully built."""
    png_fallbacks: int = 0
    """PromptDA frames handled by the general PNG decoder."""
    skipped_flat_frames: int = 0
    """Frames rejected by the configured depth-span filter."""
    skipped_low_valid_frames: int = 0
    """Ultrawide frames rejected by the minimum valid-target-fraction gate."""


@dataclass(slots=True)
class CatalogDatasetStats:
    """Cumulative catalog pipeline time and counters."""

    segment_query: float = 0.0
    """Target/video catalog query and decoder-initialization time in seconds."""
    video_decode: float = 0.0
    """Video frame decode time in seconds."""
    jpeg_decode: float = 0.0
    """Rectified ultrawide JPEG decode time in seconds."""
    blob_decode: float = 0.0
    """PromptDA PNG blob decode time in seconds."""
    augment: float = 0.0
    """Orientation, target construction, and augmentation time in seconds."""
    samples_built: int = 0
    """Samples successfully built, including samples not yet yielded."""
    png_fallbacks: int = 0
    """PromptDA frames handled by the general PNG decoder."""
    skipped_frames: int = 0
    """Video frames skipped after decode errors."""
    skipped_flat_frames: int = 0
    """Frames rejected by the configured depth-span filter."""
    skipped_low_valid_frames: int = 0
    """Ultrawide frames rejected by the minimum valid-target-fraction gate."""

    @classmethod
    def from_builder(cls, builder_stats: BuilderStats) -> "CatalogDatasetStats":
        """Lift builder-local counters into the dataset schema; dataset-only counters stay zero."""
        return cls(**asdict(builder_stats))

    def __add__(self, other: "CatalogDatasetStats") -> "CatalogDatasetStats":
        """Return field-wise sums without mutating either operand."""
        return CatalogDatasetStats(
            segment_query=self.segment_query + other.segment_query,
            video_decode=self.video_decode + other.video_decode,
            jpeg_decode=self.jpeg_decode + other.jpeg_decode,
            blob_decode=self.blob_decode + other.blob_decode,
            augment=self.augment + other.augment,
            samples_built=self.samples_built + other.samples_built,
            png_fallbacks=self.png_fallbacks + other.png_fallbacks,
            skipped_frames=self.skipped_frames + other.skipped_frames,
            skipped_flat_frames=self.skipped_flat_frames + other.skipped_flat_frames,
            skipped_low_valid_frames=self.skipped_low_valid_frames + other.skipped_low_valid_frames,
        )


def total(stats: Iterable[CatalogDatasetStats]) -> CatalogDatasetStats:
    """Return the field-wise sum of catalog statistics."""
    combined: CatalogDatasetStats = CatalogDatasetStats()
    item: CatalogDatasetStats
    for item in stats:
        combined = combined + item
    return combined
