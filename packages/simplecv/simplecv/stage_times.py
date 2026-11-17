"""Per-frame wall time of each stage of a pipeline, and its summary."""

from dataclasses import dataclass, field

import numpy as np
from serde import serde


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class StageSummary:
    """One stage over the measured frames."""

    median_ms: float
    mean_ms: float
    p90_ms: float
    frames: int


@dataclass(slots=True)
class StageTimes:
    """Per-frame milliseconds of every stage, in the order the stages were first seen."""

    values: dict[str, list[float]] = field(default_factory=dict)
    """Stage name to one sample per frame that ran it."""

    def add(self, stage: str, milliseconds: float) -> None:
        """Record one frame's time for one stage.

        Args:
            stage: Stage name.
            milliseconds: Wall time.
        """
        self.values.setdefault(stage, []).append(milliseconds)

    def summary(self) -> dict[str, StageSummary]:
        """Median, mean and p90 per stage."""
        return {
            stage: StageSummary(float(np.median(v)), float(np.mean(v)), float(np.quantile(v, 0.9)), len(v)) for stage, v in self.values.items()
        }

    def line(self, unit: str) -> str:
        """One line: each stage's median and mean, with its own count where it differs from the longest column.

        Args:
            unit: What one sample is, e.g. ``frameset``.
        """
        if not self.values:
            return f"ms per {unit}: nothing timed"
        frames: int = max(len(samples) for samples in self.values.values())
        parts: list[str] = []
        for stage, samples in self.values.items():
            count: str = "" if len(samples) == frames else f" (n={len(samples)})"
            parts.append(f"{stage} {np.median(samples):.1f} / {np.mean(samples):.1f}{count}")
        return f"ms per {unit}, median / mean over {frames}: " + ", ".join(parts)
