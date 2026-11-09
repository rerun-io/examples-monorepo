"""The relative accuracy contract for hand inference optimizations."""

import math
from collections.abc import Sequence

import numpy as np


def within_accuracy_band(baseline: float | None, candidate: float | None, *, higher_is_better: bool) -> bool:
    """Accept at most five percent relative loss; missing/nonfinite evidence fails."""
    if baseline is None or candidate is None or not math.isfinite(baseline) or not math.isfinite(candidate):
        return False
    return candidate >= 0.95 * baseline if higher_is_better else candidate <= 1.05 * baseline


def error_quantiles(errors: Sequence[float]) -> tuple[float | None, float | None]:
    """Median and p95 of scored joint errors, with no silent filtering."""
    if not errors:
        return None, None
    if not all(math.isfinite(x) for x in errors):
        raise ValueError("Scored errors must be finite.")
    return float(np.median(errors)), float(np.quantile(errors, 0.95))
