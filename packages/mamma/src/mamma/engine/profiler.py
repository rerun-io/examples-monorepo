"""Lightweight per-stage wall-clock accumulator for the streaming loop."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager


class StageProfiler:
    """Accumulates per-stage seconds across ticks; feeds the benchmark report."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        start: float = time.perf_counter()
        try:
            yield
        finally:
            elapsed: float = time.perf_counter() - start
            self.totals[name] = self.totals.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self) -> str:
        """Human-readable per-stage table sorted by total time."""
        lines: list[str] = [f"{'stage':<16}{'total_s':>9}{'calls':>7}{'ms/call':>9}"]
        for name, total in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            count: int = self.counts[name]
            lines.append(f"{name:<16}{total:>9.2f}{count:>7}{1000.0 * total / max(count, 1):>9.2f}")
        return "\n".join(lines)
