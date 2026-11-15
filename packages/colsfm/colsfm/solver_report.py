# SPDX-License-Identifier: Apache-2.0
"""The one reader of Ceres' brief report, and the only shape its numbers take.

`pycolmap.BundleAdjustmentSummary` exposes `termination_type`, `num_residuals`
and `is_solution_usable` as attributes, but the iteration count and the two
costs appear *only* inside the one-line string `brief_report()` returns. Every
caller that wants them therefore has to parse text, and until this module
existed the mapper and `tools/audit/caspar_polish.py` each carried their own
copy of the regular expression — together with the same misreading of a failed
parse as `0` iterations at `0.0` cost.

That fallback is wrong in a way that matters: CASPAR's backend writes no Ceres
line at all, so "the report did not parse" is the normal outcome on that path,
and a zero-cost solve is a perfectly plausible-looking number for a report to
carry. `SolverReport` keeps unavailable statistics as `None`, so a reader can
tell "the solver did not say" from "the solver said zero".

The module deliberately imports nothing beyond the standard library: it takes
the report *text*, not a `BundleAdjustmentSummary`, so that it costs no
pycolmap import and can be exercised without a solver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

CeresTermination: TypeAlias = Literal["CONVERGENCE", "NO_CONVERGENCE", "FAILURE"]
"""How Ceres ended a solve, as `TerminationType`'s bare name.

Three values, not a `str`: every solve in the package -- pycolmap's bundle adjustment,
the extrinsics-only pyceres problem, the pose graph -- reports the same enum, and four
docstrings used to spell the same three names in prose beside a `str` annotation. Ceres'
own `USER_SUCCESS` and `USER_FAILURE` cannot arrive here because nothing in this package
installs an `IterationCallback`, which is the only thing that produces them.
"""

BRIEF_REPORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"Iterations:\s*(\d+),\s*Initial cost:\s*([0-9.eE+-]+),\s*Final cost:\s*([0-9.eE+-]+)"
)
"""`BundleAdjustmentSummary` exposes iterations and cost only inside `brief_report()`."""


@dataclass(frozen=True, slots=True)
class SolverReport:
    """What one non-linear solve reported about its own progress.

    Every field is optional because every field is optional in reality: a
    backend that writes no Ceres-shaped report leaves all three unknown, and a
    reader must be able to see that rather than a fabricated zero.
    """

    num_iterations: int | None
    """Iterations the solver took, or None when it reported none."""
    initial_cost: float | None
    """Cost before the solve, or None when it reported none.

    This is the solver's own objective, not cuSFM's normalised RMS `Initial cost`."""
    final_cost: float | None
    """Cost after the solve, or None when it reported none."""

    @staticmethod
    def unavailable() -> SolverReport:
        """The report of a solver that published no statistics.

        Returns:
            A report whose every field is None.
        """
        return SolverReport(num_iterations=None, initial_cost=None, final_cost=None)

    @property
    def is_available(self) -> bool:
        """Whether the solver published its iterations and costs at all."""
        return self.num_iterations is not None

    @property
    def cost_reduction(self) -> float | None:
        """Fraction of the initial cost the solve removed, `1 - final / initial`.

        Returns:
            The fraction, or None when either cost is unknown or the initial cost
            is zero — in which case no fraction of it exists to report.
        """
        if self.initial_cost is None or self.final_cost is None or self.initial_cost == 0.0:
            return None
        return 1.0 - self.final_cost / self.initial_cost


def parse_brief_report(brief_report: str) -> SolverReport:
    """Read iterations and costs out of a solver's one-line report.

    Args:
        brief_report: The text `pycolmap.BundleAdjustmentSummary.brief_report()`
            returns, or any string; anything that does not carry Ceres' triple is
            unavailable rather than an error.

    Returns:
        The three numbers, or `SolverReport.unavailable()` when the text does not
        carry them.
    """
    match: re.Match[str] | None = BRIEF_REPORT_PATTERN.search(brief_report)
    if match is None:
        return SolverReport.unavailable()
    return SolverReport(
        num_iterations=int(match.group(1)),
        initial_cost=float(match.group(2)),
        final_cost=float(match.group(3)),
    )
