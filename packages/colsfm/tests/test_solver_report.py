# SPDX-License-Identifier: Apache-2.0
"""Behaviour of `colsfm.solver_report`, the one reader of Ceres' brief report.

Pure text parsing: no solver, no pycolmap, no GPU. The point of these tests is
the distinction the old duplicated parsers erased — a report that does not parse
must come back as *unavailable*, never as a zero-cost, zero-iteration solve.
"""

from __future__ import annotations

import pytest

from colsfm.solver_report import SolverReport, parse_brief_report

CERES_BRIEF_REPORT: str = (
    "Ceres Solver Report: Iterations: 22, Initial cost: 1.234500e+04, "
    "Final cost: 1.192000e+04, Termination: CONVERGENCE"
)
"""The one line `BundleAdjustmentSummary.brief_report()` returns after a solve."""


def test_a_ceres_report_yields_its_iterations_and_costs() -> None:
    """The three numbers Ceres hides inside its one-line report come back typed."""
    report: SolverReport = parse_brief_report(CERES_BRIEF_REPORT)

    assert report.num_iterations == 22
    assert report.initial_cost == pytest.approx(1.2345e4)
    assert report.final_cost == pytest.approx(1.1920e4)
    assert report.is_available


def test_an_unparsable_report_is_unavailable_not_zero() -> None:
    """CASPAR writes no Ceres line; that must not read as a zero-cost solve."""
    report: SolverReport = parse_brief_report("CASPAR solver finished.")

    assert report.num_iterations is None
    assert report.initial_cost is None
    assert report.final_cost is None
    assert not report.is_available


def test_an_empty_report_is_unavailable() -> None:
    """A solver that reported nothing at all is equally unavailable."""
    assert parse_brief_report("") == SolverReport.unavailable()


def test_cost_reduction_is_the_fraction_the_solve_removed() -> None:
    """`cost_reduction` is `1 - final / initial`, the number the audit tables print."""
    report: SolverReport = parse_brief_report(CERES_BRIEF_REPORT)

    assert report.cost_reduction == pytest.approx(1.0 - 1.1920e4 / 1.2345e4)


def test_cost_reduction_is_none_when_the_costs_are_unavailable() -> None:
    """No report means no reduction to quote, rather than a reduction of zero."""
    assert SolverReport.unavailable().cost_reduction is None


def test_cost_reduction_is_none_when_the_initial_cost_is_zero() -> None:
    """A solve that started at zero cost removed no measurable fraction of it."""
    report: SolverReport = parse_brief_report("Iterations: 0, Initial cost: 0.000000e+00, Final cost: 0.000000e+00")

    assert report.num_iterations == 0
    assert report.initial_cost == 0.0
    assert report.cost_reduction is None


def test_negative_exponent_costs_parse() -> None:
    """Ceres writes small costs in scientific notation with a negative exponent."""
    report: SolverReport = parse_brief_report("Iterations: 3, Initial cost: 5.0e-07, Final cost: 1.25e-09")

    assert report.initial_cost == pytest.approx(5.0e-7)
    assert report.final_cost == pytest.approx(1.25e-9)
