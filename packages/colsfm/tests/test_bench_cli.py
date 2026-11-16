"""The benchmark CLI's process result.

The seam is `colsfm.bench_cli.acceptance_exit_code`: the acceptance checks in,
the process status out. `run` prints and writes the artifacts; what a caller — a
person, a CI job — acts on is the exit status, and it has to mean what the report
says.

The policy under test: a failed measurable bound is a non-zero status; a bound
that could not be measured is reported but does not fail the run.
"""

from __future__ import annotations

import pytest

from colsfm.bench_cli import acceptance_exit_code
from colsfm.benchmark import AcceptanceCheck


def _check(name: str, passed: bool | None) -> AcceptanceCheck:
    """One acceptance check with the given verdict.

    Args:
        name: The check's name.
        passed: True, False, or None for a bound that could not be measured.

    Returns:
        The check.
    """
    return AcceptanceCheck(name=name, bound=">= something", value=None if passed is None else 1.0, passed=passed)


def test_a_failed_bound_fails_the_process() -> None:
    """A FAIL row in the report has to reach the caller as a non-zero status."""
    checks: tuple[AcceptanceCheck, ...] = (_check("registered images", True), _check("total runtime ratio", False))

    assert acceptance_exit_code(checks) != 0


def test_every_bound_met_exits_zero() -> None:
    """All PASS is a clean exit."""
    assert acceptance_exit_code((_check("registered images", True), _check("total runtime ratio", True))) == 0


def test_an_unmeasurable_bound_is_reported_but_does_not_fail() -> None:
    """`n/a` is a third state: not met, not failed, and not a non-zero exit."""
    checks: tuple[AcceptanceCheck, ...] = (_check("registered images", True), _check("ATE vs ground truth (mm)", None))

    assert acceptance_exit_code(checks) == 0


def test_no_bounds_at_all_exits_zero() -> None:
    """A comparison run without bounds asserts nothing, so it fails nothing."""
    assert acceptance_exit_code(()) == 0


@pytest.mark.parametrize("verdicts", [(False,), (None, False), (True, False, None)])
def test_one_failure_anywhere_is_enough(verdicts: tuple[bool | None, ...]) -> None:
    """Any failed bound decides the status, whatever the others did."""
    checks: tuple[AcceptanceCheck, ...] = tuple(_check(f"check {index}", passed) for index, passed in enumerate(verdicts))

    assert acceptance_exit_code(checks) != 0
