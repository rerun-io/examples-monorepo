"""Policy replay over the frozen, pre-decision 374-session audit."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from serde import serde
from serde.json import from_json

from selfcap_curate.exo_label import GateThresholds, accepts
from selfcap_curate.exo_refine import Refinement
from selfcap_curate.exo_refine_catalog import Gate, Score


@serde(deny_unknown_fields=True)
@dataclass(frozen=True)
class Case:
    segment: str
    source_sha256: str
    baseline: dict[str, Score]
    candidate: dict[str, Score]
    cv_converged: tuple[bool, ...]
    full_converged: bool | None
    heldout_labels: dict[str, Score]
    deployed_labels: dict[str, Score]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True)
class Snapshot:
    captured_at: str
    expected_accepts: int
    cases: list[Case]


@pytest.mark.golden
def test_gate_v3_replays_exactly_118_of_374_stored_records() -> None:
    snapshot = from_json(Snapshot, (Path(__file__).parent/'fixtures/gate_v3_374.json').read_text())
    decisions = {}
    for case in snapshot.cases:
        folds = tuple(Refinement((), np.zeros(3), 'audit', 'fixed', converged=value) for value in case.cv_converged)
        gate = Gate(case.baseline, case.candidate, folds, False, heldout_labels=case.heldout_labels,
                    deployed_labels=case.deployed_labels, full_fit_converged=case.full_converged)
        decisions[case.segment] = accepts(gate)
    assert len(decisions) == 374
    assert snapshot.expected_accepts == 118
    assert sum(decisions.values()) == 118
    for suffix in ('fe05fe3e', '021d2e8b', '9c74b2ff', '47387024'):
        assert not next(value for segment, value in decisions.items() if suffix in segment)


def test_v3_requires_full_fit_and_allows_only_one_failed_fold() -> None:
    before = {name: Score(10, 20.0, 30.0) for name in ('headset', 'phones', 'sapiens_hands', 'rtmw_hands', 'body_transfer')}
    after = {name: Score(10, 10.0, 20.0) for name in before}
    after['sapiens_hands'] = Score(10, 25.0, 30.0)
    good = Refinement((), np.zeros(3), 'test', 'fixed')
    bad = Refinement((), np.zeros(3), 'test', 'fixed', converged=False)
    assert accepts(Gate(before, after, (good, bad), False, full_fit_converged=True))
    assert not accepts(Gate(before, after, (good, bad), False, full_fit_converged=False))
    assert not accepts(Gate(before, after, (good, bad), False))
    assert not accepts(Gate(before, after, (bad, bad), False, full_fit_converged=True))
    assert not accepts(Gate(before, after, (good, bad), False, full_fit_converged=True), GateThresholds(max_nonconverged_cv_folds=0))


def test_v3_cannot_disable_full_convergence_or_allow_two_failed_folds() -> None:
    with pytest.raises(ValueError, match='full-fit convergence'):
        GateThresholds(require_full_fit_convergence=False)
    with pytest.raises(ValueError, match='at most one'):
        GateThresholds(max_nonconverged_cv_folds=2)
