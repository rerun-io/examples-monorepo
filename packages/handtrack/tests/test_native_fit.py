"""Recorded real tracker calls are the oracle for the native warm and cold fit."""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from handtrack.fit.pose_fit import fit_pose as torch_fit
from handtrack.hand.pose import landmarks

pytestmark = pytest.mark.golden
GOLDEN_CALLS: Path = Path(__file__).resolve().parents[1] / "data/fastfit/golden/fit_calls.pt"
"""Recorded torch fit calls (handtrack's export_golden), in the package's untracked data/ directory."""


def test_native_golden_calls() -> None:
    path: Path = GOLDEN_CALLS
    if not path.is_file():
        pytest.skip(f"golden fit calls absent: {path}")
    from handtrack.fit.native import fit_pose

    torch.set_num_threads(1)
    calls = torch.load(path, weights_only=False)
    translations: list[float] = []
    points: list[float] = []
    angles: list[float] = []
    energy: list[float] = []
    flips: list[bool] = []
    cold_translations: list[float] = []
    cold_energy: list[float] = []
    for call in calls:
        if call["kind"] != "fit_pose" or call["nested"]:
            continue
        bound = inspect.signature(torch_fit).bind(*call["args"], **call["kwargs"])
        bound.apply_defaults()
        model, phi, hands, previous, config = bound.arguments.values()
        got = fit_pose(model, phi, hands, previous, config)
        for hand, prior, actual, expected in zip(hands, previous, got, call["result"], strict=True):
            if prior is None:
                cold_translations.append(float((actual.pose.translation - expected.pose.translation).norm()) * 1000)
                cold_energy.append(actual.energy - expected.energy)
                continue
            translations.append(float((actual.pose.translation - expected.pose.translation).norm()) * 1000)
            points.extend(((landmarks(model, actual.pose, hand.side) - landmarks(model, expected.pose, hand.side)).norm(dim=-1) * 1000).tolist())
            angles.extend((actual.pose.joint_angles[:20] - expected.pose.joint_angles[:20]).abs().tolist())
            energy.append(max(0.0, actual.energy - expected.energy) / max(abs(expected.energy), 1e-9))
            flips.append(actual.converged != expected.converged)
    assert translations, f"no warm rows in {path}"
    assert np.median(translations) <= 0.01
    assert np.percentile(translations, 90) <= 0.05
    assert max(translations) <= 2
    assert np.median(points) <= 0.05
    assert np.percentile(points, 99) <= 1
    assert np.median(angles) <= 1e-3
    assert np.median(energy) <= 1e-4
    assert np.percentile(energy, 90) <= 1e-3
    assert np.mean(flips) <= 0.02

    assert len(cold_translations) == 21
    assert np.isfinite(cold_translations).all()
    assert np.isfinite(cold_energy).all()
    # A different minimum passes only when it improves the recorded energy.
    for percentile, limit in [(50, 0.1), (90, 0.5)]:
        if np.percentile(cold_translations, percentile) > limit:
            assert all(energy <= 0.0 for error, energy in zip(cold_translations, cold_energy, strict=True) if error > limit)


@pytest.mark.parametrize("warm", [True, False])
def test_without_evidence_matches_torch(warm: bool) -> None:
    from dataclasses import replace

    from handtrack.fit.native import fit_pose

    path = GOLDEN_CALLS
    if not path.is_file():
        pytest.skip(f"golden fit calls absent: {path}")
    torch.set_num_threads(1)
    for call in torch.load(path, weights_only=False):
        if call["kind"] != "fit_pose" or call["nested"]:
            continue
        bound = inspect.signature(torch_fit).bind(*call["args"], **call["kwargs"])
        bound.apply_defaults()
        model, phi, hands, previous, config = bound.arguments.values()
        if not all(p is not None for p in previous):
            continue
        hands = [replace(hand, views=tuple(replace(view, weights=torch.zeros_like(view.weights)) for view in hand.views)) for hand in hands]
        if not warm:
            previous = [None] * len(hands)
        for actual, expected in zip(fit_pose(model, phi, hands, previous, config), torch_fit(model, phi, hands, previous, config), strict=True):
            torch.testing.assert_close(actual.pose.translation, expected.pose.translation)
            torch.testing.assert_close(actual.pose.joint_angles, expected.pose.joint_angles)
            assert actual.energy == pytest.approx(expected.energy, abs=1e-9, nan_ok=True)
            assert (actual.converged, actual.termination, actual.iterations) == (expected.converged, expected.termination, expected.iterations)
        break
    else:
        pytest.fail("no warm row in golden fixture")
