"""Verify cards: schema (strict, coerced) and fail-closed band math."""

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dagkit.verify import Case, GraphBlock, MetricBand, case_noise, check_band, golden_path, load_card

CARD = """
package = "exo-calib"

[[graph]]
name = "calibrate"
run = "exo_calib.verify:run"
score = "exo_calib.verify:score"
metrics = [{ name = "rot_deg_mean", unit = "deg", better = "lower", floor = 0.1 }]
shots = [{ view = "3d", t_s = 0 }]

[[graph.cases]]
name = "short"
clip = "short"
reference = "gt"
dataset = "dataforge-assembly101"
segment = "seg"
noise = { rot_deg_mean = 0.02 }

[[graph.cases]]
name = "full"
clip = "full"
reference = "gt"
noise_from = "short"

[[tests]]
name = "ops"
tier = "op"
lane = "unit"
tests = ["tests/test_ops.py"]
"""

LOWER = MetricBand("ate", "m", "lower", 0.1)
HIGHER = MetricBand("psnr", "dB", "higher", 0.3)


def test_card_loads_and_full_case_reuses_short_noise(tmp_path: Path) -> None:
    (tmp_path / "VERIFY.toml").write_text(CARD)
    card = load_card(tmp_path / "VERIFY.toml")
    block: GraphBlock = card.graph[0]
    assert block.shots[0].t_s == 0.0 and card.tests[0].tier == "op"
    assert case_noise(block, block.cases[1]) == {"rot_deg_mean": 0.02}
    (tmp_path / "BAD.toml").write_text(CARD.replace('package = "exo-calib"', 'package = "exo-calib"\nowner = "x"'))
    with pytest.raises(ValueError, match="BAD.toml"):
        load_card(tmp_path / "BAD.toml")


def test_band_tolerance_is_twice_noise_or_floor() -> None:
    assert check_band(LOWER, 1.0, 1.1, {"ate": 0.01}).passed  # floor 0.1 wins
    assert not check_band(LOWER, 1.0, 1.11, {"ate": 0.01}).passed
    assert check_band(LOWER, 1.0, 1.4, {"ate": 0.2}).passed  # 2 x noise wins
    assert check_band(HIGHER, 20.0, 19.7, {"psnr": 0.0}).passed and not check_band(HIGHER, 20.0, 19.6, {"psnr": 0.0}).passed


@pytest.mark.parametrize(("pre", "post", "noise", "reason"), [
    (None, 1.0, {"ate": 0.1}, "missing metric"),
    (1.0, None, {"ate": 0.1}, "missing metric"),
    (math.nan, 1.0, {"ate": 0.1}, "non-finite"),
    (1.0, math.inf, {"ate": 0.1}, "non-finite"),
    (1.0, 1.0, {}, "no measured noise"),
])  # fmt: skip
def test_band_fails_closed(pre: float | None, post: float | None, noise: dict[str, float], reason: str) -> None:
    result = check_band(LOWER, pre, post, noise)
    assert not result.passed and reason in result.reason


@given(st.floats(0, 1e3), st.floats(0, 10), st.floats(0, 10))
def test_post_equal_to_pre_always_passes(pre: float, noise: float, floor: float) -> None:
    assert check_band(MetricBand("m", "u", "lower", floor), pre, pre, {"m": noise}).passed


def test_golden_path_needs_the_goldens_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DAGKIT_GOLDENS", raising=False)
    with pytest.raises(RuntimeError, match="DAGKIT_GOLDENS"):
        golden_path("pkg", "block", "case", "pre")
    monkeypatch.setenv("DAGKIT_GOLDENS", str(tmp_path))
    assert golden_path("pkg", "block", "short", "pre2") == tmp_path / "pkg/block/short.pre2.rrd"


def test_case_without_noise_or_source_has_none() -> None:
    case = Case("c", "short", "gt")
    assert case_noise(GraphBlock("b", "m:r", "m:s", [case], [LOWER]), case) == {}
