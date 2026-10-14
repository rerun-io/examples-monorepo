"""Verify cards: schema (strict, coerced) and fail-closed band math."""

import math
from pathlib import Path
from types import SimpleNamespace

import dagkit_fakes
import pytest
import tyro
from hypothesis import given
from hypothesis import strategies as st

from dagkit import verify
from dagkit.sources import SegmentRef
from dagkit.verify import Case, GraphBlock, MetricBand, check_band, golden_path, load_card, pre_golden

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
published = { rot_deg_mean = 0.5 }

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
    assert block.cases[0].noise == {"rot_deg_mean": 0.02} and block.cases[1].noise_from == "short"
    assert block.cases[0].published == {"rot_deg_mean": 0.5} and block.cases[1].published == {}
    (tmp_path / "BAD.toml").write_text(CARD.replace('package = "exo-calib"', 'package = "exo-calib"\nowner = "x"'))
    with pytest.raises(ValueError, match="BAD.toml"):
        load_card(tmp_path / "BAD.toml")


def test_a_case_segment_ref_carries_its_window_in_ns_and_a_local_case_has_none() -> None:
    case = Case("short", "short", "gt", dataset="ds", segment="seg", window_s=(1.5, 2.25))
    assert case.segment_ref() == SegmentRef("ds", "seg", window_ns=(1_500_000_000, 2_250_000_000))
    assert case.segment_ref("rerun+http://host:1").catalog_url == "rerun+http://host:1"
    assert Case("full", "full", "gt", dataset="ds", segment="seg").segment_ref().window_ns is None
    with pytest.raises(ValueError, match="'local' has no catalog segment"):
        Case("local", "short", "pre-port", input="data/image.png").segment_ref()


def test_band_tolerance_is_twice_noise_or_floor() -> None:
    assert check_band(LOWER, 1.0, 1.1, {"ate": 0.01}).passed  # floor 0.1 wins
    assert not check_band(LOWER, 1.0, 1.11, {"ate": 0.01}).passed
    assert check_band(LOWER, 1.0, 1.4, {"ate": 0.2}).passed  # 2 x noise wins
    assert check_band(HIGHER, 20.0, 19.7, {"psnr": 0.0}).passed and not check_band(HIGHER, 20.0, 19.6, {"psnr": 0.0}).passed


def test_a_published_value_is_the_base_only_when_it_is_worse_than_pre() -> None:
    assert check_band(LOWER, 1.0, 1.5, {"ate": 0.0}, published=1.45).passed  # base 1.45 (the paper), tol = floor 0.1
    assert not check_band(LOWER, 1.0, 1.56, {"ate": 0.0}, published=1.45).passed
    assert check_band(LOWER, 1.0, 1.1, {"ate": 0.0}, published=0.5).passed and not check_band(LOWER, 1.0, 1.11, {"ate": 0.0}, published=0.5).passed
    assert check_band(HIGHER, 20.0, 18.8, {"psnr": 0.0}, published=19.0).passed and not check_band(HIGHER, 20.0, 18.6, {"psnr": 0.0}, published=19.0).passed
    result = check_band(LOWER, 1.0, 1.0, {"ate": 0.0}, published=math.nan)
    assert not result.passed and "non-finite" in result.reason


@pytest.mark.parametrize(("pre", "post", "noise", "reason"), [
    (None, 1.0, {"ate": 0.1}, "missing metric"),
    (1.0, None, {"ate": 0.1}, "missing metric"),
    (math.nan, 1.0, {"ate": 0.1}, "non-finite"),
    (1.0, math.inf, {"ate": 0.1}, "non-finite"),
    (1.0, 1.0, {}, "no finite, non-negative measured noise"),
    (1.0, 1e99, {"ate": math.inf}, "no finite, non-negative measured noise"),
    (1.0, 1.0, {"ate": -0.1}, "no finite, non-negative measured noise"),
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


def test_pre_golden_is_the_pre_run_beside_a_golden_or_its_partial(tmp_path: Path) -> None:
    case = Case("short", "short", "pre-port", dataset="ds", segment="seg")
    with pytest.raises(FileNotFoundError, match="run --phase pre first"):
        pre_golden(case, tmp_path / "short.post.rrd")
    (tmp_path / "short.pre.rrd").write_bytes(b"")
    assert pre_golden(case, tmp_path / "short.post.partial") == pre_golden(case, tmp_path / "short.pre2.rrd") == tmp_path / "short.pre.rrd"
    (tmp_path / "short.pre.partial").write_bytes(b"")
    assert pre_golden(case, tmp_path / "short.pre.partial") == tmp_path / "short.pre.partial"  # the pre run scores itself


SHORT_CARD = """
package = "dagkit"

[[graph]]
name = "fake"
run = "dagkit_fakes:verify_run"
score = "dagkit_fakes:verify_score"
metrics = [{ name = "ate", unit = "m", better = "lower", floor = 0.1 }]

[[graph.cases]]
name = "short"
clip = "short"
reference = "pre-port"
dataset = "ds"
segment = "seg0"
noise = { ate = 0.01 }
"""


@pytest.fixture
def card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DAGKIT_GOLDENS", str(tmp_path / "goldens"))
    monkeypatch.setattr(verify, "git", lambda *args: "abc1234")
    (tmp_path / "VERIFY.toml").write_text(SHORT_CARD)
    return tmp_path / "VERIFY.toml"


def test_post_fails_closed_without_pre2_and_passes_with_it(card: Path) -> None:
    assert verify.main(verify.VerifyArgs(card, "pre")) == 0
    assert verify.main(verify.VerifyArgs(card, "post")) == 1  # no pre2 golden: no noise evidence
    assert verify.main(verify.VerifyArgs(card, "pre2")) == 0
    assert verify.main(verify.VerifyArgs(card, "post")) == 0


def test_reference_goldens_are_write_once(card: Path) -> None:
    """A pre / pre2 run on a ported tree would replace the pre-port reference with the port's own output."""
    assert verify.main(verify.VerifyArgs(card, "pre")) == 0
    golden: Path = verify.golden_path("dagkit", "fake", "short", "pre")
    kept: int = golden.stat().st_ino
    assert verify.main(verify.VerifyArgs(card, "pre")) == 1 and golden.stat().st_ino == kept
    assert verify.main(verify.VerifyArgs(card, "pre", overwrite=True)) == 0 and golden.stat().st_ino != kept
    assert verify.main(verify.VerifyArgs(card, "pre2")) == 0 and verify.main(verify.VerifyArgs(card, "pre2")) == 1
    assert verify.main(verify.VerifyArgs(card, "post")) == 0 and verify.main(verify.VerifyArgs(card, "post")) == 0  # post is rewritten


def test_post_scores_each_golden_once(card: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert verify.main(verify.VerifyArgs(card, "pre")) == 0 and verify.main(verify.VerifyArgs(card, "pre2")) == 0
    scored: list[str] = []

    def score(case: object, golden: Path) -> dict[str, float]:
        scored.append(golden.name)
        return {"ate": 1.0}

    monkeypatch.setattr(dagkit_fakes, "verify_score", score)
    assert verify.main(verify.VerifyArgs(card, "post")) == 0
    assert sorted(scored) == ["short.post.partial", "short.pre.rrd", "short.pre2.rrd"]


def test_an_empty_selection_fails(card: Path) -> None:
    assert verify.main(verify.VerifyArgs(card, "pre", block="typo")) == 1
    assert verify.main(verify.VerifyArgs(card, "pre", clip="full")) == 1


def test_register_publishes_an_immutable_copy(card: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[tuple[list[str], str]] = []

    class FakeDataset:
        def register(self, uris: list[str], layer_name: str) -> SimpleNamespace:
            registered.append((uris, layer_name))
            return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(verify.SegmentRef, "entry", lambda self: FakeDataset())
    assert verify.main(verify.VerifyArgs(card, "pre", register=True)) == 0
    assert verify.main(verify.VerifyArgs(card, "pre", register=True, overwrite=True)) == 0
    uris: list[str] = [uris[0] for uris, _ in registered]
    golden: Path = verify.golden_path("dagkit", "fake", "short", "pre")
    assert len(set(uris)) == 2 and golden.as_uri() not in uris and {name for _, name in registered} == {"verify-fake-pre-abc1234"}
    assert all(Path(uri.removeprefix("file://")).exists() for uri in uris)


# ── regressions from the 02-leaf review-loop thermo review ─────────────────────────────────────

FULL_CASE = """
[[graph.cases]]
name = "full"
clip = "full"
reference = "pre-port"
dataset = "ds"
segment = "seg0"
noise_from = "short"
"""


def test_cli_takes_the_card_as_a_positional_argument() -> None:
    args: verify.VerifyArgs = tyro.cli(verify.VerifyArgs, args=["VERIFY.toml", "--phase", "post", "--clip", "short"])
    assert args.card == Path("VERIFY.toml") and args.phase == "post" and args.clip == "short"


def test_noise_from_needs_the_donor_cases_noise_run(card: Path) -> None:
    card.write_text(SHORT_CARD + FULL_CASE.replace('noise_from = "short"', 'noise_from = "typo"'))
    assert verify.main(verify.VerifyArgs(card, "pre", clip="full")) == 0
    assert verify.main(verify.VerifyArgs(card, "pre2", clip="short")) == 0
    assert verify.main(verify.VerifyArgs(card, "post", clip="full")) == 1  # an unknown noise_from case: no noise
    card.write_text(SHORT_CARD + FULL_CASE)
    verify.golden_path("dagkit", "fake", "short", "pre2").unlink()
    assert verify.main(verify.VerifyArgs(card, "post", clip="full")) == 1  # the short case never ran pre2
    assert verify.main(verify.VerifyArgs(card, "pre2", clip="short")) == 0
    assert verify.main(verify.VerifyArgs(card, "post", clip="full")) == 1  # nor pre: one run is no noise measurement
    assert verify.main(verify.VerifyArgs(card, "pre", clip="short")) == 0
    assert verify.main(verify.VerifyArgs(card, "post", clip="full")) == 0


@pytest.mark.parametrize(("phase", "metrics"), [("pre", {"ate": math.nan}), ("pre2", {}), ("post", {"ate": math.inf})])
def test_every_phase_fails_on_a_missing_or_non_finite_metric(card: Path, monkeypatch: pytest.MonkeyPatch, phase: verify.Phase, metrics: dict) -> None:
    monkeypatch.setattr(dagkit_fakes, "verify_score", lambda case, golden: metrics)
    assert verify.main(verify.VerifyArgs(card, phase)) == 1


def test_register_skips_windowed_and_failed_cases(card: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[str] = []

    class FakeDataset:
        def register(self, uris: list[str], layer_name: str) -> SimpleNamespace:
            registered.append(layer_name)
            return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(verify.SegmentRef, "entry", lambda self: FakeDataset())
    card.write_text(SHORT_CARD.replace("noise = { ate = 0.01 }", "noise = { ate = 0.01 }\nwindow_s = [1.0, 2.0]"))
    assert verify.main(verify.VerifyArgs(card, "pre", register=True)) == 0
    card.write_text(SHORT_CARD)
    assert verify.main(verify.VerifyArgs(card, "post", register=True)) == 1  # no pre / pre2 goldens: the case failed
    assert registered == []


def test_running_the_module_parses_the_imported_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m dagkit.verify` must build `dagkit.verify.VerifyArgs` (one Case class), or beartype rejects package callbacks."""
    import runpy
    import warnings

    seen: list[type] = []

    def cli(cls: type) -> None:
        seen.append(cls)
        raise SystemExit(0)

    monkeypatch.setattr(tyro, "cli", cli)
    with warnings.catch_warnings(), pytest.raises(SystemExit):
        warnings.simplefilter("ignore", RuntimeWarning)  # runpy: the module is already imported
        runpy.run_module("dagkit.verify", run_name="__main__")
    assert seen == [verify.VerifyArgs]
