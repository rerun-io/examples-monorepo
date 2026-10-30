"""Quality ranking follows refinement, SOMA acceptance, then fit residuals."""

import math

import pytest

pytest.importorskip("rerun.catalog", reason="requires the dataforge catalog environment")

from dataforge.apis.selfcap_quality import QualityInput, rank_segments


def test_rank_segments_prioritizes_refinement_then_pass_then_mean_residual() -> None:
    rows: list[QualityInput] = [
        QualityInput("unrefined-pass", False, "pass", 0.1, 0.1),
        QualityInput("refined-flag", True, "flag", 0.1, 0.1),
        QualityInput("refined-pass-worse", True, "pass", 10.0, 30.0),
        QualityInput("refined-pass-best", True, "pass", 20.0, 10.0),
    ]
    assert rank_segments(rows) == {
        "refined-pass-best": 1,
        "refined-pass-worse": 2,
        "refined-flag": 3,
        "unrefined-pass": 4,
    }


def test_rank_segments_puts_unknown_residuals_last_and_breaks_ties_by_segment_id() -> None:
    rows: list[QualityInput] = [
        QualityInput("z", True, "pass", None, 1.0),
        QualityInput("b", True, "pass", 4.0, 2.0),
        QualityInput("a", True, "pass", 2.0, 4.0),
        QualityInput("nan", True, "pass", math.nan, 1.0),
        QualityInput("negative", True, "pass", -1.0, 1.0),
        QualityInput("unknown", None, None, None, None),
    ]
    assert rank_segments(rows) == {"a": 1, "b": 2, "nan": 3, "negative": 4, "z": 5, "unknown": 6}
