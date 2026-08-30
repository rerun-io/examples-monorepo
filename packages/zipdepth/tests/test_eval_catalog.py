"""Pure scoring tests for catalog evaluation."""

import numpy as np
import pytest
from jaxtyping import Bool, Float32
from numpy import ndarray

from zipdepth.apis.eval_catalog import (
    CatalogDepthMetrics,
    MetricCatalogDepthMetrics,
    affine_reference_depth,
    align_and_score_inverse_depth,
    aligned_inverse_diagnostic,
    score_metric_depth,
)


def test_inverse_depth_alignment_recovers_affine_prediction_and_excludes_masked_pixels() -> None:
    """Fit affine inverse depth only on valid pixels before AbsRel and delta1."""
    target_hw: Float32[ndarray, "4 4"] = np.linspace(0.5, 2.0, 16, dtype=np.float32).reshape(4, 4)
    prediction_hw: Float32[ndarray, "4 4"] = 3.0 * target_hw + 7.0
    valid_hw: Bool[ndarray, "4 4"] = np.ones((4, 4), dtype=bool)
    valid_hw[0, 0] = False
    prediction_hw[0, 0] = -10000.0

    metrics: CatalogDepthMetrics = align_and_score_inverse_depth(prediction_hw, target_hw, valid_hw)

    assert metrics.abs_rel == pytest.approx(0.0, abs=1e-6)
    assert metrics.delta1 == 1.0


def test_direct_metric_scoring_reports_unaligned_absrel_delta1_and_mae() -> None:
    """Measure metre predictions directly and ignore invalid teacher pixels."""
    target_hw: Float32[ndarray, "4 4"] = np.ones((4, 4), dtype=np.float32)
    prediction_hw: Float32[ndarray, "4 4"] = np.full((4, 4), 2.0, dtype=np.float32)
    valid_hw: Bool[ndarray, "4 4"] = np.ones((4, 4), dtype=bool)
    target_hw[0, 0] = 0.0
    prediction_hw[0, 0] = 1000.0

    metrics: MetricCatalogDepthMetrics = score_metric_depth(prediction_hw, target_hw, valid_hw)

    assert metrics.abs_rel == pytest.approx(1.0)
    assert metrics.delta1 == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(1.0)


def test_affine_reference_fits_inverse_depth_to_valid_lidar() -> None:
    """Turn released relative inverse depth into metric depth only for the reference row."""
    relative_inverse_hw: Float32[ndarray, "192 256"] = np.linspace(0.2, 0.8, 192 * 256, dtype=np.float32).reshape(192, 256)
    expected_inverse_hw: Float32[ndarray, "192 256"] = 2.0 * relative_inverse_hw + 0.5
    prompt_depth_hw: Float32[ndarray, "192 256"] = 1.0 / expected_inverse_hw
    prompt_valid_hw: Bool[ndarray, "192 256"] = np.ones((192, 256), dtype=bool)
    prompt_valid_hw[40:80, 60:120] = False

    depth_hw: Float32[ndarray, "192 256"] = affine_reference_depth(
        relative_inverse_hw,
        prompt_depth_hw,
        prompt_valid_hw,
    )

    np.testing.assert_allclose(depth_hw, 1.0 / expected_inverse_hw, rtol=1e-5, atol=1e-6)


def test_aligned_inverse_result_is_named_as_a_diagnostic() -> None:
    """Keep affine alignment separate from direct metric accuracy."""
    target_depth_hw: Float32[ndarray, "4 4"] = np.linspace(0.5, 2.0, 16, dtype=np.float32).reshape(4, 4)
    prediction_depth_hw: Float32[ndarray, "4 4"] = 1.0 / (3.0 / target_depth_hw + 7.0)
    valid_hw: Bool[ndarray, "4 4"] = np.ones((4, 4), dtype=bool)

    metrics: MetricCatalogDepthMetrics = aligned_inverse_diagnostic(prediction_depth_hw, target_depth_hw, valid_hw)

    assert metrics.abs_rel == pytest.approx(0.0, abs=1e-6)
    assert metrics.delta1 == pytest.approx(1.0)
    assert metrics.mae == pytest.approx(0.0, abs=1e-6)
