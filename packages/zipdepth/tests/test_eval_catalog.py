"""Pure scoring tests for catalog evaluation."""

import numpy as np
import pytest
from jaxtyping import Bool, Float32
from numpy import ndarray

from zipdepth.apis.eval_catalog import (
    MetricCatalogDepthMetrics,
    affine_reference_depth,
    aligned_inverse_diagnostic,
    score_metric_depth,
)
from zipdepth.evaluation.edge_metrics import EdgeStratifiedResult, edge_stratified_mae


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


def test_edge_weighted_mae_uses_valid_teacher_gradient_magnitude() -> None:
    """Weight errors at teacher steps of magnitude one and three by those magnitudes."""
    target_depth_hw: Float32[ndarray, "10 10"] = np.ones((10, 10), dtype=np.float32)
    target_depth_hw[:, 5:] += 1.0
    target_depth_hw[:, 8:] += 3.0
    target_valid_hw: Bool[ndarray, "10 10"] = np.ones((10, 10), dtype=bool)
    prediction_depth_hw: Float32[ndarray, "10 10"] = target_depth_hw.copy()
    prediction_depth_hw[:, 5] += 2.0
    prediction_depth_hw[:, 8] += 4.0
    baseline_depth_hw: Float32[ndarray, "10 10"] = target_depth_hw.copy()
    baseline_depth_hw[:, 5] += 5.0
    baseline_depth_hw[:, 8] += 1.0

    metrics: EdgeStratifiedResult | None = edge_stratified_mae(
        prediction_depth_hw,
        target_depth_hw,
        target_valid_hw,
        baseline_depth_hw,
    )

    assert metrics is not None
    assert metrics.prediction.ewmae_m == pytest.approx((1.0 * 2.0 + 3.0 * 4.0) / (1.0 + 3.0))
    assert metrics.baseline is not None
    assert metrics.baseline.ewmae_m == pytest.approx((1.0 * 5.0 + 3.0 * 1.0) / (1.0 + 3.0))


def test_edge_stencil_does_not_treat_invalid_hole_borders_as_edges() -> None:
    """Exclude depth differences that cross an invalid teacher pixel."""
    target_depth_hw: Float32[ndarray, "10 11"] = np.ones((10, 11), dtype=np.float32)
    target_depth_hw[:, 6:] = 2.0
    target_depth_hw[:, 9:] = 4.0
    target_depth_hw[:, 5] = 1000.0
    target_valid_hw: Bool[ndarray, "10 11"] = np.ones((10, 11), dtype=bool)
    target_valid_hw[:, 5] = False
    prediction_depth_hw: Float32[ndarray, "10 11"] = target_depth_hw.copy()
    baseline_depth_hw: Float32[ndarray, "10 11"] = target_depth_hw.copy()
    prediction_depth_hw[:, 6] += 3.0

    metrics: EdgeStratifiedResult | None = edge_stratified_mae(
        prediction_depth_hw,
        target_depth_hw,
        target_valid_hw,
        baseline_depth_hw,
    )

    assert metrics is not None
    assert not bool(metrics.edge_hw[:, 6].any())
    assert metrics.prediction.edge_mae_m == pytest.approx(0.0)
