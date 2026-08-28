"""Pure scoring tests for catalog evaluation."""

import numpy as np
import pytest
from jaxtyping import Bool, Float32
from numpy import ndarray

from zipdepth.apis.eval_catalog import CatalogDepthMetrics, align_and_score_inverse_depth


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
