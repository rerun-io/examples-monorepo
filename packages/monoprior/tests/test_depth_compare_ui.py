"""Depth-compare UI: registered predictors are selectable and cross-type choices are rejected."""

from typing import get_args

import gradio as gr
import numpy as np
import pytest
from jaxtyping import Float, UInt8

from monopriors.gradio_ui.depth_compare_ui import _relative_predictor_name, change_dropdown, on_submit
from monopriors.models.metric_depth import BaseMetricPredictor, MetricDepthPrediction, metric_predictor_defaults
from monopriors.models.relative_depth import RELATIVE_PREDICTORS


class SyntheticMetricPredictor(BaseMetricPredictor):
    """Exercise selection and logging without downloading model weights."""

    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None) -> MetricDepthPrediction:
        assert K_33 is not None
        return MetricDepthPrediction(
            depth_meters=np.full(rgb.shape[:2], 2.0, dtype=np.float32),
            confidence=np.ones(rgb.shape[:2], dtype=np.float32),
            K_33=K_33,
        )


@pytest.mark.parametrize("name", get_args(RELATIVE_PREDICTORS))
def test_every_relative_predictor_is_selectable(name: str) -> None:
    assert _relative_predictor_name(name) == name  # type: ignore[arg-type]


@pytest.mark.parametrize("name", metric_predictor_defaults)
def test_every_metric_predictor_is_selectable(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    dropdowns: tuple[gr.Dropdown, gr.Dropdown] = change_dropdown("Metric")
    assert all(name in [value for _, value in dropdown.choices] for dropdown in dropdowns)
    monkeypatch.setattr(type(metric_predictor_defaults[name]), "setup", lambda self, device: SyntheticMetricPredictor())
    rgb: UInt8[np.ndarray, "h w 3"] = np.zeros((8, 12, 3), dtype=np.uint8)
    assert on_submit(rgb, False, 0.05, "Metric", name, name)


def test_cross_type_selection_is_rejected() -> None:
    with pytest.raises(gr.Error):
        _relative_predictor_name("UniDepthMetricPredictor")
    rgb: UInt8[np.ndarray, "h w 3"] = np.zeros((8, 12, 3), dtype=np.uint8)
    with pytest.raises(gr.Error, match="not a metric depth predictor"):
        on_submit(rgb, False, 0.05, "Metric", "ZipDepthPredictor", "unidepth-metric")
