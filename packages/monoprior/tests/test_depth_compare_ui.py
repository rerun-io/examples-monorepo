"""Depth-compare UI: every registered predictor must be selectable from the model dropdowns.

The dropdowns are populated from ``RELATIVE_PREDICTORS`` / ``METRIC_PREDICTORS``, while the submit
handler narrows the chosen name through ``_relative_predictor_name`` / ``_metric_predictor_name``.
A predictor registered in the Literal but missing from the narrowing match is shown to the user and
then rejected with "is not a ... predictor" on submit.
"""

from typing import get_args

import gradio as gr
import pytest

from monopriors.gradio_ui.depth_compare_ui import _metric_predictor_name, _relative_predictor_name
from monopriors.models.metric_depth import METRIC_PREDICTORS
from monopriors.models.relative_depth import RELATIVE_PREDICTORS


@pytest.mark.parametrize("name", get_args(RELATIVE_PREDICTORS))
def test_every_relative_predictor_is_selectable(name: str) -> None:
    assert _relative_predictor_name(name) == name  # type: ignore[arg-type]


@pytest.mark.parametrize("name", get_args(METRIC_PREDICTORS))
def test_every_metric_predictor_is_selectable(name: str) -> None:
    assert _metric_predictor_name(name) == name  # type: ignore[arg-type]


def test_cross_type_selection_is_rejected() -> None:
    with pytest.raises(gr.Error):
        _relative_predictor_name("UniDepthMetricPredictor")
    with pytest.raises(gr.Error):
        _metric_predictor_name("ZipDepthPredictor")
