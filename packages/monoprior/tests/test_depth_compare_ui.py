"""Depth-compare UI: registered predictors are selectable and cross-type choices are rejected."""

from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from jaxtyping import Float, Float32, UInt8
from rerun.experimental import RrdReader

from monopriors.gradio_ui.depth_compare_ui import change_dropdown, on_submit
from monopriors.models.metric_depth import BaseMetricPredictor, MetricDepthPrediction, metric_predictor_defaults
from monopriors.models.relative_depth import BaseRelativePredictor, RelativeDepthPrediction, relative_predictor_defaults


class SyntheticMetricPredictor(BaseMetricPredictor):
    """Exercise selection and logging without downloading model weights."""

    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float[np.ndarray, "3 3"] | None) -> MetricDepthPrediction:
        assert K_33 is not None
        return MetricDepthPrediction(
            depth_meters=np.full(rgb.shape[:2], 2.0, dtype=np.float32),
            confidence=np.ones(rgb.shape[:2], dtype=np.float32),
            K_33=K_33,
        )


class SyntheticRelativePredictor(BaseRelativePredictor):
    """Exercise relative selection and disparity logging without model weights."""

    def __call__(self, rgb: UInt8[np.ndarray, "h w 3"], K_33: Float32[np.ndarray, "3 3"] | None) -> RelativeDepthPrediction:
        assert K_33 is None
        disparity: Float32[np.ndarray, "h w"] = np.linspace(0.25, 1.0, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(rgb.shape[:2])
        return RelativeDepthPrediction(
            disparity=disparity,
            depth=1.0 / disparity,
            K_33=np.array([[10.0, 0.0, 6.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            confidence=None,
        )


@pytest.mark.parametrize("name", relative_predictor_defaults)
def test_every_relative_predictor_is_selectable(name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dropdowns: tuple[gr.Dropdown, gr.Dropdown] = change_dropdown("Relative")
    assert all(name in [value for _, value in dropdown.choices] for dropdown in dropdowns)
    monkeypatch.setattr(type(relative_predictor_defaults[name]), "setup", lambda self, device: SyntheticRelativePredictor())
    rgb: UInt8[np.ndarray, "h w 3"] = np.zeros((8, 12, 3), dtype=np.uint8)
    recording_path: Path = tmp_path / "relative.rrd"
    recording_path.write_bytes(on_submit(rgb, False, 0.05, "Relative", name, name))
    reader: RrdReader = RrdReader(recording_path)
    entity_paths: set[str] = {
        str(chunk.entity_path) for recording in reader.recordings() for chunk in reader.stream(store=recording).to_chunks()
    }
    assert f"/{name}/camera/pinhole/depth" in entity_paths
    assert f"/{name}/camera/disparity" in entity_paths


@pytest.mark.parametrize("name", metric_predictor_defaults)
def test_every_metric_predictor_is_selectable(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    dropdowns: tuple[gr.Dropdown, gr.Dropdown] = change_dropdown("Metric")
    assert all(name in [value for _, value in dropdown.choices] for dropdown in dropdowns)
    monkeypatch.setattr(type(metric_predictor_defaults[name]), "setup", lambda self, device: SyntheticMetricPredictor())
    rgb: UInt8[np.ndarray, "h w 3"] = np.zeros((8, 12, 3), dtype=np.uint8)
    assert on_submit(rgb, False, 0.05, "Metric", name, name)


def test_cross_type_selection_is_rejected() -> None:
    rgb: UInt8[np.ndarray, "h w 3"] = np.zeros((8, 12, 3), dtype=np.uint8)
    with pytest.raises(gr.Error, match="not a relative depth predictor"):
        on_submit(rgb, False, 0.05, "Relative", "unidepth-metric", "zipdepth")
    with pytest.raises(gr.Error, match="not a metric depth predictor"):
        on_submit(rgb, False, 0.05, "Metric", "zipdepth", "unidepth-metric")
