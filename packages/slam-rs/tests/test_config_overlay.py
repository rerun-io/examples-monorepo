"""``--config-overlay`` merges a JSON object into the VIO config's ``value0`` (no data, no GPU)."""

import json

import pytest

from slam_rs import _core
from slam_rs.config import load_slam_config, overlay_config_text


def test_an_overlay_updates_a_section_and_replaces_a_scalar() -> None:
    text: str = _core.VioConfig.from_json(load_slam_config().robocap_config_text("fast")).to_json()
    loop_closing: dict = json.loads(text)["value0"]["backend.loop_closing"]
    merged: dict = json.loads(overlay_config_text(text, '{"backend.loop_closing": {"frame_time_margin_s": 120.0}, "config.vio_max_iterations": 5}'))["value0"]
    assert merged["backend.loop_closing"] == {**loop_closing, "frame_time_margin_s": 120.0}
    assert merged["config.vio_max_iterations"] == 5


def test_an_overlay_adds_a_section_the_text_leaves_to_its_defaults() -> None:
    text: str = load_slam_config().robocap_config_text("fast")
    assert "backend.loop_closing" not in json.loads(text)["value0"]
    merged: dict = json.loads(overlay_config_text(text, '{"backend.loop_closing": {"frame_time_margin_s": 120.0}}'))["value0"]
    assert merged["backend.loop_closing"] == {"frame_time_margin_s": 120.0}


def test_an_overlay_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        overlay_config_text(load_slam_config().robocap_config_text("fast"), "[1, 2]")


def test_an_overlay_the_estimator_cannot_read_is_refused() -> None:
    with pytest.raises(ValueError):
        overlay_config_text(load_slam_config().robocap_config_text("fast"), '{"config.vio_max_iterations": "many"}')
