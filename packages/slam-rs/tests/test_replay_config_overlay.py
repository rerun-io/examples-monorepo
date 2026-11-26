"""The replay's --config-overlay merges into the VIO config's value0 (no data, no GPU)."""

import json

import pytest

from slam_rs.apis.replay import overlay_config


def test_overlay_updates_a_section_and_replaces_scalars() -> None:
    text: str = json.dumps({"value0": {"config.vio_max_iterations": 7, "backend.loop_closing": {"pnp_min_inliers": 10, "frame_time_margin_s": 20.0}}})
    merged: dict = json.loads(overlay_config(text, '{"backend.loop_closing": {"frame_time_margin_s": 120.0}, "config.vio_max_iterations": 5}'))
    assert merged["value0"] == {"config.vio_max_iterations": 5, "backend.loop_closing": {"pnp_min_inliers": 10, "frame_time_margin_s": 120.0}}


def test_overlay_adds_a_missing_section_whole() -> None:
    merged: dict = json.loads(overlay_config(json.dumps({"value0": {}}), '{"backend.loop_closing": {"frame_time_margin_s": 120.0}}'))
    assert merged["value0"] == {"backend.loop_closing": {"frame_time_margin_s": 120.0}}


def test_overlay_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        overlay_config(json.dumps({"value0": {}}), "[1, 2]")
