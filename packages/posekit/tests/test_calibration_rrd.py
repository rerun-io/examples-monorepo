"""Calibration records survive the static .rrd layout of posekit.models.calibration_rrd value for value."""

from pathlib import Path

import pytest

from posekit.models.calibration_rrd import CalibrationRecording, range_tables, read_calibration_rrd, write_calibration_rrd


def example_recording() -> CalibrationRecording:
    return CalibrationRecording(
        {"model": "example", "revision": "abc123", "n_crops": 3, "headroom": 1.5},
        {
            "/ranges/input_amax": {"module": ["blocks.0.qkv", "blocks.0.mlp", "head"], "amax": [44.403011322021484, 1e-30, 516.0]},
            "/split/calibration": {"source": ["segment", "segment"], "index": [2200, 8150], "sha256": ["a" * 64, "b" * 64]},
        },
    )


def test_round_trip_is_exact(tmp_path: Path) -> None:
    write_calibration_rrd(tmp_path / "example.rrd", example_recording())
    assert read_calibration_rrd(tmp_path / "example.rrd", "example") == example_recording()


def test_range_tables_round_trip_in_order(tmp_path: Path) -> None:
    input_amax: dict[str, float] = {"blocks.1.mlp": 516.0, "blocks.0.qkv": 44.403011322021484, "head": 1e-30}
    attention_amax: dict[str, float] = {"blocks.0.attn.q": 17.0}
    write_calibration_rrd(tmp_path / "ranges.rrd", CalibrationRecording({"model": "example"}, range_tables(input_amax, attention_amax)))
    recording: CalibrationRecording = read_calibration_rrd(tmp_path / "ranges.rrd", "example")
    assert list(recording.ranges("input_amax").items()) == list(input_amax.items())
    assert recording.ranges("attention_amax") == attention_amax


def test_reader_rejects_another_models_recording(tmp_path: Path) -> None:
    write_calibration_rrd(tmp_path / "example.rrd", example_recording())
    with pytest.raises(ValueError, match="example.rrd"):
        read_calibration_rrd(tmp_path / "example.rrd", "other")


def test_writer_rejects_columns_of_unequal_length(tmp_path: Path) -> None:
    recording = CalibrationRecording({"model": "example"}, {"/ranges/input_amax": {"module": ["a", "b"], "amax": [1.0]}})
    with pytest.raises(ValueError, match="equal lengths"):
        write_calibration_rrd(tmp_path / "example.rrd", recording)
