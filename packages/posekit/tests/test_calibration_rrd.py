"""Calibration records survive the static .rrd layout of posekit.models.calibration_rrd value for value."""

import hashlib
from pathlib import Path

import pytest
from serde.json import to_json

from posekit.models.calibration_rrd import CalibrationRecording, range_tables, read_calibration_rrd, write_calibration_rrd
from posekit.models.egohand_quantization import load_calibration, write_calibration


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


# SHA256 of to_json() of the EgoHand record loaded from the JSON asset it replaced. JSON floats
# are shortest round-trip reprs, so an equal digest means every value is bit-identical.
EGOHAND_NVFP4_SHA256: str = "687fcda618207bc8988393dae2702f8bccb86ce9a341443022f969f10a16a633"


def test_egohand_calibration_rrd_matches_the_former_json() -> None:
    calibration = load_calibration()
    assert hashlib.sha256(to_json(calibration).encode()).hexdigest() == EGOHAND_NVFP4_SHA256
    assert (len(calibration.frame_indices), len(calibration.excluded_mount_indices), calibration.n_crops) == (71, 64, 138)
    assert calibration.input_amax["encoder.blocks.0.attn.qkv"] == 44.403011322021484
    assert calibration.frame_sha256[-1] == "65f98660a3a111517a227033c93fae3383e3e93123f49c1f60fe71cfa8d93ccb"


def test_egohand_writer_round_trips_and_names_its_model(tmp_path: Path) -> None:
    hands = load_calibration()
    write_calibration(tmp_path / "hands.rrd", hands)
    assert load_calibration(tmp_path / "hands.rrd") == hands
    with pytest.raises(ValueError, match="egohand"):
        read_calibration_rrd(tmp_path / "hands.rrd", "other")
