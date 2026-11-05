"""The committed Sapiens2 nvfp4-768 activation ranges: what the .rrd holds, and the writer/reader round trip."""

from pathlib import Path

import pytest

from posekit.models.calibration_rrd import CalibrationRecording, write_calibration_rrd
from posekit.models.sapiens_net.calibration import (
    FAST_POSE_CALIBRATION_MODEL,
    FastPoseCalibration,
    calibration_digest,
    load_calibration,
    write_calibration,
)

RETIRED_JSON_SHA256: str = "b41ccf31b5fc1cf4da06ebf405ba3bbd553967f0ead6ca7907817b0bf50a6bd7"
"""SHA-256 of fast_pose_768.json, the file the .rrd replaced."""


def test_committed_ranges_equal_the_retired_json() -> None:
    """The digest is the canonical JSON of the loaded ranges, so equal to the old file's hash means every key, order and float is equal."""
    calibration: FastPoseCalibration = load_calibration()
    assert (calibration.input_hw, calibration.n_crops) == ((768, 576), 180)
    assert (len(calibration.input_amax), len(calibration.attention_amax)) == (151, 72)
    assert calibration.input_amax["backbone.patch_embed.projection"] == 2.640000581741333
    assert calibration.attention_amax["backbone.blocks.0.attn.q"] == 9.602294921875
    assert calibration_digest(calibration) == RETIRED_JSON_SHA256


def test_round_trip_is_exact(tmp_path: Path) -> None:
    attention: dict[str, float] = {f"blocks.{block}.attn.{tag}": 1.0 + block / 3.0 for block in range(24) for tag in "qkv"}
    calibration = FastPoseCalibration((768, 576), 3, {"patch_embed": 0.1, "head": 1e-30, "ffn": 116.5}, attention)
    path: Path = tmp_path / "ranges.rrd"
    write_calibration(path, calibration)
    assert load_calibration(path) == calibration


def test_missing_ranges_name_the_source(tmp_path: Path) -> None:
    path: Path = tmp_path / "ranges.rrd"
    properties_only = CalibrationRecording({"model": FAST_POSE_CALIBRATION_MODEL, "input_height": 768, "input_width": 576, "n_crops": 3}, {})
    write_calibration_rrd(path, properties_only)
    with pytest.raises(ValueError, match="ranges.rrd"):
        load_calibration(path)
