"""Committed activation ranges for the 768x576 fast pose export."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import orjson
from serde import SerdeError, from_dict, serde
from serde.json import to_json

from posekit.models.calibration_rrd import CalibrationRecording, CalibrationScalar, range_tables, read_calibration_rrd, write_calibration_rrd

FAST_POSE_INPUT_HW: tuple[int, int] = (768, 576)
"""Crop height and width the fast export runs at and its calibration was measured on."""
FAST_POSE_HEATMAP_HW: tuple[int, int] = (192, 144)
"""Heatmap height and width of the fast export: the input at stride 4."""
FAST_POSE_CALIBRATION_MODEL: str = "sapiens2-0.4b-nvfp4-768"
"""Model name and recording id of the committed ranges."""
CALIBRATION_PATH: Path = Path(__file__).with_name("fast_pose_768.rrd")
"""The committed ranges (layout: ``posekit.models.calibration_rrd``)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FastPoseCalibration:
    """Static activation ranges measured on person crops."""

    input_hw: tuple[int, int]
    """Crop height and width."""
    n_crops: int
    """Number of calibration crops."""
    input_amax: dict[str, float]
    """Absolute input maxima by quantized module name."""
    attention_amax: dict[str, float]
    """Absolute Q/K/V maxima after normalization and RoPE."""

    def __post_init__(self) -> None:
        if self.input_hw != FAST_POSE_INPUT_HW or self.n_crops < 1:
            raise ValueError(f"Fast pose calibration requires {FAST_POSE_INPUT_HW[0]}x{FAST_POSE_INPUT_HW[1]} person crops.")
        if len(self.attention_amax) != 72 or not self.input_amax:
            raise ValueError("Fast pose calibration must include all 24 attention blocks and module inputs.")
        if any(not math.isfinite(value) or value <= 0.0 for value in (*self.input_amax.values(), *self.attention_amax.values())):
            raise ValueError("Calibration maxima must be finite and positive.")


def calibration_digest(calibration: FastPoseCalibration) -> str:
    """SHA-256 of the ranges in the canonical JSON they shipped as before the .rrd; fast export caches are keyed on it."""
    return hashlib.sha256((to_json(calibration, option=orjson.OPT_INDENT_2) + "\n").encode()).hexdigest()


def write_calibration(path: Path, calibration: FastPoseCalibration) -> None:
    """Save measured ranges in the layout ``load_calibration`` reads."""
    height, width = calibration.input_hw
    properties: dict[str, CalibrationScalar] = {
        "model": FAST_POSE_CALIBRATION_MODEL, "input_height": height, "input_width": width, "n_crops": calibration.n_crops,
    }
    write_calibration_rrd(path, CalibrationRecording(properties, range_tables(calibration.input_amax, calibration.attention_amax)))


def load_calibration(path: Path = CALIBRATION_PATH) -> FastPoseCalibration:
    """Load the committed ranges, or an explicit newly measured .rrd; decode errors name the file."""
    recording: CalibrationRecording = read_calibration_rrd(path, FAST_POSE_CALIBRATION_MODEL)
    try:
        return from_dict(FastPoseCalibration, {
            "input_hw": [recording.properties["input_height"], recording.properties["input_width"]],
            "n_crops": recording.properties["n_crops"],
            "input_amax": recording.ranges("input_amax"),
            "attention_amax": recording.ranges("attention_amax"),
        })
    except (SerdeError, KeyError, ValueError) as error:
        raise ValueError(f"Invalid calibration {path}: {error!r}") from error
