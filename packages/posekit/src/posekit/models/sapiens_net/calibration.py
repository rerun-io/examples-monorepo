"""Committed activation ranges for the 768x576 fast pose export."""

import math
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from serde import SerdeError, serde
from serde.json import from_json

FAST_POSE_INPUT_HW: tuple[int, int] = (768, 576)
"""Crop height and width the fast export runs at and its calibration was measured on."""
FAST_POSE_HEATMAP_HW: tuple[int, int] = (192, 144)
"""Heatmap height and width of the fast export: the input at stride 4."""


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


def calibration_path() -> Path:
    """Return the committed calibration file."""
    return Path(__file__).with_name("fast_pose_768.json")


def load_calibration() -> FastPoseCalibration:
    """Load the strict calibration schema, naming the file on decode errors."""
    path: Path = calibration_path()
    try:
        return from_json(FastPoseCalibration, path.read_text())
    except (SerdeError, JSONDecodeError) as error:
        raise ValueError(f"Invalid calibration {path}: {error}") from error
