"""Committed activation ranges for the 768x576 fast pose export."""

import math
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

from serde import SerdeError, serde
from serde.json import from_json


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
        if self.input_hw != (768, 576) or self.n_crops < 1:
            raise ValueError("Fast pose calibration requires 768x576 person crops.")
        if len(self.attention_amax) != 72 or not self.input_amax:
            raise ValueError("Fast pose calibration must include all 24 attention blocks and module inputs.")
        if any(not math.isfinite(value) or value <= 0.0 for value in (*self.input_amax.values(), *self.attention_amax.values())):
            raise ValueError("Calibration maxima must be finite and positive.")


def calibration_path() -> Path:
    """Return the package's committed calibration file."""
    return Path(__file__).parent.parent / "assets" / "fast_pose_768.json"


def load_calibration() -> FastPoseCalibration:
    """Load the strict calibration schema, naming the file on decode errors."""
    path: Path = calibration_path()
    try:
        return from_json(FastPoseCalibration, path.read_text())
    except (SerdeError, JSONDecodeError) as error:
        raise ValueError(f"Invalid calibration {path}: {error}") from error
