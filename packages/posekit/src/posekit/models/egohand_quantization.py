"""EgoHand-v1 activation ranges: the calibration record, its loader, writer and artifact name."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from serde import SerdeError, from_dict, serde
from serde.json import to_json

from posekit.models.calibration_rrd import (
    CalibrationColumn,
    CalibrationRecording,
    CalibrationScalar,
    range_tables,
    read_calibration_rrd,
    write_calibration_rrd,
)
from posekit.models.dinov3_calibration import CALIBRATED_ATTENTION_INPUTS, CALIBRATED_LINEARS, DINO_BLOCKS

EGOHAND_CALIBRATION_MODEL: str = "egohand"
"""Model name and recording id of the EgoHand calibration .rrd."""


def validate_fp8_tail_blocks(fp8_tail_blocks: int) -> None:
    """Raise ValueError unless ``fp8_tail_blocks`` names 0 to all of the DINO blocks."""
    if not 0 <= fp8_tail_blocks <= DINO_BLOCKS:
        raise ValueError(f"fp8_tail_blocks must be between 0 and {DINO_BLOCKS}.")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EgoHandCalibration:
    """Activation ranges and the exact calibration split."""

    segment: str
    """Source catalog segment; evaluation uses separate frame indices or datasets."""
    frame_indices: list[int]
    """Source OAK-left video frame indices."""
    excluded_mount_indices: list[int]
    """The segment's mount-fit sample, excluded before calibration."""
    frame_sha256: list[str]
    """Hashes of decoded RGB frames, checked against every benchmark frame."""
    n_crops: int
    """Detected hand crops used to measure ranges."""
    input_amax: dict[str, float]
    """Absolute activation maxima at every DINO linear input (CALIBRATED_LINEARS)."""
    attention_amax: dict[str, float]
    """Absolute Q/K/V maxima after RoPE, for every attention block (CALIBRATED_ATTENTION_INPUTS)."""

    def __post_init__(self) -> None:
        if self.n_crops < 1 or len(self.input_amax) != CALIBRATED_LINEARS or len(self.attention_amax) != CALIBRATED_ATTENTION_INPUTS:
            raise ValueError(f"EgoHand calibration requires all {CALIBRATED_LINEARS} linears and {DINO_BLOCKS} Q/K/V attention inputs.")
        if not self.frame_indices or len(self.frame_indices) != len(self.frame_sha256):
            raise ValueError("Each calibration frame requires its source index and SHA256 hash.")
        if set(self.frame_indices).intersection(self.excluded_mount_indices):
            raise ValueError("Calibration frames overlap the mount-fit evaluation sample.")
        if any(not math.isfinite(value) or value <= 0.0 for value in (*self.input_amax.values(), *self.attention_amax.values())):
            raise ValueError("Calibration ranges must be finite and positive.")


def load_calibration(path: Path | None = None) -> EgoHandCalibration:
    """Load the committed five-dataset catalog calibration, or an explicit calibration .rrd (layout: ``posekit.models.calibration_rrd``)."""
    source: Path = path if path is not None else Path(__file__).parents[1] / "assets" / "egohand_catalog.rrd"
    recording: CalibrationRecording = read_calibration_rrd(source, EGOHAND_CALIBRATION_MODEL)
    tables: dict[str, dict[str, CalibrationColumn]] = recording.tables
    try:
        segment: CalibrationScalar = recording.properties["segment"]
        if any(frame != segment for name in ("calibration", "excluded_mount") for frame in tables[f"/split/{name}"]["source"]):
            raise ValueError(f"Every split frame must come from segment {segment}.")
        return from_dict(EgoHandCalibration, {
            "segment": segment,
            "frame_indices": tables["/split/calibration"]["index"],
            "excluded_mount_indices": tables["/split/excluded_mount"]["index"],
            "frame_sha256": tables["/split/calibration"]["sha256"],
            "n_crops": recording.properties["n_crops"],
            "input_amax": recording.ranges("input_amax"),
            "attention_amax": recording.ranges("attention_amax"),
        })
    except (SerdeError, KeyError, ValueError) as error:
        raise ValueError(f"Invalid EgoHand calibration {source}: {error!r}") from error


def write_calibration(path: Path, calibration: EgoHandCalibration) -> None:
    """Save a measured calibration in the layout ``load_calibration`` reads; every split frame names the segment."""
    tables: dict[str, dict[str, CalibrationColumn]] = {
        "/split/calibration": {
            "source": [calibration.segment] * len(calibration.frame_indices), "index": calibration.frame_indices, "sha256": calibration.frame_sha256,
        },
        "/split/excluded_mount": {"source": [calibration.segment] * len(calibration.excluded_mount_indices), "index": calibration.excluded_mount_indices},
        **range_tables(calibration.input_amax, calibration.attention_amax),
    }
    properties: dict[str, CalibrationScalar] = {"model": EGOHAND_CALIBRATION_MODEL, "segment": calibration.segment, "n_crops": calibration.n_crops}
    write_calibration_rrd(path, CalibrationRecording(properties, tables))


def calibrated_backbone_path(base: Path, calibration: EgoHandCalibration, *, fp8_tail_blocks: int = 0) -> Path:
    """Name the exported backbone by its calibration contents, including the held-out split."""
    validate_fp8_tail_blocks(fp8_tail_blocks)
    digest: str = hashlib.sha256(to_json(calibration).encode()).hexdigest()[:16]
    precision: str = f"-fp8tail{fp8_tail_blocks}" if fp8_tail_blocks else ""
    return base.with_stem(f"{base.stem}-{digest}{precision}")
