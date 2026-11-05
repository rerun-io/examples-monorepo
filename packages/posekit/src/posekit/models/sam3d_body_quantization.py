"""SAM body activation ranges with a checked, reproducible calibration split."""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import torch
from jaxtyping import UInt8
from serde import SerdeError, from_dict, serde
from serde.json import to_json
from torch import Tensor

from posekit.models.calibration_rrd import (
    CalibrationColumn,
    CalibrationRecording,
    CalibrationScalar,
    range_tables,
    read_calibration_rrd,
    write_calibration_rrd,
)
from posekit.models.dinov3_calibration import CALIBRATED_ATTENTION_INPUTS, CALIBRATED_LINEARS
from posekit.models.sam3d_body import Sam3dBodyConfig
from posekit.predictions import BoxDetections

BodyBackbonePrecision: TypeAlias = Literal["nvfp4", "bf16"]
"""Calibrated mixed NVFP4/FP8, or the unchanged BF16 ONNX export."""
BODY_CALIBRATION_MODEL: str = "sam3d-body"
"""Model name and recording id of the SAM body calibration .rrd."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CalibrationFrame:
    """One source index plus a digest of its decoded RGB pixels."""

    source: str
    """Catalog segment and video entity, or a local evaluation asset identity."""
    index: int
    """Zero-based decoded frame index within that source."""
    sha256: str
    """SHA256 of the contiguous uint8 RGB pixels."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyCalibrationSplit:
    """Disjoint source indices and pixel contents, enforced before calibration."""

    calibration: list[CalibrationFrame]
    """Frames used to observe activations."""
    evaluation: list[CalibrationFrame]
    """Frames reserved for accuracy measurement."""

    def __post_init__(self) -> None:
        if not self.calibration or not self.evaluation:
            raise ValueError("Calibration and evaluation must both contain frames.")
        selected = {(frame.source, frame.index) for frame in self.calibration}
        held_out = {(frame.source, frame.index) for frame in self.evaluation}
        if len(selected) != len(self.calibration) or selected.intersection(held_out):
            raise ValueError("Calibration source indices must be unique and disjoint from evaluation.")
        if {frame.sha256 for frame in self.calibration}.intersection(frame.sha256 for frame in self.evaluation):
            raise ValueError("Calibration contains a decoded evaluation frame.")




@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyCalibration:
    """SAM-specific 512-pixel activation ranges; never reuse EgoHand ranges."""

    split: BodyCalibrationSplit
    """Audited calibration and held-out frame identities."""
    revision: str
    """Checkpoint revision used for measurement."""
    body_crops: int
    """Number of body crops observed; each also contributes two hand crops."""
    input_amax: dict[str, float]
    """Input ranges for the 160 transformer linears."""
    attention_amax: dict[str, float]
    """Post-RoPE Q/K/V ranges for the 32 attention blocks."""
    crop_size: int = 512
    """Square calibration crop side in pixels."""

    def __post_init__(self) -> None:
        if self.crop_size != 512 or self.body_crops < 1 or len(self.input_amax) != CALIBRATED_LINEARS or len(self.attention_amax) != CALIBRATED_ATTENTION_INPUTS:
            raise ValueError(
                f"SAM calibration requires body and hand crops at 512 pixels, {CALIBRATED_LINEARS} linear ranges and {CALIBRATED_ATTENTION_INPUTS} attention ranges."
            )
        if any(not math.isfinite(value) or value <= 0 for value in (*self.input_amax.values(), *self.attention_amax.values())):
            raise ValueError("Calibration ranges must be finite and positive.")


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One in-memory calibration input, obtained from a catalog or evaluation source."""

    source: str
    """Video source identity."""
    index: int
    """Decoded source frame index."""
    image: UInt8[Tensor, "h w 3"]
    """Decoded RGB frame."""
    detections: BoxDetections
    """Person boxes for this one frame, all with frame index zero."""

    def reference(self) -> CalibrationFrame:
        """Hash decoded pixels, independently of source/container metadata."""
        return CalibrationFrame(self.source, self.index, hashlib.sha256(self.image.cpu().contiguous().numpy().tobytes()).hexdigest())


def load_body_calibration(path: Path | None = None) -> BodyCalibration:
    """Load the committed split, or validate an explicit calibration .rrd (layout: ``posekit.models.calibration_rrd``)."""
    source: Path = path if path is not None else Path(__file__).parents[1] / "assets" / "sam3d_body_nvfp4.rrd"
    recording: CalibrationRecording = read_calibration_rrd(source, BODY_CALIBRATION_MODEL)
    tables: dict[str, dict[str, CalibrationColumn]] = recording.tables
    try:
        split: dict[str, list[dict[str, CalibrationScalar]]] = {}
        for name in ("calibration", "evaluation"):
            frames: dict[str, CalibrationColumn] = tables[f"/split/{name}"]
            split[name] = [{"source": frame, "index": index, "sha256": sha256}
                           for frame, index, sha256 in zip(frames["source"], frames["index"], frames["sha256"], strict=True)]
        return from_dict(BodyCalibration, {
            "split": split,
            "input_amax": recording.ranges("input_amax"),
            "attention_amax": recording.ranges("attention_amax"),
            "revision": recording.properties["revision"], "body_crops": recording.properties["body_crops"], "crop_size": recording.properties["crop_size"],
        })
    except (SerdeError, KeyError, ValueError) as error:
        raise ValueError(f"Invalid SAM body calibration {source}: {error!r}") from error


def write_body_calibration(path: Path, calibration: BodyCalibration) -> None:
    """Save a measured calibration in the layout ``load_body_calibration`` reads."""
    tables: dict[str, dict[str, CalibrationColumn]] = {}
    for name, frames in (("calibration", calibration.split.calibration), ("evaluation", calibration.split.evaluation)):
        tables[f"/split/{name}"] = {"source": [frame.source for frame in frames], "index": [frame.index for frame in frames], "sha256": [frame.sha256 for frame in frames]}
    tables |= range_tables(calibration.input_amax, calibration.attention_amax)
    properties: dict[str, CalibrationScalar] = {
        "model": BODY_CALIBRATION_MODEL, "revision": calibration.revision, "body_crops": calibration.body_crops, "crop_size": calibration.crop_size,
    }
    write_calibration_rrd(path, CalibrationRecording(properties, tables))


def body_backbone_path(
    base: Path, calibration: BodyCalibration, fp8_tail_blocks: int, *, source_version: str = "native-bf16-v3",
) -> Path:
    """Identify the calibrated tail and source export independently of engine tuning."""
    digest: str = hashlib.sha256((to_json(calibration) + source_version + str(fp8_tail_blocks)).encode()).hexdigest()[:16]
    suffix: str = f"-{digest}-fp8tail{fp8_tail_blocks}"
    return base if base.stem.endswith(suffix) else base.with_stem(f"{base.stem}{suffix}")








@torch.inference_mode()
def measure_body_calibration(
    calibration: Sequence[CalibrationSample], evaluation: Sequence[CalibrationFrame],
    config: Sam3dBodyConfig | None = None,
) -> BodyCalibration:
    """Observe SAM's released full inference after checking both forms of split overlap."""
    from posekit.models.dinov3_calibration import DinoActivationObserver

    split = BodyCalibrationSplit([sample.reference() for sample in calibration], list(evaluation))
    config = Sam3dBodyConfig() if config is None else config
    if config.mode != "full":
        raise ValueError("Calibration must observe full mode, including both hand crops.")
    predictor = config.setup()
    crops: int = 0
    with DinoActivationObserver(predictor.estimator.model.backbone) as observer:
        for sample in calibration:
            predictor(sample.image[None].cuda(), sample.detections)
            crops += sample.detections.num_detections
            print(f"[body-calibrate] {sample.source} frame {sample.index}: {sample.detections.num_detections} bodies", flush=True)
    linear, attention = observer.ranges()
    return BodyCalibration(split, config.hf_revision, crops, linear, attention)
