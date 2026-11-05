"""EgoHand-v1 activation ranges: the calibration record, its loader and artifact name, and their measurement."""

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from jaxtyping import Float, UInt8
from serde import SerdeError, from_dict, serde
from serde.json import to_json
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from posekit.cameras import CalibratedCamera
from posekit.models.calibration_rrd import (
    CalibrationColumn,
    CalibrationRecording,
    CalibrationScalar,
    range_tables,
    read_calibration_rrd,
    write_calibration_rrd,
)

EGOHAND_CALIBRATION_MODEL: str = "egohand"
"""Model name and recording id of the EgoHand calibration .rrd."""

DINO_BLOCKS: int = 32
"""Transformer blocks in EgoHand's DINOv3 encoder; each has one native Attention op."""
LINEARS_PER_BLOCK: int = 5
"""Linears per block: attention qkv and proj, and the gated MLP's w1, w2 and w3."""
CALIBRATED_LINEARS: int = DINO_BLOCKS * LINEARS_PER_BLOCK
"""Linear inputs with a calibrated range."""
CALIBRATED_ATTENTION_INPUTS: int = DINO_BLOCKS * 3
"""Query, key and value inputs with a calibrated range."""


def validate_fp8_tail_blocks(fp8_tail_blocks: int) -> None:
    """Raise ValueError unless ``fp8_tail_blocks`` names 0 to all of the DINO blocks."""
    if not 0 <= fp8_tail_blocks <= DINO_BLOCKS:
        raise ValueError(f"fp8_tail_blocks must be between 0 and {DINO_BLOCKS}.")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EgoHandCalibration:
    """Activation ranges and the exact SelfCap calibration split."""

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
    """Load the committed split, or an explicit newly measured calibration .rrd (layout: ``posekit.models.calibration_rrd``)."""
    source: Path = path if path is not None else Path(__file__).parents[1] / "assets" / "egohand_nvfp4.rrd"
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


@torch.inference_mode()
def measure_calibration(
    frames: Iterable[tuple[int, UInt8[Tensor, "h w 3"]]], camera: CalibratedCamera, *, segment: str,
    excluded_mount_indices: Sequence[int] = (), excluded_frame_sha256: Sequence[str] = (), assets: Path | None = None,
) -> EgoHandCalibration:
    """Measure released BF16 activations through the ordinary detector and calibrated pose roles.

    Frame indices and decoded RGB hashes are checked before model loading. Hooks
    exist only during calibration and are removed even when inference fails.
    The caller supplies catalog frames and the held-out evaluation split.
    """
    from egohand_dinov3.layers.attention import SelfAttention
    from posekit.models.egohand import selfcap_egohand_configs

    selected: list[tuple[int, UInt8[Tensor, "h w 3"]]] = list(frames)
    indices: list[int] = [index for index, _ in selected]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Calibration needs a non-empty set of distinct frame indices.")
    if set(indices).intersection(excluded_mount_indices):
        raise ValueError("Calibration frames overlap the mount-fit evaluation sample.")
    hashes: list[str] = [hashlib.sha256(frame.cpu().numpy().tobytes()).hexdigest() for _, frame in selected]
    if set(hashes).intersection(excluded_frame_sha256):
        raise ValueError("Calibration contains a decoded evaluation frame.")
    detector_config, predictor_config = selfcap_egohand_configs(assets)
    detector = detector_config.setup()
    predictor = replace(predictor_config, backbone_onnx=None).setup()
    backbone: nn.Module = predictor.predictor.model.backbone
    maxima: dict[str, float] = {}
    attention: dict[str, float] = {}
    handles: list[RemovableHandle] = []
    originals: list[tuple[SelfAttention, Callable[..., Tensor]]] = []

    def linear_hook(name: str) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def record(module: nn.Module, args: tuple[Tensor, ...]) -> None:
            maxima[name] = max(maxima.get(name, 0.0), float(args[0].abs().amax()))

        return record

    def attention_hook(name: str, module: SelfAttention, original: Callable[..., Tensor]) -> Callable[..., Tensor]:
        def record(
            qkv: Float[Tensor, "b n c"], attn_bias: Tensor | None = None, rope: tuple[Tensor, Tensor] | None = None,
        ) -> Tensor:
            batch: int = qkv.shape[0]
            tokens: int = qkv.shape[1]
            channels: int = module.qkv.in_features
            q, k, v = (item.transpose(1, 2) for item in qkv.reshape(batch, tokens, 3, module.num_heads, channels // module.num_heads).unbind(2))
            if rope is not None:
                q, k = module.apply_rope(q, k, rope)
            for tag, value in zip("qkv", (q, k, v), strict=True):
                key: str = f"{name}.{tag}"
                attention[key] = max(attention.get(key, 0.0), float(value.abs().amax()))
            return original(qkv, attn_bias, rope)

        return record

    crops: int = 0
    try:
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_pre_hook(linear_hook(name)))
            if isinstance(module, SelfAttention):
                originals.append((module, module.compute_attention))
                module.compute_attention = attention_hook(name, module, module.compute_attention)
        for index, frame in selected:
            batch: UInt8[Tensor, "1 h w 3"] = frame.unsqueeze(0)
            detections = detector(batch)
            if detections.num_detections:
                predictor(batch, detections, camera)
                crops += detections.num_detections
            print(f"[egohand-calibrate] frame {index}: {detections.num_detections} crops", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        for module, original in originals:
            module.compute_attention = original
    return EgoHandCalibration(segment, indices, list(excluded_mount_indices), hashes, crops, maxima, attention)
