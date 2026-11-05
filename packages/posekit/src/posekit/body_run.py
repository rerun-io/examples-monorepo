"""Reproducible body model settings at accuracy and benchmark report boundaries."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from serde import serde
from serde.json import to_json

from posekit.models.sam3d_body import Sam3dBodyConfig
from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig
from posekit.models.sam3d_body_quantization import BodyCalibration, BodyPrecisionPlan, body_backbone_path, load_body_calibration


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class BodyRunIdentity:
    """Effective settings and resolved exports for one measured body role."""

    config: Sam3dBodyFastConfig | Sam3dBodyConfig
    """Complete effective config, including checkpoint, plans, capture and TensorRT profile."""
    calibration_sha256: str | None
    """SHA256 of the canonical calibration JSON; None when no quantized engine is used."""
    backbone_artifacts: tuple[Path, ...]
    """Resolved body and hand ONNX paths, whose names identify calibration and precision plans."""
    export_sha256: dict[str, str | None]
    """ONNX graph digests, not external weight digests; null means that export was absent."""
    code_revision: str | None
    """Git revision of the source checkout, or None for an installation outside Git."""
    source_dirty: bool | None
    """Whether tracked source differs from that revision; None when Git identity is unavailable."""


def body_run_identity(config: Sam3dBodyConfig) -> BodyRunIdentity:
    """Identify the effective configuration without loading weights or running inference."""
    calibration_digest: str | None = None
    backbones: tuple[Path, ...] = ()
    exports: list[Path] = []
    if isinstance(config, Sam3dBodyFastConfig):
        if config.backbone_onnx is not None:
            if config.backbone_precision == "bf16":
                backbones = (config.backbone_source_onnx,)
            else:
                calibration: BodyCalibration = load_body_calibration(config.calibration)
                calibration_digest = hashlib.sha256(to_json(calibration).encode()).hexdigest()
                plans: tuple[BodyPrecisionPlan, BodyPrecisionPlan] = config.precision_plans(calibration)
                backbones = tuple(body_backbone_path(
                    config.backbone_onnx, calibration, config.fp8_tail_blocks,
                    plan=plan, source_version=config.backbone_source_onnx.name,
                ) for plan in (plans if config.mode == "full" else plans[:1]))
            exports.extend((config.backbone_source_onnx, *backbones))
        if config.decoder_onnx_dir is not None:
            exports.extend(config.decoder_onnx_dir / f"{branch}-{index}.onnx" for branch in ("body", "hand") for index in range(6))
    hashes: dict[str, str | None] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in exports}
    revision: str | None = None
    dirty: bool | None = None
    try:
        head: subprocess.CompletedProcess[str] = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if head.returncode == 0:
            revision = head.stdout.strip()
            status: subprocess.CompletedProcess[str] = subprocess.run(["git", "-C", str(Path(__file__).parent), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False)
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except FileNotFoundError:
        pass
    return BodyRunIdentity(config, calibration_digest, backbones, hashes, revision, dirty)
