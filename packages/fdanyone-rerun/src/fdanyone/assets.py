"""Locate the model files used by 4DAnyone.

Every published file is anchored by one immutable Hugging Face revision. In the
monorepo the pixi download tasks (``fdanyone-download``) fetch them; the resolvers
here only locate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fdanyone.errors import AssetError
from fdanyone.io import sha256_file

HF_REPO_ID = "AntResearch/4DAnyone"
HF_REVISION = "4c80e87b805a5f8461cf339cdbe2fb4249e585aa"

BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BIREFNET_DIR = "birefnet"
BIREFNET_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
)

CHECKPOINT = "4danyone/model.safetensors"
MHR70_REGRESSOR = "4danyone/smplx_to_goliath70.pt"
WAN_VAE = "4danyone/Wan2.2_VAE.pth"
PROMPT_CONTEXT = "4danyone/prompt_context.safetensors"

TURBO_LORA = "4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
TURBO_LORA_NAME = "wan22_ti2v_5b_turbo_lora"
TURBO_LORA_SIZE_BYTES = 332_348_584
TURBO_LORA_SHA256 = "0ace5244e3d1256f884662c261b017249796cf5b95f05d5ed93cc02a478967b8"

GVHMR_CHECKPOINT = "gvhmr/gvhmr_siga24_release.ckpt"
HMR2_CHECKPOINT = "gvhmr/epoch=10-step=25000.ckpt"
VITPOSE_CHECKPOINT = "gvhmr/vitpose-h-multi-coco.pth"
YOLO_CHECKPOINT = "gvhmr/yolov8x.pt"

TAEW2_2 = "taehv/taew2_2.pth"
TAEW2_2_URL = "https://raw.githubusercontent.com/madebyollin/taehv/e743234f3217ab3d1570f65642ab06596d1bd7c5/taew2_2.pth"
TAEW2_2_SHA256 = "d053e216ca50e2bb837bbcd79b85f0366bea00e5938025572382a773b74c559a"

SMPLX_MODEL = "body_models/smplx/SMPLX_NEUTRAL.npz"


# Upstream GVHMR resolves its model files relative to its own checkout, so the
# install commands link each downloaded file to the location GVHMR expects.
GVHMR_LINKS = (
    (GVHMR_CHECKPOINT, "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"),
    (HMR2_CHECKPOINT, "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"),
    (VITPOSE_CHECKPOINT, "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    (YOLO_CHECKPOINT, "inputs/checkpoints/yolo/yolov8x.pt"),
    (SMPLX_MODEL, "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"),
)


@dataclass(frozen=True)
class BaseAssets:
    vae: Path
    prompt_context: Path
    tiny_decoder: Path | None = None


def _require_file(path: Path, label: str, command: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AssetError(f"{label} does not exist: {resolved}. Run `python {command}` to install it.")
    return resolved


def resolve_checkpoint(path: str | Path | None = None, model_dir: str | Path = "models") -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise AssetError(f"Checkpoint override does not exist: {resolved}")
        return resolved
    return _require_file(Path(model_dir) / CHECKPOINT, "Checkpoint", "scripts/download_model.py")


def resolve_turbo_lora(model_dir: str | Path = "models") -> Path:
    """Resolve and authenticate the exact Wan2.2 5B Turbo LoRA."""

    resolved = _require_file(Path(model_dir) / TURBO_LORA, "Turbo LoRA", "scripts/download_model.py")
    size = resolved.stat().st_size
    if size != TURBO_LORA_SIZE_BYTES:
        raise AssetError(f"Turbo LoRA size mismatch: {size} != {TURBO_LORA_SIZE_BYTES} bytes ({resolved})")
    digest = sha256_file(resolved)
    if digest != TURBO_LORA_SHA256:
        raise AssetError(f"Turbo LoRA SHA-256 mismatch: {digest} != {TURBO_LORA_SHA256} ({resolved})")
    return resolved


def resolve_regressor(path: str | Path | None = None, model_dir: str | Path = "models") -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise AssetError(f"MHR70 regressor override does not exist: {resolved}")
        return resolved
    return _require_file(Path(model_dir) / MHR70_REGRESSOR, "MHR70 regressor", "scripts/download_model.py")


def resolve_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser() / BIREFNET_DIR
    for relative in BIREFNET_FILES:
        _require_file(root / relative, "BiRefNet file", "scripts/download_model.py")
    return root.resolve()


def resolve_base_assets(model_dir: str | Path = "models") -> BaseAssets:
    """Resolve the local VAE and frozen prompt conditioning."""

    from fdanyone.config import INFERENCE

    root = Path(model_dir).expanduser()
    return BaseAssets(
        vae=_require_file(root / WAN_VAE, "VAE", "scripts/download_model.py"),
        prompt_context=_require_file(root / PROMPT_CONTEXT, "Prompt conditioning", "scripts/download_model.py"),
        tiny_decoder=(
            _require_file(root / TAEW2_2, "TAEW2.2 decoder", "scripts/download_model.py")
            if INFERENCE.target_decoder == "taew2_2"
            else None
        ),
    )
