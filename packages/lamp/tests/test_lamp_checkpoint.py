"""Smoke tests for the released non-commercial LAMP checkpoint and SMPL file."""

import hashlib
from pathlib import Path

import pytest
import torch
from conftest import CHECKPOINT_PATH, SMPL_MODEL_PATH, requires_cuda, slow

from lamptrack.third_party.lamp.models.model_loader import build_lampnet_from_checkpoint

LAMP_CHECKPOINT_SHA256: str = "6b5b1430ed2bdad089e579dfe53d98aa397535ca7a869570eda2de1ed82973e5"
SMPL_MODEL_SHA256: str = "213aa58a3c58c7c2dcb2e1c83b9dde0190516861fb2fc20e9476bf9b52c64a7e"


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@slow
@requires_cuda
def test_released_checkpoint_loads_with_neutral_smpl() -> None:
    """The pinned plain state dict loads with only the two runtime buffers absent."""
    if not CHECKPOINT_PATH.is_file():
        pytest.skip(f"LAMP checkpoint missing: run `pixi run -e lamp-dev lamp-download-artifacts` ({CHECKPOINT_PATH})")
    if not SMPL_MODEL_PATH.is_file():
        pytest.skip(f"SMPL model missing: run `pixi run -e lamp-dev lamp-download-artifacts` ({SMPL_MODEL_PATH})")
    assert _sha256(CHECKPOINT_PATH) == LAMP_CHECKPOINT_SHA256
    assert _sha256(SMPL_MODEL_PATH) == SMPL_MODEL_SHA256

    model = build_lampnet_from_checkpoint(CHECKPOINT_PATH, SMPL_MODEL_PATH, torch.device("cuda"))
    assert model.training is False
    assert tuple(model.smpl.faces.shape) == (13_776, 3)
