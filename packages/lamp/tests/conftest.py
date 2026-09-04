"""Shared test markers for lamptrack."""

from pathlib import Path

import pytest
import torch

PACKAGE_DIR: Path = Path(__file__).parents[1]
CHECKPOINT_PATH: Path = PACKAGE_DIR / "data" / "checkpoints" / "lamp_smpl_aria_gen2.pt"
SMPL_MODEL_PATH: Path = PACKAGE_DIR / "data" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"
FIXTURE_DIR: Path = PACKAGE_DIR / "data" / "fixtures" / "test-library"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
"""Skip a test when CUDA is unavailable."""

slow = pytest.mark.slow
"""Mark tests that download or load model artifacts."""
