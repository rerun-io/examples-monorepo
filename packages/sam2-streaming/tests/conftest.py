"""Shared fixtures: the EfficientTAM-ti predictor (Hugging Face hub checkpoint, CUDA) and a two-lens synthetic scene."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictor


@pytest.fixture(scope="session")
def efficienttam_ti_files() -> tuple[str, str]:
    """(config name, checkpoint path) of EfficientTAM-ti at 512 px, mamma's and posekit's default tracker."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (EfficientTAM-ti from the Hugging Face hub)")
    from huggingface_hub import hf_hub_download

    checkpoint: str = hf_hub_download(
        repo_id="pablovela5620/mamma-streaming-data", repo_type="dataset", filename="weights/efficienttam/efficienttam_ti.pt"
    )
    return "configs/efficienttam/efficienttam_ti_512x512.yaml", checkpoint


@pytest.fixture(scope="session")
def efficienttam_ti(efficienttam_ti_files: tuple[str, str]) -> SAM2GenericVideoPredictor:
    """The EfficientTAM-ti predictor on CUDA, built as posekit and mamma build it."""
    from sam2.build_sam import build_sam2_generic_video_predictor

    return build_sam2_generic_video_predictor(*efficienttam_ti_files, device="cuda")


def _scene(step: int) -> torch.Tensor:
    """Two lenses, 256 px: a square moving right on lens 0, a disc moving down and a fixed square on lens 1."""
    frames: torch.Tensor = torch.full((2, 256, 256, 3), 40, dtype=torch.uint8)
    frames[0, 80:150, 30 + 4 * step : 100 + 4 * step] = torch.tensor([220, 60, 60], dtype=torch.uint8)
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    frames[1][(yy - 60 - 5 * step) ** 2 + (xx - 128) ** 2 < 35**2] = torch.tensor([60, 200, 90], dtype=torch.uint8)
    frames[1, 180:230, 20:70] = torch.tensor([200, 200, 40], dtype=torch.uint8)
    return frames


@pytest.fixture
def scene() -> Callable[[int], torch.Tensor]:
    """The scene at a step, (2, 256, 256, 3) uint8 on the host."""
    return _scene
