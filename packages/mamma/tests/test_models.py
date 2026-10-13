"""Model tier (GPU and weights): the streaming MammaNet estimator returns owned tensors (integration lane)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mamma.apis.capture import PACKAGE_DIR, engine_plan

pytestmark = pytest.mark.integration
WEIGHTS: Path = PACKAGE_DIR / "data/weights/ma_2d/mamma_mask_full_cvpr.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available() or not WEIGHTS.exists(), reason=f"needs a CUDA GPU and the MammaNet weights ({WEIGHTS.name})")
def test_estimator_outputs_survive_the_next_call() -> None:
    from mamma.models.landmarks.estimator import LandmarkEstimator

    estimator = LandmarkEstimator(WEIGHTS, engine_path=engine_plan())
    frames = torch.randint(0, 255, (2, 3, 96, 128), dtype=torch.uint8, device="cuda")
    masks = torch.zeros((2, 96, 128), dtype=torch.bool, device="cuda")
    masks[:, 20:80, 40:90] = True
    boxes = np.array([[40.0, 20.0, 90.0, 80.0]] * 2, dtype=np.float32)
    first = estimator(frames, np.array([0, 1]), masks, boxes)
    kept = first.xy.clone()
    estimator(torch.flip(frames, dims=[3]), np.array([1, 0]), masks, boxes + 3.0)
    torch.cuda.synchronize()
    assert torch.equal(first.xy, kept) and first.xy.shape == (2, 512, 2)
