"""Model tier (GPU and weights): the streaming MammaNet estimator returns owned tensors (integration lane)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mamma.apis.capture import PACKAGE_DIR, engine_plan

pytestmark = pytest.mark.integration
WEIGHTS: Path = PACKAGE_DIR / "data/weights/ma_2d/mamma_mask_full_cvpr.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available() or not WEIGHTS.exists(), reason=f"needs a CUDA GPU and the MammaNet weights ({WEIGHTS.name})")
@pytest.mark.parametrize("backend", ["eager", "tensorrt"])
def test_estimator_outputs_survive_the_next_call(backend: str) -> None:
    """Every returned head is owned: the TensorRT runner reuses its output buffers and eager reuses the autocast cache."""
    from mamma.models.landmarks.estimator import LandmarkEstimator

    plan: Path | None = engine_plan()
    if backend == "tensorrt" and plan is None:
        pytest.skip("needs this machine's MammaNet static-batch TensorRT plan (.trt_cache, tools/build_trt_engine.py)")
    estimator = LandmarkEstimator(WEIGHTS, engine_path=plan if backend == "tensorrt" else None)
    assert (estimator.runner is not None) == (backend == "tensorrt")
    frames = torch.randint(0, 255, (2, 3, 96, 128), dtype=torch.uint8, device="cuda")
    masks = torch.zeros((2, 96, 128), dtype=torch.bool, device="cuda")
    masks[:, 20:80, 40:90] = True
    boxes = np.array([[40.0, 20.0, 90.0, 80.0]] * 2, dtype=np.float32)
    first = estimator(frames, np.array([0, 1]), masks, boxes)
    kept = {name: getattr(first, name).clone() for name in ("xy", "log_variance", "visibility", "floor_contact")}
    second = estimator(torch.flip(frames, dims=[3]), np.array([1, 0]), masks, boxes + 3.0)
    torch.cuda.synchronize()
    assert first.xy.shape == (2, 512, 2)
    for name, value in kept.items():
        assert torch.equal(getattr(first, name), value), name
        assert not torch.equal(getattr(second, name), value), f"{name}: the second call must differ for the check to mean anything"
