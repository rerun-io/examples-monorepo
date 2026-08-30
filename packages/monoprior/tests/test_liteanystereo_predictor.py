"""LiteAnyStereoPredictor: registration, checkpoint loading, and the StereoDepthPrediction contract.

Runs on CPU against a randomly initialised LAS2-S saved to disk, so no download or GPU is needed.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import tyro

from monopriors.models.stereo_depth import AnnotatedStereoPredictorUnion, LiteAnyStereoConfig, LiteAnyStereoPredictor, stereo_predictor_defaults
from monopriors.models.stereo_depth import liteanystereo as liteanystereo_module
from monopriors.third_party.liteanystereo.liteanystereov2 import build_liteanystereo


def test_liteanystereo_config_registered() -> None:
    assert set(stereo_predictor_defaults) == {"liteanystereo", "fast-foundationstereo"}
    config = stereo_predictor_defaults["liteanystereo"]
    assert isinstance(config, LiteAnyStereoConfig)
    assert config.max_disp == 192
    assert isinstance(tyro.cli(AnnotatedStereoPredictorUnion, args=["liteanystereo"]), LiteAnyStereoConfig)


@pytest.fixture(scope="module")
def released_style_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint in the released format: DDP-prefixed keys under a ``model`` entry."""
    torch.manual_seed(0)
    model = build_liteanystereo(model_size="s", fnet_pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "LAS2_S.pth"
    torch.save({"model": {"module." + k: v for k, v in model.state_dict().items()}}, path)
    return path


@pytest.fixture(scope="module")
def predictor(released_style_checkpoint: Path) -> LiteAnyStereoPredictor:
    return LiteAnyStereoPredictor(device="cpu", model_size="s", checkpoint=released_style_checkpoint)


def test_local_checkpoint_bypasses_download(monkeypatch: pytest.MonkeyPatch, released_style_checkpoint: Path) -> None:
    monkeypatch.setattr(liteanystereo_module, "download_liteanystereo_checkpoint", lambda model_size: pytest.fail("should not download"))
    LiteAnyStereoPredictor(device="cpu", model_size="s", checkpoint=released_style_checkpoint)


def test_call_requires_calibration(predictor: LiteAnyStereoPredictor) -> None:
    left = np.zeros((64, 96, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        predictor(left, left)


def test_call_contract_with_calibration(predictor: LiteAnyStereoPredictor) -> None:
    left = np.zeros((64, 96, 3), dtype=np.uint8)
    K_33 = np.array([[100.0, 0, 48], [0, 100.0, 32], [0, 0, 1]], dtype=np.float32)
    pred = predictor(left, left, K_33=K_33, baseline_m=0.1)
    assert pred.depth_meters.shape == (64, 96) and pred.depth_meters.dtype == np.float32
    valid = pred.disparity > 0.0
    assert np.allclose(pred.depth_meters[valid], 100.0 * 0.1 / pred.disparity[valid])
    assert (pred.depth_meters[~valid] == 0.0).all()
    assert pred.baseline_m == 0.1 and pred.K_33 is K_33
