"""ZipDepthPredictor: registration, checkpoint handling, and the RelativeDepthPrediction contract.

Runs on CPU against a randomly initialised ZipDepth-base saved to disk, so no download or GPU is needed.
"""

from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch

from monopriors.models.relative_depth import RELATIVE_PREDICTORS, ZipDepthPredictor, get_relative_predictor
from monopriors.models.relative_depth import zipdepth as zipdepth_module
from monopriors.third_party.zipdepth.architecture import create_model


def test_registered() -> None:
    assert "ZipDepthPredictor" in get_args(RELATIVE_PREDICTORS)
    assert get_relative_predictor("ZipDepthPredictor") is ZipDepthPredictor


@pytest.fixture(scope="module")
def trainer_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A checkpoint in the training package's format: DDP+compile-prefixed keys plus optimizer state."""
    model = create_model(variant="base")
    path = tmp_path_factory.mktemp("ckpt") / "final_model.pth"
    torch.save({"model_state_dict": {"module._orig_mod." + k: v for k, v in model.state_dict().items()}, "optimizer_state_dict": {}}, path)
    return path


@pytest.fixture(scope="module")
def predictor(trainer_checkpoint: Path) -> ZipDepthPredictor:
    return ZipDepthPredictor(device="cpu", checkpoint=trainer_checkpoint, input_size=64)


def test_local_checkpoint_bypasses_download(monkeypatch: pytest.MonkeyPatch, trainer_checkpoint: Path) -> None:
    monkeypatch.setattr(zipdepth_module, "download_zipdepth_checkpoint", lambda npu=False: pytest.fail("should not download"))
    ZipDepthPredictor(device="cpu", checkpoint=trainer_checkpoint, input_size=64)


def test_call_contract(predictor: ZipDepthPredictor) -> None:
    rgb = np.random.default_rng(0).integers(0, 255, (48, 72, 3), dtype=np.uint8)
    pred = predictor(rgb, None)
    assert pred.disparity.shape == (48, 72) and pred.disparity.dtype == np.float32 and np.isfinite(pred.disparity).all()
    assert pred.depth.shape == (48, 72) and pred.depth.dtype == np.float32
    assert pred.confidence.shape == (48, 72)
    assert pred.K_33.dtype == np.float32 and pred.K_33[0, 2] == 36.0 and pred.K_33[1, 2] == 24.0


def test_supplied_intrinsics_are_kept_as_float32(predictor: ZipDepthPredictor) -> None:
    K = np.array([[500.0, 0, 36], [0, 500.0, 24], [0, 0, 1]], dtype=np.float64)
    pred = predictor(np.zeros((48, 72, 3), dtype=np.uint8), K)
    assert pred.K_33.dtype == np.float32 and np.array_equal(pred.K_33, K.astype(np.float32))


def test_network_size_keeps_aspect_and_multiple_of_32(predictor: ZipDepthPredictor) -> None:
    assert predictor._network_size(480, 640) == (64, 96)  # shorter side -> 64, longer side 85.3 rounded to /32
    assert predictor._network_size(640, 480) == (96, 64)


def test_set_model_device(predictor: ZipDepthPredictor) -> None:
    predictor.set_model_device("cpu")
    assert next(predictor.model.parameters()).device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and the Hub weights")
def test_real_weights_predict() -> None:
    pred = ZipDepthPredictor(device="cuda")(np.full((240, 320, 3), 128, dtype=np.uint8), None)
    assert pred.disparity.shape == (240, 320) and np.isfinite(pred.disparity).all()
