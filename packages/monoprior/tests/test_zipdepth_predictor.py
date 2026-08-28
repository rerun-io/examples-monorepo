"""ZipDepthPredictor: registration, checkpoint-prefix handling, and the RelativeDepthPrediction contract."""

from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch

from monopriors.models.relative_depth import RELATIVE_PREDICTORS, ZipDepthPredictor, get_relative_predictor
from monopriors.models.relative_depth import zipdepth as zipdepth_module
from monopriors.third_party.zipdepth.architecture import create_model
from monopriors.third_party.zipdepth.model_utils import strip_state_dict_prefixes


def test_registered() -> None:
    assert "ZipDepthPredictor" in get_args(RELATIVE_PREDICTORS)
    assert get_relative_predictor("ZipDepthPredictor") is ZipDepthPredictor


def test_strip_prefixes_roundtrip() -> None:
    # Training checkpoints carry DDP / torch.compile prefixes; the plain model must accept them.
    model = create_model(variant="base")
    prefixed = {"module._orig_mod." + k: v for k, v in model.state_dict().items()}
    model.load_state_dict(strip_state_dict_prefixes(prefixed), strict=True)


class _FakeInference:
    """Stands in for DepthInference: records the BGR input and returns a deterministic disparity."""

    def __init__(self, checkpoint_path: str, **kwargs: object) -> None:
        self.checkpoint_path = checkpoint_path
        self.kwargs = kwargs
        self.device = kwargs["device"]
        self.model = create_model(variant="base")
        self._resize_buf_shape: tuple[int, int] | None = (1, 1)
        self.last_bgr: np.ndarray | None = None

    def infer_image(self, bgr: np.ndarray) -> np.ndarray:
        self.last_bgr = bgr
        h, w = bgr.shape[:2]
        return np.linspace(0.01, 0.15, h * w, dtype=np.float32).reshape(h, w)


@pytest.fixture
def predictor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ZipDepthPredictor:
    monkeypatch.setattr(zipdepth_module, "DepthInference", _FakeInference)
    monkeypatch.setattr(zipdepth_module, "download_zipdepth_checkpoint", lambda npu=False: tmp_path / "fake.pth")
    return ZipDepthPredictor(device="cpu")


def test_call_contract_and_bgr_conversion(predictor: ZipDepthPredictor) -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    pred = predictor(rgb, None)
    fake: _FakeInference = predictor._inference  # type: ignore[assignment]
    assert fake.last_bgr is not None and fake.last_bgr[0, 0].tolist() == [0, 0, 255]  # red lands in the B channel
    assert pred.disparity.shape == (4, 6) and pred.disparity.dtype == np.float32
    assert pred.depth.shape == (4, 6) and pred.depth.dtype == np.float32
    assert np.array_equal(pred.confidence, np.ones((4, 6), dtype=np.float32))
    assert pred.K_33.dtype == np.float32 and pred.K_33[0, 2] == 3.0 and pred.K_33[1, 2] == 2.0


def test_supplied_intrinsics_are_kept_as_float32(predictor: ZipDepthPredictor) -> None:
    K = np.array([[500.0, 0, 3], [0, 500.0, 2], [0, 0, 1]], dtype=np.float64)
    pred = predictor(np.zeros((4, 6, 3), dtype=np.uint8), K)
    assert pred.K_33.dtype == np.float32 and np.array_equal(pred.K_33, K.astype(np.float32))


def test_set_model_device_retargets_runtime(predictor: ZipDepthPredictor) -> None:
    fake: _FakeInference = predictor._inference  # type: ignore[assignment]
    predictor.set_model_device("cpu")
    assert fake.device == "cpu" and fake._resize_buf_shape is None
    assert predictor.model is fake.model


def test_local_checkpoint_bypasses_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(zipdepth_module, "DepthInference", _FakeInference)
    monkeypatch.setattr(zipdepth_module, "download_zipdepth_checkpoint", lambda npu=False: pytest.fail("should not download"))
    predictor = ZipDepthPredictor(device="cpu", checkpoint=tmp_path / "trained.pth")
    assert predictor._inference.checkpoint_path == str(tmp_path / "trained.pth")  # type: ignore[attr-defined]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and the Hub weights")
def test_real_weights_predict() -> None:
    pred = ZipDepthPredictor(device="cuda")(np.full((240, 320, 3), 128, dtype=np.uint8), None)
    assert pred.disparity.shape == (240, 320) and np.isfinite(pred.disparity).all()
