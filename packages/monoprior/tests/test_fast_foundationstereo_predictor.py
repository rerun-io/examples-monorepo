"""FastFoundationStereoPredictor registry and stereo prediction contract."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast, get_args

import numpy as np
import pytest
import torch
from jaxtyping import Bool, Float32, UInt8
from omegaconf import DictConfig, OmegaConf
from torch import nn

from monopriors.models.stereo_depth import STEREO_PREDICTORS, FastFoundationStereoPredictor, StereoDepthPrediction, get_stereo_predictor
from monopriors.models.stereo_depth import fast_foundationstereo as predictor_module
from monopriors.third_party.fast_foundationstereo import extractor as extractor_module
from monopriors.third_party.fast_foundationstereo import foundation_stereo as foundation_stereo_module
from monopriors.third_party.fast_foundationstereo import submodule as submodule_module
from monopriors.third_party.fast_foundationstereo.foundation_stereo import FastFoundationStereo

TIMM_CREATE_MODEL: Callable[..., nn.Module] = cast(Callable[..., nn.Module], extractor_module.timm.create_model)


class FakeFastFoundationStereo(nn.Module):
    """Cheap model double that exposes the upstream forward contract at the public predictor seam."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor: nn.Parameter = nn.Parameter(torch.empty(0))

    def forward(
        self,
        left_13hw: Float32[torch.Tensor, "1 3 h w"],
        right_13hw: Float32[torch.Tensor, "1 3 h w"],
        iters: int,
        test_mode: bool,
        optimize_build_volume: str,
    ) -> Float32[torch.Tensor, "1 1 h w"]:
        assert left_13hw.shape == right_13hw.shape
        assert left_13hw.shape[-2] % 32 == 0 and left_13hw.shape[-1] % 32 == 0
        assert left_13hw.dtype == torch.float32 and 0.0 <= float(left_13hw.min()) <= float(left_13hw.max()) <= 255.0
        assert iters == 3 and test_mode is True and optimize_build_volume == "pytorch1"
        return left_13hw[:, :1] - 100.0 + self.anchor.sum()


def test_registered() -> None:
    assert "FastFoundationStereoPredictor" in get_args(STEREO_PREDICTORS)
    assert get_stereo_predictor("FastFoundationStereoPredictor") is FastFoundationStereoPredictor


def test_random_module_runs_tiny_cpu_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config-built random module runs the PyTorch cost-volume path without CUDA or released weights."""

    def create_model_without_pretrained(model_name: str, *args: Any, **kwargs: Any) -> nn.Module:
        kwargs["pretrained"] = False
        return TIMM_CREATE_MODEL(model_name, *args, **kwargs)

    monkeypatch.setattr(extractor_module.timm, "create_model", create_model_without_pretrained)
    eager_gwc: Callable[..., torch.Tensor] = cast(Callable[..., torch.Tensor], submodule_module.build_gwc_volume_optimized_pytorch1._torchdynamo_orig_callable)
    eager_concat: Callable[..., torch.Tensor] = cast(Callable[..., torch.Tensor], submodule_module.build_concat_volume_optimized_pytorch1._torchdynamo_orig_callable)
    monkeypatch.setattr(foundation_stereo_module, "build_gwc_volume_optimized_pytorch1", eager_gwc)
    monkeypatch.setattr(foundation_stereo_module, "build_concat_volume_optimized_pytorch1", eager_concat)
    config: DictConfig = OmegaConf.create(
        {
            "hidden_dims": [128],
            "vit_size": "vitl",
            "n_gru_layers": 1,
            "n_downsample": 2,
            "corr_levels": 2,
            "corr_radius": 4,
            "mixed_precision": False,
            "valid_iters": 1,
            "low_memory": 0,
            "max_disp": 32,
            "normalize": True,
            "cv_group": 8,
        }
    )
    model: FastFoundationStereo = FastFoundationStereo(config).eval()
    left_13hw: Float32[torch.Tensor, "1 3 32 64"] = torch.rand((1, 3, 32, 64), dtype=torch.float32) * 255.0
    right_13hw: Float32[torch.Tensor, "1 3 32 64"] = torch.rand((1, 3, 32, 64), dtype=torch.float32) * 255.0
    with torch.inference_mode():
        disparity_11hw: Float32[torch.Tensor, "1 1 32 64"] = model(left_13hw, right_13hw, iters=1, test_mode=True, optimize_build_volume="pytorch1")
    assert disparity_11hw.shape == (1, 1, 32, 64)
    assert disparity_11hw.dtype == torch.float32
    assert torch.isfinite(disparity_11hw).all()


def test_predictor_preprocesses_and_returns_metric_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint: Path = Path("unused.pth")
    fake_model: FakeFastFoundationStereo = FakeFastFoundationStereo()

    def fake_load(path: Path, valid_iters: int, max_disp: int) -> nn.Module:
        assert path == checkpoint and valid_iters == 3 and max_disp == 64
        return fake_model

    monkeypatch.setattr(predictor_module, "load_fast_foundationstereo", fake_load)
    monkeypatch.setattr(predictor_module, "download_fast_foundationstereo_checkpoint", lambda: pytest.fail("local checkpoint must bypass download"))
    predictor: FastFoundationStereoPredictor = FastFoundationStereoPredictor(device="cpu", checkpoint=checkpoint, valid_iters=3, max_disp=64)
    left_rgb: UInt8[np.ndarray, "35 61 3"] = np.arange(35 * 61 * 3, dtype=np.uint8).reshape(35, 61, 3)
    right_rgb: UInt8[np.ndarray, "35 61 3"] = np.flip(left_rgb, axis=1).copy()
    K_33: Float32[np.ndarray, "3 3"] = np.array([[100.0, 0.0, 30.0], [0.0, 100.0, 17.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    prediction: StereoDepthPrediction = predictor(left_rgb, right_rgb, K_33=K_33, baseline_m=0.1)

    expected_disparity: Float32[np.ndarray, "35 61"] = np.clip(left_rgb[..., 0].astype(np.float32) - 100.0, 0.0, None)
    assert np.array_equal(prediction.disparity, expected_disparity)
    assert prediction.depth_meters is not None
    valid_hw: Bool[np.ndarray, "35 61"] = expected_disparity > 0.0
    assert np.allclose(prediction.depth_meters[valid_hw], 10.0 / expected_disparity[valid_hw])
    assert np.all(prediction.depth_meters[~valid_hw] == 0.0)
    assert prediction.K_33 is K_33 and prediction.baseline_m == 0.1
