"""Contract, parity, and convention tests for batched MoGe v2 normals."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import requires_cuda, slow_cuda, synthetic_rgb_batch
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor


def test_trt_predictor_loads_prebuilt_engine_without_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A supplied engine bypasses ONNX export and TensorRT build work."""
    import monopriors.models.surface_normal.moge_v2_trt as moge_v2_trt

    class FakeRuntime:
        def __init__(self, engine_path: Path) -> None:
            self.engine_path: Path = engine_path

    def fail_export(*args: object, **kwargs: object) -> Path:
        raise AssertionError("prebuilt engine path must bypass ONNX export")

    engine_path: Path = tmp_path / "moge.engine"
    engine_path.write_bytes(b"engine")
    monkeypatch.setattr(moge_v2_trt, "export_moge_v2_normal_onnx", fail_export)
    monkeypatch.setitem(sys.modules, "trtkit.tensorrt_runtime", SimpleNamespace(TensorRtRuntime=FakeRuntime))

    predictor: moge_v2_trt.MoGeV2TrtNormalPredictor = moge_v2_trt.MoGeV2TrtNormalPredictor(engine_path=engine_path)

    assert predictor.runtime.engine_path == engine_path


@slow_cuda
@requires_cuda
def test_torch_predictor_returns_batched_unit_normals_and_masks() -> None:
    """The torch twin honors the shared batch, dtype, and unit-normal contract."""
    from monopriors.models.surface_normal.moge_v2_trt import MoGeV2NormalOutput, MoGeV2TorchNormalPredictor

    image_hw: tuple[int, int] = (56, 70)
    rgb_bhw3: UInt8[Tensor, "2 56 70 3"] = synthetic_rgb_batch(2, image_hw)
    predictor: MoGeV2TorchNormalPredictor = MoGeV2TorchNormalPredictor(
        encoder="vits",
        image_hw=image_hw,
        resolution_level=0,
    )

    prediction: MoGeV2NormalOutput = predictor(rgb_bhw3)

    assert prediction.normals_bhw3.shape == (2, 56, 70, 3)
    assert prediction.normals_bhw3.dtype == torch.float32
    assert prediction.normals_bhw3.is_cuda
    assert prediction.mask_bhw.shape == (2, 56, 70)
    assert prediction.mask_bhw.dtype == torch.float32
    assert prediction.mask_bhw.is_cuda
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.normals_bhw3, dim=-1),
        torch.ones((2, 56, 70), device="cuda"),
        atol=2e-3,
        rtol=0.0,
    )
    assert prediction.mask_bhw.min().item() >= 0.0
    assert prediction.mask_bhw.max().item() <= 1.0


@slow_cuda
@requires_cuda
def test_trt_matches_torch_twin_at_default_arkitscenes_resolution() -> None:
    """TensorRT and ONNX-compatible torch normals meet the angular parity bound."""
    from monopriors.models.surface_normal.moge_v2_trt import (
        DEFAULT_IMAGE_HW,
        MoGeV2NormalOutput,
        MoGeV2TorchNormalPredictor,
        MoGeV2TrtNormalPredictor,
    )

    rgb_bhw3: UInt8[Tensor, "2 756 1008 3"] = synthetic_rgb_batch(2, DEFAULT_IMAGE_HW)
    rgb_bhw3[1] = torch.flip(rgb_bhw3[1], dims=(1,))
    trt_predictor: MoGeV2TrtNormalPredictor = MoGeV2TrtNormalPredictor(batch_size=8)
    torch_predictor: MoGeV2TorchNormalPredictor = MoGeV2TorchNormalPredictor()

    trt_prediction: MoGeV2NormalOutput = trt_predictor(rgb_bhw3)
    torch_prediction: MoGeV2NormalOutput = torch_predictor(rgb_bhw3)
    torch.cuda.synchronize()

    assert trt_prediction.normals_bhw3.shape == (2, 756, 1008, 3)
    assert trt_prediction.normals_bhw3.dtype == torch.float32
    assert trt_prediction.mask_bhw.shape == (2, 756, 1008)
    assert trt_prediction.mask_bhw.dtype == torch.float32
    torch.testing.assert_close(
        torch.linalg.vector_norm(trt_prediction.normals_bhw3, dim=-1),
        torch.ones((2, 756, 1008), device="cuda"),
        atol=2e-3,
        rtol=0.0,
    )
    cosine_bhw: Float32[Tensor, "2 756 1008"] = torch.sum(
        trt_prediction.normals_bhw3 * torch_prediction.normals_bhw3,
        dim=-1,
    ).clamp(-1.0, 1.0)
    angular_error_bhw: Float32[Tensor, "2 756 1008"] = torch.rad2deg(torch.acos(cosine_bhw))
    mean_error_degrees: float = angular_error_bhw.mean().item()
    p99_error_degrees: float = torch.quantile(angular_error_bhw, 0.99).item()

    assert mean_error_degrees < 0.1
    assert p99_error_degrees < 0.5

    held_prediction: MoGeV2NormalOutput = trt_predictor(rgb_bhw3[:1])
    held_normals_copy_bhw3: Float32[Tensor, "1 756 1008 3"] = held_prediction.normals_bhw3.clone()
    reused_prediction: MoGeV2NormalOutput = trt_predictor(torch.flip(rgb_bhw3[:1], dims=(2,)))
    assert reused_prediction.normals_bhw3.data_ptr() != held_prediction.normals_bhw3.data_ptr()
    torch.testing.assert_close(held_prediction.normals_bhw3, held_normals_copy_bhw3)


@slow_cuda
@requires_cuda
def test_trt_normals_match_existing_single_image_convention() -> None:
    """The TRT output keeps the existing camera-space layout and sign convention."""
    from monopriors.models.surface_normal.base_normal_model import SurfaceNormalPrediction
    from monopriors.models.surface_normal.moge_v2 import MoGeV2NormalPredictor
    from monopriors.models.surface_normal.moge_v2_trt import (
        DEFAULT_IMAGE_HW,
        MoGeV2NormalOutput,
        MoGeV2TrtNormalPredictor,
    )

    rgb_bhw3: UInt8[Tensor, "1 756 1008 3"] = synthetic_rgb_batch(1, DEFAULT_IMAGE_HW)
    rgb_hw3: UInt8[np.ndarray, "756 1008 3"] = rgb_bhw3[0].cpu().numpy()
    trt_predictor: MoGeV2TrtNormalPredictor = MoGeV2TrtNormalPredictor(batch_size=8)
    existing_predictor: MoGeV2NormalPredictor = MoGeV2NormalPredictor(device="cuda", encoder="vitl")

    trt_prediction: MoGeV2NormalOutput = trt_predictor(rgb_bhw3)
    existing_prediction: SurfaceNormalPrediction = existing_predictor(rgb_hw3)
    existing_normals_bhw3: Float32[Tensor, "1 756 1008 3"] = torch.from_numpy(existing_prediction.normal_hw3).cuda()[None]
    valid_bhw: Bool[Tensor, "1 756 1008"] = torch.from_numpy(existing_prediction.confidence_hw1[..., 0] > 0.5).cuda()[None]
    assert valid_bhw.any()
    cosine_valid: Float32[Tensor, "valid"] = torch.sum(
        trt_prediction.normals_bhw3 * existing_normals_bhw3,
        dim=-1,
    )[valid_bhw]

    assert cosine_valid.mean().item() > 0.999
    assert torch.quantile(cosine_valid, 0.01).item() > 0.995
