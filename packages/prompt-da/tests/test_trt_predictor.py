"""Unit tests for the PromptDA TensorRT engine/predictor helpers.

Fast checks run everywhere; GPU-only checks skip without CUDA; the full
export->build->parity path is opt-in via ``PROMPTDA_TRT_E2E=1`` because the
first run builds an engine (minutes).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from rerun_prompt_da.apis.prompt_da_trt_polycam import network_image_hw
from rerun_prompt_da.trt_engine import TrtBuildConfig, cached_engine_path

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_network_image_hw_downscales_and_14_aligns() -> None:
    """The Polycam capture resolution maps to the 14-aligned network size."""

    assert network_image_hw((768, 1024), max_image_size=1008) == (756, 1008)


def test_network_image_hw_never_upscales() -> None:
    """Captures smaller than max_image_size only get 14-aligned, not enlarged."""

    assert network_image_hw((280, 294), max_image_size=1008) == (280, 294)
    assert network_image_hw((290, 300), max_image_size=1008) == (280, 294)


def test_cached_engine_path_rejects_bad_batch_range() -> None:
    """opt_batch_size outside [1, max_batch_size] is a configuration error."""

    config = TrtBuildConfig(max_batch_size=4, opt_batch_size=8)
    with pytest.raises(ValueError, match="opt_batch_size"):
        cached_engine_path(Path("model.onnx"), config)


@requires_cuda
def test_cached_engine_path_is_deterministic_and_cache_keyed() -> None:
    """The engine name encodes batch range, precision, TRT version, and SM."""

    config = TrtBuildConfig(max_batch_size=8, opt_batch_size=8)
    first: Path = cached_engine_path(Path("promptda-large_756x1008_op17.onnx"), config)
    second: Path = cached_engine_path(Path("promptda-large_756x1008_op17.onnx"), config)
    assert first == second
    assert "_b1-8-8_strong_" in first.name
    other: Path = cached_engine_path(Path("promptda-large_756x1008_op17.onnx"), TrtBuildConfig(max_batch_size=4, opt_batch_size=4))
    assert other != first


@requires_cuda
def test_preprocess_batch_shapes_dtypes_and_range() -> None:
    """Preprocessing yields normalized float32 CUDA tensors at the network size."""

    from rerun_prompt_da.trt_predictor import preprocess_batch

    rgb_bhw3 = torch.randint(0, 256, (2, 768, 1024, 3), dtype=torch.uint8)
    prompt_bhw = torch.rand((2, 192, 256), dtype=torch.float32) * 4.0

    image_b3hw, prompt_b1hw = preprocess_batch(rgb_bhw3, prompt_bhw, image_hw=(756, 1008))

    assert image_b3hw.shape == (2, 3, 756, 1008)
    assert image_b3hw.dtype == torch.float32
    assert image_b3hw.is_cuda
    assert image_b3hw.min().item() >= 0.0
    assert image_b3hw.max().item() <= 1.0
    assert prompt_b1hw.shape == (2, 1, 192, 256)
    assert prompt_b1hw.is_cuda


@pytest.mark.skipif(os.environ.get("PROMPTDA_TRT_E2E") != "1", reason="set PROMPTDA_TRT_E2E=1 to run the engine-building parity test")
@requires_cuda
def test_trt_matches_torch_on_synthetic_frames() -> None:
    """The fp16 TRT engine agrees with the fp32 torch model within tolerance."""

    from rerun_prompt_da.trt_predictor import PromptDATorchPredictor, PromptDATrtPredictor

    rgb_bhw3 = torch.randint(0, 256, (8, 768, 1024, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    prompt_bhw = torch.rand((8, 192, 256), dtype=torch.float32, generator=torch.Generator().manual_seed(1)) * 3.0 + 0.5

    trt_predictor = PromptDATrtPredictor(batch_size=8)
    torch_predictor = PromptDATorchPredictor()
    depth_trt = trt_predictor(rgb_bhw3.cuda(), prompt_bhw.cuda())
    depth_torch = torch_predictor(rgb_bhw3.cuda(), prompt_bhw.cuda())
    torch.cuda.synchronize()

    assert depth_trt.shape == depth_torch.shape == (8, 768, 1024)
    diff_mm = (depth_trt - depth_torch).abs().mul(1000.0)
    assert diff_mm.median().item() < 10.0
    # Partial batches run through the same dynamic engine.
    depth_b3 = trt_predictor(rgb_bhw3[:3].cuda(), prompt_bhw[:3].cuda())
    torch.cuda.synchronize()
    np.testing.assert_allclose(
        depth_b3.cpu().numpy(),
        depth_trt[:3].cpu().numpy(),
        atol=5e-2,
    )


@pytest.mark.skipif(os.environ.get("PROMPTDA_TRT_E2E") != "1", reason="set PROMPTDA_TRT_E2E=1 to run the engine-building parity test")
@requires_cuda
def test_trt_matches_torch_family_pipeline() -> None:
    """The TRT path agrees with the shared torch predictor contract end to end."""

    import torch.nn.functional as F

    from rerun_prompt_da.trt_predictor import PromptDATorchPredictor, PromptDATrtPredictor

    generator = torch.Generator().manual_seed(7)
    low_freq = torch.rand((2, 3, 24, 32), generator=generator)
    rgb_b3hw = F.interpolate(low_freq, size=(768, 1024), mode="bilinear", align_corners=False)
    rgb_bhw3 = (rgb_b3hw * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    prompt_base = torch.rand((2, 1, 12, 16), generator=generator) * 3.0 + 0.5
    prompt_m = F.interpolate(prompt_base, size=(192, 256), mode="bilinear", align_corners=False)[:, 0]

    trt_predictor = PromptDATrtPredictor(batch_size=8)
    depth_trt_mm = (trt_predictor(rgb_bhw3.cuda(), prompt_m.cuda()) * 1000.0).cpu().numpy()
    torch.cuda.synchronize()

    torch_predictor = PromptDATorchPredictor()
    depth_torch_mm = (torch_predictor(rgb_bhw3.cuda(), prompt_m.cuda()) * 1000.0).cpu().numpy()
    for i in range(rgb_bhw3.shape[0]):
        diff_mm = np.abs(depth_trt_mm[i] - depth_torch_mm[i])
        assert np.median(diff_mm) < 20.0, f"frame {i}: median {np.median(diff_mm):.1f} mm vs original pipeline"
