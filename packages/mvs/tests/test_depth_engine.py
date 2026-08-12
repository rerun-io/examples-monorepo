from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Bool, Float, UInt8
from numpy import ndarray
from torch import Tensor
from trtkit import OnnxBackendConfig, TensorRtBackendConfig

from mvs.depth_engine import DepthEngine, DepthInputs, preprocess_image, s1_intrinsics
from tests.generate_golden import seeded_inputs  # pyrefly: ignore  # test helper is outside the configured source-only search path

MODEL_PATH: Path = Path(__file__).parents[1] / "data" / "models" / "depth_model_dot.onnx"
GOLDEN_PATH: Path = Path(__file__).parent / "data" / "depth_golden.npz"


def test_onnx_depth_engine_matches_cpu_golden() -> None:
    """CUDA depth max abs/rel: 0.001004/0.000753; confidence close pixels: 99.601%, max abs 4.725057."""

    if not MODEL_PATH.exists():
        pytest.skip(f"depth model is absent: {MODEL_PATH}")
    with np.load(GOLDEN_PATH) as golden:
        seed: int = int(golden["seed"])
        inputs: dict[str, ndarray] = seeded_inputs(seed)
        engine: DepthEngine = DepthEngine(MODEL_PATH, OnnxBackendConfig())
        tensors: dict[str, Tensor] = {
            name: torch.from_numpy(value).to(device="cuda", dtype=torch.float32) for name, value in inputs.items()
        }
        depth_b1hw: Float[Tensor, "1 1 192 256"]
        lowest_cost_bhw: Float[Tensor, "1 96 128"]
        depth_b1hw, lowest_cost_bhw = engine(DepthInputs(**tensors))
        np.testing.assert_allclose(depth_b1hw.cpu().numpy(), golden["depth_pred_s0_b1hw"], rtol=1e-2, atol=1e-3)
        confidence_close_bhw: Bool[ndarray, "1 96 128"] = np.isclose(
            lowest_cost_bhw.cpu().numpy(), golden["lowest_cost_bhw"], rtol=1e-2, atol=1e-3
        )
        assert float(confidence_close_bhw.mean()) >= 0.995


def test_onnx_depth_engine_batch_outputs_match_single_samples() -> None:
    """Depth stays within 1e-3; at least 99.5% of CUDA's discrete confidence proxy agrees."""

    if not MODEL_PATH.exists():
        pytest.skip(f"depth model is absent: {MODEL_PATH}")
    inputs: dict[str, ndarray] = seeded_inputs(20260811, batch_size=8)
    engine: DepthEngine = DepthEngine(MODEL_PATH, OnnxBackendConfig())
    tensors: dict[str, Tensor] = {
        name: torch.from_numpy(value).to(device="cuda", dtype=torch.float32) for name, value in inputs.items()
    }
    single_depths: list[Float[Tensor, "c=1 h=192 w=256"]] = []
    single_lowest_costs: list[Float[Tensor, "h=96 w=128"]] = []
    sample_index: int
    for sample_index in range(8):
        sample_tensors: dict[str, Tensor] = {
            name: tensor[sample_index : sample_index + 1] for name, tensor in tensors.items()
        }
        sample_depth_b1hw: Float[Tensor, "1 1 192 256"]
        sample_lowest_cost_bhw: Float[Tensor, "1 96 128"]
        sample_depth_b1hw, sample_lowest_cost_bhw = engine(DepthInputs(**sample_tensors))
        single_depths.append(sample_depth_b1hw[0].clone())
        single_lowest_costs.append(sample_lowest_cost_bhw[0].clone())

    batch_depth_b1hw: Float[Tensor, "8 1 192 256"]
    batch_lowest_cost_bhw: Float[Tensor, "8 96 128"]
    batch_depth_b1hw, batch_lowest_cost_bhw = engine(DepthInputs(**tensors))
    expected_depth_b1hw: Float[Tensor, "8 1 192 256"] = torch.stack(single_depths)
    expected_lowest_cost_bhw: Float[Tensor, "8 96 128"] = torch.stack(single_lowest_costs)
    torch.testing.assert_close(batch_depth_b1hw, expected_depth_b1hw, rtol=0.0, atol=1e-3)
    confidence_close_bhw: Bool[Tensor, "8 96 128"] = torch.isclose(
        batch_lowest_cost_bhw, expected_lowest_cost_bhw, rtol=0.0, atol=1e-3
    )
    confidence_close_fraction_b: Float[Tensor, "8"] = confidence_close_bhw.float().mean(dim=(1, 2))
    assert bool(torch.all(confidence_close_fraction_b >= 0.995)), confidence_close_fraction_b.tolist()


def test_tensorrt_engine_matches_onnx_runtime() -> None:
    """TensorRT parity guards the cost-volume GridSample materialization.

    Without ``TRT_EXTRA_OUTPUT_PATTERNS`` TensorRT fuses the plane-sweep
    GridSamples into miscompiled kernels (~0.3 m mean depth error); with the
    materialization the measured max depth difference is 3.6e-3.
    """

    if not MODEL_PATH.exists():
        pytest.skip(f"depth model is absent: {MODEL_PATH}")
    if not torch.cuda.is_available():
        pytest.skip("TensorRT requires CUDA")
    inputs: dict[str, ndarray] = seeded_inputs(20260811, batch_size=8)
    tensors: dict[str, Tensor] = {
        name: torch.from_numpy(value).to(device="cuda", dtype=torch.float32) for name, value in inputs.items()
    }
    ort_outputs = DepthEngine(MODEL_PATH, OnnxBackendConfig())(DepthInputs(**tensors))
    trt_outputs = DepthEngine(MODEL_PATH, TensorRtBackendConfig())(DepthInputs(**tensors))
    torch.testing.assert_close(trt_outputs.depth_b1hw, ort_outputs.depth_b1hw, rtol=0.0, atol=1e-2)
    confidence_close_bhw: Bool[Tensor, "8 96 128"] = torch.isclose(
        trt_outputs.lowest_cost_bhw, ort_outputs.lowest_cost_bhw, rtol=1e-2, atol=1e-3
    )
    confidence_close_fraction_b: Float[Tensor, "8"] = confidence_close_bhw.float().mean(dim=(1, 2))
    assert bool(torch.all(confidence_close_fraction_b >= 0.995)), confidence_close_fraction_b.tolist()


def test_preprocess_image_resizes_and_normalizes_rgb() -> None:
    """A constant uint8 image keeps its color through bicubic resizing."""

    image_chw: UInt8[Tensor, "3 h w"] = torch.full((3, 9, 13), 255, dtype=torch.uint8)
    normalized_3hw: Float[Tensor, "3 384 512"] = preprocess_image(image_chw)

    expected_3: Float[Tensor, "3"] = torch.tensor(
        [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225], dtype=torch.float32
    )
    assert normalized_3hw.shape == (3, 384, 512)
    assert normalized_3hw.dtype == torch.float32
    torch.testing.assert_close(normalized_3hw[:, 200, 300], expected_3)


def test_s1_intrinsics_scales_native_calibration_to_matching_resolution() -> None:
    """The s1 calibration targets 128x96 after depth-resolution scaling."""

    K_n33: Float[Tensor, "1 3 3"] = torch.tensor(
        [[[1200.0, 0.0, 960.0], [0.0, 1200.0, 720.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
    )
    K_s1_n44: Float[Tensor, "1 4 4"]
    invK_s1_n44: Float[Tensor, "1 4 4"]
    K_s1_n44, invK_s1_n44 = s1_intrinsics(K_n33, (1920, 1440))

    expected_44: Float[Tensor, "4 4"] = torch.tensor(
        [[80.0, 0.0, 64.0, 0.0], [0.0, 80.0, 48.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(K_s1_n44[0], expected_44)
    torch.testing.assert_close(K_s1_n44[0] @ invK_s1_n44[0], torch.eye(4), atol=1e-5, rtol=0.0)
