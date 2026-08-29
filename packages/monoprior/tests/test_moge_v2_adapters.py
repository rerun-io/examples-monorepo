"""Contracts for MoGe v2 single-image NumPy adapters."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from conftest import requires_cuda, slow_cuda
from einops import rearrange
from jaxtyping import Bool, Float32, UInt8
from torch import Tensor

from monopriors.models.metric_depth import MetricDepthPrediction
from monopriors.models.moge_v2 import MoGeV2GeometryOutput, MoGeV2NormalOutput
from monopriors.models.moge_v2.adapters import geometry_to_metric_prediction, normal_to_surface_prediction
from monopriors.models.surface_normal import SurfaceNormalPrediction


def test_geometry_conversion_uses_pixel_intrinsics_binary_confidence_and_infinite_fill() -> None:
    """Metric conversion returns caller-pixel intrinsics and masks invalid depth."""
    geometry: MoGeV2GeometryOutput = MoGeV2GeometryOutput(
        depth_bhw=torch.tensor([[[2.0, 3.0], [-1.0, 5.0]]], dtype=torch.float32),
        points_bhw3=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        normals_bhw3=torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        mask_bhw=torch.tensor([[[0.6, 0.5], [0.0, 1.0]]], dtype=torch.float32),
        intrinsics_b33=torch.tensor([[[0.75, 0.0, 0.4], [0.0, 0.8, 0.6], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        metric_scale_b=torch.ones(1, dtype=torch.float32),
        shift_b=torch.zeros(1, dtype=torch.float32),
    )

    prediction: MetricDepthPrediction = geometry_to_metric_prediction(geometry)

    expected_depth_hw: Float32[np.ndarray, "2 2"] = np.array([[2.0, np.inf], [np.inf, 5.0]], dtype=np.float32)
    expected_confidence_hw: Float32[np.ndarray, "2 2"] = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    expected_K_33: Float32[np.ndarray, "3 3"] = np.array(
        [[1.5, 0.0, 0.8], [0.0, 1.6, 1.2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(prediction.depth_meters, expected_depth_hw)
    np.testing.assert_array_equal(prediction.confidence, expected_confidence_hw)
    np.testing.assert_allclose(prediction.K_33, expected_K_33)


def test_normal_conversion_zeros_invalid_vectors_and_returns_binary_confidence() -> None:
    """Normal conversion applies the strict mask threshold to both outputs."""
    output: MoGeV2NormalOutput = MoGeV2NormalOutput(
        normals_bhw3=torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        ),
        mask_bhw=torch.tensor([[[0.51, 0.5], [0.0, 1.0]]], dtype=torch.float32),
    )

    prediction: SurfaceNormalPrediction = normal_to_surface_prediction(output)

    expected_normals_hw3: Float32[np.ndarray, "2 2 3"] = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    expected_confidence_hw1: Float32[np.ndarray, "2 2 1"] = np.array([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=np.float32)
    np.testing.assert_array_equal(prediction.normal_hw3, expected_normals_hw3)
    np.testing.assert_array_equal(prediction.confidence_hw1, expected_confidence_hw1)


def test_single_image_adapters_reject_cpu_at_construction() -> None:
    """Both registry adapters fail early with the shared CUDA-only message."""
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from monopriors.models.surface_normal.moge_v2 import MoGeV2NormalPredictor

    with pytest.raises(ValueError, match="^MoGe v2 predictors require CUDA;"):
        MoGeV2MetricPredictor(device="cpu")
    with pytest.raises(ValueError, match="^MoGe v2 predictors require CUDA;"):
        MoGeV2NormalPredictor(device="cpu")


@slow_cuda
@requires_cuda
def test_all_single_image_adapters_match_vendored_infer_on_room() -> None:
    """Metric, normal, and paired adapters match the pinned vendored reference."""
    from monopriors.models.metric_depth.moge_v2 import MoGeV2MetricPredictor
    from monopriors.models.moge_v2 import MOGE_V2_CHECKPOINTS
    from monopriors.models.monoprior import MoGeV2MonoPrior, MonoPriorPrediction
    from monopriors.models.surface_normal.moge_v2 import MoGeV2NormalPredictor
    from monopriors.third_party.moge.model._inference import InferenceOutput
    from monopriors.third_party.moge.model.v2 import MoGeModel

    image_path: Path = Path("data/examples/single-image/room.jpg")
    bgr_hw3: UInt8[np.ndarray, "h w 3"] | None = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_hw3 is None:
        pytest.skip(f"{image_path} missing; run the monoprior download task")
    rgb_hw3: UInt8[np.ndarray, "h w 3"] = cv2.cvtColor(bgr_hw3, cv2.COLOR_BGR2RGB)
    image_3hw: Float32[Tensor, "3 h w"] = rearrange(torch.from_numpy(rgb_hw3).to(device="cuda"), "h w c -> c h w").float() / 255.0  # pyrefly: ignore  # bad-argument-type — einops stub false positive
    checkpoint: tuple[str, str] = MOGE_V2_CHECKPOINTS["vitl"]
    reference_model: MoGeModel = MoGeModel.from_pretrained(checkpoint[0], revision=checkpoint[1]).to("cuda").eval()
    reference: InferenceOutput = reference_model.infer(image_3hw, force_projection=False, apply_mask=True)
    reference_depth_hw: Float32[np.ndarray, "h w"] = reference["depth"].numpy(force=True)
    reference_K_33: Float32[np.ndarray, "3 3"] = reference["intrinsics"].numpy(force=True).astype(np.float32, copy=True)
    reference_K_33[0, :] *= rgb_hw3.shape[1]
    reference_K_33[1, :] *= rgb_hw3.shape[0]
    reference_normals_hw3: Float32[np.ndarray, "h w 3"] = reference["normal"].numpy(force=True)
    reference_confidence_hw: Float32[np.ndarray, "h w"] = reference["mask"].numpy(force=True).astype(np.float32)
    del reference_model, reference
    torch.cuda.empty_cache()

    metric_adapter: MoGeV2MetricPredictor = MoGeV2MetricPredictor(device="cuda")
    metric_adapter.set_model_device("cuda")
    metric_prediction: MetricDepthPrediction = metric_adapter(rgb_hw3)
    del metric_adapter
    torch.cuda.empty_cache()
    normal_adapter: MoGeV2NormalPredictor = MoGeV2NormalPredictor(device="cuda")
    normal_prediction: SurfaceNormalPrediction = normal_adapter(rgb_hw3)
    del normal_adapter
    torch.cuda.empty_cache()
    paired_adapter: MoGeV2MonoPrior = MoGeV2MonoPrior()
    paired_prediction: MonoPriorPrediction = paired_adapter(rgb_hw3)
    del paired_adapter
    torch.cuda.empty_cache()

    valid_metric_hw: Bool[np.ndarray, "h w"] = np.isfinite(reference_depth_hw) & np.isfinite(metric_prediction.depth_meters)
    metric_relative_error_valid: Float32[np.ndarray, "valid"] = np.abs(
        metric_prediction.depth_meters[valid_metric_hw] - reference_depth_hw[valid_metric_hw]
    ) / np.maximum(
        np.abs(reference_depth_hw[valid_metric_hw]), 1e-6
    )
    assert np.median(metric_relative_error_valid) < 0.01
    intrinsics_indices: tuple[tuple[int, ...], tuple[int, ...]] = ((0, 1, 0, 1), (0, 1, 2, 2))
    np.testing.assert_allclose(
        metric_prediction.K_33[intrinsics_indices],
        reference_K_33[intrinsics_indices],
        rtol=0.01,
        atol=0.0,
    )
    assert np.mean(metric_prediction.confidence == reference_confidence_hw) > 0.99

    valid_normal_hw: Bool[np.ndarray, "h w"] = (normal_prediction.confidence_hw1[..., 0] > 0.5) & (reference_confidence_hw > 0.5)
    normal_cosine_valid: Float32[np.ndarray, "valid"] = np.sum(normal_prediction.normal_hw3 * reference_normals_hw3, axis=-1)[valid_normal_hw].clip(
        -1.0, 1.0
    )
    assert np.rad2deg(np.arccos(normal_cosine_valid)).mean() < 1.0
    assert np.mean(normal_prediction.confidence_hw1[..., 0] == reference_confidence_hw) > 0.99

    valid_paired_hw: Bool[np.ndarray, "h w"] = np.isfinite(reference_depth_hw) & np.isfinite(paired_prediction.metric_pred.depth_meters)
    paired_relative_error_valid: Float32[np.ndarray, "valid"] = np.abs(
        paired_prediction.metric_pred.depth_meters[valid_paired_hw] - reference_depth_hw[valid_paired_hw]
    ) / np.maximum(np.abs(reference_depth_hw[valid_paired_hw]), 1e-6)
    assert np.median(paired_relative_error_valid) < 0.01
    paired_normal_valid_hw: Bool[np.ndarray, "h w"] = (paired_prediction.normal_pred.confidence_hw1[..., 0] > 0.5) & (reference_confidence_hw > 0.5)
    paired_cosine_valid: Float32[np.ndarray, "valid"] = np.sum(paired_prediction.normal_pred.normal_hw3 * reference_normals_hw3, axis=-1)[
        paired_normal_valid_hw
    ].clip(-1.0, 1.0)
    assert np.rad2deg(np.arccos(paired_cosine_valid)).mean() < 1.0
    np.testing.assert_allclose(
        paired_prediction.metric_pred.K_33[intrinsics_indices],
        reference_K_33[intrinsics_indices],
        rtol=0.01,
        atol=0.0,
    )
    assert np.mean(paired_prediction.metric_pred.confidence == reference_confidence_hw) > 0.99
    assert np.mean(paired_prediction.normal_pred.confidence_hw1[..., 0] == reference_confidence_hw) > 0.99
