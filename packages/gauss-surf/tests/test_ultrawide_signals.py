"""Behavior checks for ultrawide-resolution MoGe normal inference."""

import numpy as np
import pytest
import torch
from arkitscenes_download.ingest.paths import FRAME_SELECTION_ULTRAWIDE
from jaxtyping import Float32, UInt8
from monopriors.models.surface_normal.moge_v2_trt import MoGeV2NormalOutput, MoGeV2TrtNormalPredictor
from numpy import ndarray
from torch import Tensor

from gauss_surf.apis.ultrawide_signals import ULTRAWIDE_IMAGE_HW, apple_distortion_coefficients
from gauss_surf.contracts import (
    APPLE_FORWARD_DISTORTION_COLUMN,
    DISTORTION_MODEL_COLUMN,
    LEGACY_DISTORTION_COEFFICIENTS_COLUMN,
    MOGE_INFERENCE_BATCH_SIZE,
    ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
)
from gauss_surf.normals_encoding import decode_normals_png, encode_normals_png, to_away_from_camera

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and TensorRT required")


def test_ultrawide_provenance_is_anchored_to_chosen_rows() -> None:
    """Rectified RGB, raycast depth, and MoGe normals share the chosen timeline."""
    assert f"/{FRAME_SELECTION_ULTRAWIDE}:sharpness" == ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN


def test_canonical_distortion_reads_apple_provenance_under_a_brown_label() -> None:
    """New recordings use exact Apple provenance while generic components stay standard."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: ["brown_conrady"],
        APPLE_FORWARD_DISTORTION_COLUMN: [[float(index) for index in range(8)]],
        LEGACY_DISTORTION_COEFFICIENTS_COLUMN: [[99.0] * 14],
    }

    coefficients_8: Float32[ndarray, "8"] = apple_distortion_coefficients(static_row)

    np.testing.assert_array_equal(coefficients_8, np.arange(8, dtype=np.float32))


@pytest.mark.parametrize("legacy_model", ["brown_conrady", "apple_radial_poly"])
def test_legacy_distortion_fallback_accepts_only_documented_stale_labels(legacy_model: str) -> None:
    """Old corpus rows may carry Apple coefficients under either known stale model label."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: [legacy_model],
        LEGACY_DISTORTION_COEFFICIENTS_COLUMN: [[float(index) for index in range(8)]],
    }

    coefficients_8: Float32[ndarray, "8"] = apple_distortion_coefficients(static_row)

    np.testing.assert_array_equal(coefficients_8, np.arange(8, dtype=np.float32))


def test_distortion_reader_rejects_an_unrecognized_model_label() -> None:
    """The fallback never ignores an unknown generic distortion label."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: ["mystery_model"],
        LEGACY_DISTORTION_COEFFICIENTS_COLUMN: [[float(index) for index in range(8)]],
    }

    with pytest.raises(SystemExit, match="mystery_model"):
        apple_distortion_coefficients(static_row)


@requires_cuda
def test_ultrawide_trt_normals_store_away_from_camera_positive_z() -> None:
    """The 480×640 engine keeps the same signed gaussurf convention as Part 4."""
    height: int = ULTRAWIDE_IMAGE_HW[0]
    width: int = ULTRAWIDE_IMAGE_HW[1]
    x_w: Float32[Tensor, "w"] = torch.linspace(0.0, 255.0, width, dtype=torch.float32, device="cuda")
    y_h: Float32[Tensor, "h"] = torch.linspace(0.0, 255.0, height, dtype=torch.float32, device="cuda")
    red_hw: Float32[Tensor, "h w"] = x_w[None, :].expand(height, width)
    green_hw: Float32[Tensor, "h w"] = y_h[:, None].expand(height, width)
    blue_hw: Float32[Tensor, "h w"] = (red_hw + green_hw) / 2.0
    rgb_bhw3: UInt8[Tensor, "b=1 h=480 w=640 3"] = torch.stack((red_hw, green_hw, blue_hw), dim=-1).to(torch.uint8)[None]
    predictor: MoGeV2TrtNormalPredictor = MoGeV2TrtNormalPredictor(
        image_hw=ULTRAWIDE_IMAGE_HW,
        batch_size=MOGE_INFERENCE_BATCH_SIZE,
    )

    prediction: MoGeV2NormalOutput = predictor(rgb_bhw3)
    toward_hw3: Float32[ndarray, "h=480 w=640 3"] = prediction.normals_bhw3[0].detach().cpu().numpy().astype(np.float32, copy=False)
    away_hw3: Float32[ndarray, "h=480 w=640 3"] = to_away_from_camera(toward_hw3)
    decoded_hw3: Float32[ndarray, "h=480 w=640 3"] = decode_normals_png(encode_normals_png(away_hw3))

    assert float(np.mean(decoded_hw3[..., 2] > 0.0)) > 0.9
