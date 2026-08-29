"""Behavior checks for ultrawide-resolution MoGe normal inference."""

import numpy as np
import pytest
import torch
from arkitscenes_download.ingest.paths import FRAME_SELECTION_ULTRAWIDE
from jaxtyping import Float32, UInt8
from monopriors.models.moge_v2 import MoGeV2NormalOutput, MoGeV2TrtPredictor
from numpy import ndarray
from torch import Tensor

from gauss_surf.apis.ultrawide_signals import ULTRAWIDE_IMAGE_HW, brown_conrady_coefficients
from gauss_surf.contracts import (
    DISTORTION_COEFFICIENTS_COLUMN,
    DISTORTION_MODEL_COLUMN,
    MOGE_INFERENCE_BATCH_SIZE,
    ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN,
)
from gauss_surf.normals_encoding import decode_normals_png, encode_normals_png, to_away_from_camera

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA and TensorRT required")


def test_ultrawide_provenance_is_anchored_to_chosen_rows() -> None:
    """Rectified RGB, raycast depth, and MoGe normals share the chosen timeline."""
    assert f"/{FRAME_SELECTION_ULTRAWIDE}:sharpness" == ULTRAWIDE_CHOSEN_SHARPNESS_COLUMN


def test_canonical_distortion_reads_the_standard_fourteen_vector() -> None:
    """The reader consumes the standard Brown-Conrady coefficient component."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: ["brown_conrady"],
        DISTORTION_COEFFICIENTS_COLUMN: [[float(index) for index in range(14)]],
    }

    coefficients_14: Float32[ndarray, "14"] = brown_conrady_coefficients(static_row, video_id="47115416")

    np.testing.assert_array_equal(coefficients_14, np.arange(14, dtype=np.float32))


def test_unmigrated_apple_distortion_requires_calibration_reingest() -> None:
    """An old Apple-model row fails with the exact migration tools to run."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: ["apple_radial_poly"],
        DISTORTION_COEFFICIENTS_COLUMN: [[float(index) for index in range(8)]],
    }

    with pytest.raises(SystemExit, match="arkitscenes-download-ingest.*gauss-surf-register-segment"):
        brown_conrady_coefficients(static_row, video_id="47115416")


def test_brown_conrady_distortion_requires_fourteen_coefficients() -> None:
    """A stale eight-vector under a Brown label also receives the migration instruction."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: ["brown_conrady"],
        DISTORTION_COEFFICIENTS_COLUMN: [[float(index) for index in range(8)]],
    }

    with pytest.raises(SystemExit, match="arkitscenes-download-ingest.*gauss-surf-register-segment"):
        brown_conrady_coefficients(static_row, video_id="47115416")


def test_distortion_reader_rejects_an_unrecognized_model_label() -> None:
    """An unknown generic label cannot bypass the standard-only contract."""
    static_row: dict[str, object] = {
        DISTORTION_MODEL_COLUMN: ["mystery_model"],
        DISTORTION_COEFFICIENTS_COLUMN: [[float(index) for index in range(14)]],
    }

    with pytest.raises(SystemExit, match="mystery_model.*arkitscenes-download-ingest"):
        brown_conrady_coefficients(static_row, video_id="47115416")


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
    predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(
        heads="normal",
        network_hw_options=(ULTRAWIDE_IMAGE_HW,),
        batch_size=MOGE_INFERENCE_BATCH_SIZE,
    )

    prediction: MoGeV2NormalOutput = predictor.predict_normals(rgb_bhw3)
    toward_hw3: Float32[ndarray, "h=480 w=640 3"] = prediction.normals_bhw3[0].detach().cpu().numpy().astype(np.float32, copy=False)
    away_hw3: Float32[ndarray, "h=480 w=640 3"] = to_away_from_camera(toward_hw3)
    decoded_hw3: Float32[ndarray, "h=480 w=640 3"] = decode_normals_png(encode_normals_png(away_hw3))

    assert float(np.mean(decoded_hw3[..., 2] > 0.0)) > 0.9
