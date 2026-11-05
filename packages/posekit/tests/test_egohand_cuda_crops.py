"""Calibrated GPU geometry agrees with the released inverse lens and crop code."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("warp", reason="GPU calibrated crops require posekit-hands")

from egohand_inference.camera.perspective_crop import perspective_crop_geometry
from posekit.apis.hands_bench import EXAMPLES
from posekit.cameras import CalibratedCamera, LensModel, load_calibrated_camera
from posekit.models.egohand_cuda_crops import CudaHandCropper
from posekit.models.egohand_geometry import lens_model

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("lens", ["RATIONAL_POLYNOMIAL", "PINHOLE", "KB4", "FISHEYE624"])
def test_cuda_inverse_and_virtual_camera_match_reference(lens: LensModel) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    camera_path: Path = EXAMPLES / "selfcap_oak_left.rrd"
    if not camera_path.is_file():
        pytest.skip(f"missing camera fixture: {camera_path}")
    reference = load_calibrated_camera(camera_path)
    coefficients = {"PINHOLE": [], "KB4": [0.05, -0.01, 0.001, -0.0001], "FISHEYE624": [0.04, -0.01, 0.001, 0.0, 0.0, 0.0, 0.001, -0.001, 0.0002, 0.0, -0.0001, 0.0]}
    camera = reference if lens == "RATIONAL_POLYNOMIAL" else CalibratedCamera(reference.intrinsics, np.asarray(coefficients[lens], dtype=np.float64), 1280, 720, lens)
    cropper = CudaHandCropper(camera, torch.device("cuda:0"))
    pixels = torch.tensor([[[300.25, 520.75], [640.5, 361.5], [920.75, 285.25]]], device="cuda", dtype=torch.float64)
    rays = cropper.unproject(pixels)
    np.testing.assert_allclose(rays.cpu().numpy(), lens_model(camera).unproject(pixels.cpu().numpy()), atol=2e-7, rtol=2e-7)
    boxes = torch.tensor([[250.0, 420.0, 390.0, 570.0], [800.0, 200.0, 950.0, 365.0]], device="cuda")
    focal, rotation = cropper.geometry(boxes, 0.2)
    for index, box in enumerate(boxes.cpu().numpy()):
        center = (box[:2] + box[2:]) / 2.0
        bbox = np.array([*center, float(np.max(box[2:] - box[:2])) * 1.2], dtype=np.float32)
        intrinsic, _, expected_rotation = perspective_crop_geometry(bbox, lens_model(camera), 256)
        np.testing.assert_allclose(rotation[index].cpu().numpy(), expected_rotation, atol=1e-6)
        np.testing.assert_allclose(float(focal[index]), intrinsic[0, 0], atol=0.001)
