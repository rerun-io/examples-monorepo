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
from posekit.predictions import BoxDetections, HandDetections

pytestmark = pytest.mark.integration


def test_cuda_fisheye_rejects_radii_outside_reference_domain() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    intrinsic = np.array([[500.0, 0.0, 500.0], [0.0, 500.0, 500.0], [0.0, 0.0, 1.0]])
    camera = CalibratedCamera(intrinsic, np.array([-0.2, 0.0, 0.0, 0.0]), 1000, 1000, "KB4")
    cropper = CudaHandCropper(camera, torch.device("cuda:0"))
    pixels = torch.tensor([[[500.0, 750.0], [500.0, 925.0]]], device="cuda", dtype=torch.float64)
    expected = lens_model(camera).unproject(pixels.cpu().numpy())
    assert np.isfinite(expected[0, 0]).all() and np.isnan(expected[0, 1]).all()
    np.testing.assert_allclose(cropper.unproject(pixels).cpu().numpy(), expected, atol=2e-7, rtol=2e-7)


def test_captured_crops_replay_new_boxes_and_keep_prior_outputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    camera = CalibratedCamera(np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]]), np.empty(0), 256, 256, "PINHOLE")
    cropper = CudaHandCropper(camera, torch.device("cuda:0"))
    frame = torch.arange(256, device="cuda", dtype=torch.uint8)[None, None, :, None].expand(1, 256, 256, 3).contiguous()
    boxes = torch.tensor([[50.0, 70.0, 100.0, 130.0]], device="cuda")
    detected = HandDetections(BoxDetections(boxes, torch.ones(1, device="cuda"), torch.zeros(1, device="cuda", dtype=torch.int64)), ("right",))
    first = cropper.prepare(frame, detected, 0.2)
    saved = first.images.clone()
    boxes[:, [0, 2]] += 60.0
    second = cropper.prepare(frame, detected, 0.2)
    torch.testing.assert_close(first.images, saved, rtol=0.0, atol=0.0)
    assert (second.images - first.images).abs().mean().item() > 0.2
    assert len(cropper.graphs) == 1



def test_crop_graphs_depend_on_hand_count_not_side_order() -> None:
    """Batches with new left/right sequences replay one graph and mirror exactly the right hands."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    camera = CalibratedCamera(np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]]), np.empty(0), 256, 256, "PINHOLE")
    cropper = CudaHandCropper(camera, torch.device("cuda:0"))
    frame = torch.arange(256, device="cuda", dtype=torch.uint8)[None, None, :, None].expand(1, 256, 256, 3).contiguous()
    boxes = torch.tensor([[50.0, 70.0, 100.0, 130.0], [140.0, 60.0, 200.0, 120.0]], device="cuda")
    for sides in (("left", "right"), ("right", "left"), ("right", "right"), ("left", "left")):
        detected = HandDetections(BoxDetections(boxes, torch.ones(2, device="cuda"), torch.zeros(2, device="cuda", dtype=torch.int64)), sides)
        replayed = cropper.prepare(frame, detected, 0.2)
        captured = CudaHandCropper(camera, torch.device("cuda:0")).prepare(frame, detected, 0.2)
        torch.testing.assert_close(replayed.images, captured.images, rtol=0.0, atol=0.0)
    assert len(cropper.graphs) == 1

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
