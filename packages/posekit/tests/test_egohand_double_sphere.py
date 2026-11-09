"""Double-sphere camera support preserves source pixels and physical rays."""

import numpy as np
import pytest
from jaxtyping import Float32, Float64
from numpy import ndarray

from posekit.cameras import CalibratedCamera
from posekit.models.egohand_double_sphere import DoubleSphereCamera


def test_double_sphere_matches_catalog_projection_and_inverts() -> None:
    k: Float64[ndarray, "3 3"] = np.array([[513.0, 0.0, 823.0], [0.0, 516.0, 646.0], [0.0, 0.0, 1.0]])
    camera: CalibratedCamera = CalibratedCamera(k, np.array([-0.0037, 0.573]), 1600, 1300, "DOUBLE_SPHERE")
    model: DoubleSphereCamera = DoubleSphereCamera(camera)
    rays: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [-0.6, 0.2, 0.5], [1.2, -0.8, 0.3]])
    expected: Float64[ndarray, "3 2"] = model.project(rays)
    np.testing.assert_allclose(model.unproject(expected), rays / np.linalg.norm(rays, axis=-1, keepdims=True), atol=1e-10)
    assert np.isnan(model.unproject(np.array([[1e6, 1e6]]))).all()


@pytest.mark.integration
def test_double_sphere_cuda_matches_reference() -> None:
    pytest.importorskip("warp", reason="requires posekit-hands CUDA camera support")
    pytest.importorskip("egohand_inference", reason="requires posekit-hands camera adapters")
    import torch
    from torch import Tensor

    from posekit.models.egohand_cuda_crops import CudaHandCropper

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    k: Float64[ndarray, "3 3"] = np.array([[513.0, 0.0, 823.0], [0.0, 516.0, 646.0], [0.0, 0.0, 1.0]])
    camera: CalibratedCamera = CalibratedCamera(k, np.array([-0.0037, 0.573]), 1600, 1300, "DOUBLE_SPHERE")
    model: DoubleSphereCamera = DoubleSphereCamera(camera)
    cropper: CudaHandCropper = CudaHandCropper(camera, torch.device("cuda:0"))
    pixels: Float64[Tensor, "1 4 2"] = torch.tensor([[[823.0, 646.0], [400.0, 900.0], [1200.0, 250.0], [1e6, 1e6]]], device="cuda", dtype=torch.float64)
    np.testing.assert_allclose(cropper.unproject(pixels).cpu(), model.unproject(pixels.cpu().numpy()), atol=2e-7)
    points: Float32[Tensor, "1 25 3"] = torch.tensor([[[0.0, 0.0, 1.0], [-0.6, 0.2, 0.5], [1.2, -0.8, 0.3]]], device="cuda").repeat(1, 9, 1)[:, :25].contiguous()
    np.testing.assert_allclose(cropper.project(points).cpu(), model.project(points.cpu().numpy().astype(np.float64)), atol=1e-4)


def test_double_sphere_stereographic_special_case() -> None:
    k: Float64[ndarray, "3 3"] = np.array([[300.0, 0.0, 500.0], [0.0, 300.0, 400.0], [0.0, 0.0, 1.0]])
    model: DoubleSphereCamera = DoubleSphereCamera(CalibratedCamera(k, np.array([0.0, 0.5]), 1000, 800, "DOUBLE_SPHERE"))
    rays: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 4.0], [0.0, 4.0, 3.0]])
    np.testing.assert_allclose(model.project(rays), [[500.0, 400.0], [700.0, 400.0], [500.0, 700.0]], atol=1e-10)
