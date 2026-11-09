"""Double-sphere camera support preserves source pixels and physical rays."""

import numpy as np
import pytest

pytest.importorskip("egohand_inference", reason="requires posekit-hands camera adapters")

from posekit.cameras import CalibratedCamera
from posekit.models.egohand_geometry import lens_model


def test_double_sphere_matches_catalog_projection_and_inverts() -> None:
    k = np.array([[513.0, 0.0, 823.0], [0.0, 516.0, 646.0], [0.0, 0.0, 1.0]])
    camera = CalibratedCamera(k, np.array([-0.0037, 0.573]), 1600, 1300, "DOUBLE_SPHERE")
    model = lens_model(camera)
    rays = np.array([[0.0, 0.0, 1.0], [-0.6, 0.2, 0.5], [1.2, -0.8, 0.3]])
    expected = model.project(rays)
    np.testing.assert_allclose(model.unproject(expected), rays / np.linalg.norm(rays, axis=-1, keepdims=True), atol=1e-10)
    assert np.isnan(model.unproject(np.array([[1e6, 1e6]]))).all()


@pytest.mark.integration
def test_double_sphere_cuda_matches_reference() -> None:
    import torch

    from posekit.models.egohand_cuda_crops import CudaHandCropper

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    k = np.array([[513.0, 0.0, 823.0], [0.0, 516.0, 646.0], [0.0, 0.0, 1.0]])
    camera = CalibratedCamera(k, np.array([-0.0037, 0.573]), 1600, 1300, "DOUBLE_SPHERE")
    model = lens_model(camera)
    cropper = CudaHandCropper(camera, torch.device("cuda:0"))
    pixels = torch.tensor([[[823.0, 646.0], [400.0, 900.0], [1200.0, 250.0], [1e6, 1e6]]], device="cuda", dtype=torch.float64)
    np.testing.assert_allclose(cropper.unproject(pixels).cpu(), model.unproject(pixels.cpu().numpy()), atol=2e-7)
    points = torch.tensor([[[0.0, 0.0, 1.0], [-0.6, 0.2, 0.5], [1.2, -0.8, 0.3]]], device="cuda").repeat(1, 9, 1)[:, :25].contiguous()
    np.testing.assert_allclose(cropper.project(points).cpu(), model.project(points.cpu().numpy().astype(np.float64)), atol=1e-4)


def test_double_sphere_stereographic_special_case() -> None:
    k = np.array([[300.0, 0.0, 500.0], [0.0, 300.0, 400.0], [0.0, 0.0, 1.0]])
    model = lens_model(CalibratedCamera(k, np.array([0.0, 0.5]), 1000, 800, "DOUBLE_SPHERE"))
    rays = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 4.0], [0.0, 4.0, 3.0]])
    np.testing.assert_allclose(model.project(rays), [[500.0, 400.0], [700.0, 400.0], [500.0, 700.0]], atol=1e-10)
