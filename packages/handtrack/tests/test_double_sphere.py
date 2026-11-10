"""The double-sphere lens (Gen-HumanEgo's head cameras) in handtrack's CameraRig: projection and its inverse."""

import numpy as np
import pytest
import torch

from handtrack.geometry.camera import CameraRig, project
from handtrack.labels.perspective import unproject

FX, FY, CX, CY, XI, ALPHA = 513.49, 515.35, 835.68, 658.48, 0.00152, 0.570


def _rig() -> CameraRig:
    return CameraRig(names=("/world/rig_00/cam_00",), image_size=torch.tensor([[1600.0, 1300.0]]), cam_from_rig=torch.eye(4)[None],
                     focal=torch.tensor([[FX, FY]]), principal=torch.tensor([[CX, CY]]), fisheye62=None, double_sphere=torch.tensor([[XI, ALPHA]]))


def _reference(points: np.ndarray) -> np.ndarray:
    """Usenko et al. 2018 eq. 40-45, written out (dataforge.das_ego.source.DoubleSphere.project)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    d1 = np.sqrt(x * x + y * y + z * z)
    zz = XI * d1 + z
    d2 = np.sqrt(x * x + y * y + zz * zz)
    denominator = ALPHA * d2 + (1 - ALPHA) * zz
    return np.stack([FX * x / denominator + CX, FY * y / denominator + CY], axis=-1)


def test_double_sphere_projection_matches_the_reference_formula() -> None:
    points = torch.tensor([[0.0, 0.0, 1.0], [0.3, -0.2, 0.5], [-0.4, 0.1, 0.3], [1.0, 0.5, 0.2]])
    mine = project(_rig(), points[None])[0].numpy()
    np.testing.assert_allclose(mine, _reference(points.double().numpy()), atol=1e-3)


def test_unproject_inverts_double_sphere_projection() -> None:
    rays = torch.nn.functional.normalize(torch.tensor([[0.0, 0.0, 1.0], [0.3, -0.2, 0.5], [-0.4, 0.1, 0.3], [0.6, 0.4, 0.4]]), dim=-1)
    pixels = project(_rig(), rays[None])[0]
    back = unproject(_rig(), pixels, iterations=20)
    torch.testing.assert_close(back, rays, atol=1e-4, rtol=0)


def test_rig_keeps_double_sphere_through_select_cat_and_to() -> None:
    rig = CameraRig.cat([_rig(), _rig()]).select([1]).to("cpu")
    assert rig.double_sphere is not None and rig.double_sphere.shape == (1, 2)


def test_rig_rejects_two_lens_models_on_one_camera() -> None:
    with pytest.raises(ValueError, match="one lens model"):
        CameraRig(names=("a",), image_size=torch.ones(1, 2), cam_from_rig=torch.eye(4)[None], focal=torch.ones(1, 2), principal=torch.ones(1, 2),
                  fisheye62=torch.zeros(1, 8), double_sphere=torch.zeros(1, 2))
