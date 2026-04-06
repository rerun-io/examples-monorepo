"""Unit tests for mast3r_slam.geometry pure functions."""

import torch
from jaxtyping import Float

from mast3r_slam.geometry import backproject, get_pixel_coords, point_to_dist, point_to_ray_dist, project_calib, skew_sym


def test_skew_sym_antisymmetric() -> None:
    """Skew-symmetric matrix should satisfy S^T = -S."""
    x: Float[torch.Tensor, "3"] = torch.tensor([1.0, 2.0, 3.0])
    S: Float[torch.Tensor, "3 3"] = skew_sym(x)
    torch.testing.assert_close(S, -S.T)


def test_skew_sym_cross_product() -> None:
    """[x]_× @ y should equal x × y."""
    x: Float[torch.Tensor, "3"] = torch.tensor([1.0, 0.0, 0.0])
    y: Float[torch.Tensor, "3"] = torch.tensor([0.0, 1.0, 0.0])
    S: Float[torch.Tensor, "3 3"] = skew_sym(x)
    result: Float[torch.Tensor, "3"] = S @ y
    expected: Float[torch.Tensor, "3"] = torch.linalg.cross(x, y)
    torch.testing.assert_close(result, expected)


def test_skew_sym_batched() -> None:
    """Skew-symmetric should work on batched inputs."""
    x: Float[torch.Tensor, "2 3"] = torch.randn(2, 3)
    S: Float[torch.Tensor, "2 3 3"] = skew_sym(x)
    assert S.shape == (2, 3, 3)
    torch.testing.assert_close(S, -S.transpose(-2, -1))


def test_point_to_dist_unit_vector() -> None:
    """Unit vectors should have distance 1."""
    X: Float[torch.Tensor, "3 3"] = torch.eye(3)
    d: Float[torch.Tensor, "3 1"] = point_to_dist(X)
    torch.testing.assert_close(d, torch.ones(3, 1))


def test_point_to_ray_dist_roundtrip() -> None:
    """ray * distance should reconstruct the original point."""
    X: Float[torch.Tensor, "5 3"] = torch.randn(5, 3)
    rd_result = point_to_ray_dist(X, jacobian=False)
    assert isinstance(rd_result, torch.Tensor)
    rd: Float[torch.Tensor, "5 4"] = rd_result
    ray: Float[torch.Tensor, "5 3"] = rd[..., :3]
    dist: Float[torch.Tensor, "5 1"] = rd[..., 3:]
    reconstructed: Float[torch.Tensor, "5 3"] = ray * dist
    torch.testing.assert_close(reconstructed, X, atol=1e-5, rtol=1e-5)


def test_point_to_ray_dist_jacobian_shape() -> None:
    """Jacobian should have shape (..., 4, 3)."""
    X: Float[torch.Tensor, "2 3 3"] = torch.randn(2, 3, 3)
    rd_jac_result = point_to_ray_dist(X, jacobian=True)
    assert isinstance(rd_jac_result, tuple)
    rd, drd_dX = rd_jac_result
    assert rd.shape == (2, 3, 4)
    assert drd_dX.shape == (2, 3, 4, 3)


def test_project_backproject_roundtrip() -> None:
    """project_calib -> backproject should reconstruct the original 3D point."""
    K: Float[torch.Tensor, "3 3"] = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    P: Float[torch.Tensor, "5 3"] = torch.randn(5, 3)
    P[..., 2] = P[..., 2].abs() + 0.5  # positive depth

    img_size: tuple[int, int] = (480, 640)
    proj_result = project_calib(P, K, img_size)
    assert len(proj_result) == 2
    pz, valid = proj_result[0], proj_result[1]

    uv: Float[torch.Tensor, "5 2"] = pz[..., :2]
    z: Float[torch.Tensor, "5 1"] = torch.exp(pz[..., 2:3])  # log(z) -> z
    P_recon: Float[torch.Tensor, "5 3"] = backproject(uv, z, K)

    # Only check valid projections
    valid_mask: torch.Tensor = valid.squeeze(-1)
    if valid_mask.any():
        torch.testing.assert_close(P_recon[valid_mask], P[valid_mask], atol=1e-4, rtol=1e-4)


def test_get_pixel_coords_shape() -> None:
    """Pixel coords should have shape (b, h, w, 2)."""
    uv: Float[torch.Tensor, "2 3 4 2"] = get_pixel_coords(2, (3, 4), device=torch.device("cpu"), dtype=torch.float32)
    assert uv.shape == (2, 3, 4, 2)


def test_get_pixel_coords_values() -> None:
    """Corner pixels should have correct (u, v) values."""
    uv: Float[torch.Tensor, "1 2 3 2"] = get_pixel_coords(1, (2, 3), device=torch.device("cpu"), dtype=torch.float32)
    # Top-left: (0, 0)
    torch.testing.assert_close(uv[0, 0, 0], torch.tensor([0.0, 0.0]))
    # Top-right: (2, 0)
    torch.testing.assert_close(uv[0, 0, 2], torch.tensor([2.0, 0.0]))
    # Bottom-left: (0, 1)
    torch.testing.assert_close(uv[0, 1, 0], torch.tensor([0.0, 1.0]))
