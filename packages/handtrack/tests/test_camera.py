import numpy as np
import torch
from simplecv.camera_parameters import KannalaBrandtDistortion, apply_radial_tangential_distortion

from handtrack.geometry.camera import CameraRig, in_front, inside_image, project, transform_points, world_to_cameras

COEFFS: list[float] = [0.35, -0.02, 0.01, -0.004, 0.0008, -0.00005, 0.0003, -0.0002]


def _rig(fisheye: bool) -> CameraRig:
    return CameraRig(
        names=("/world/rig_00/cam_00", "/world/rig_00/cam_01"),
        image_size=torch.tensor([[640.0, 480.0], [640.0, 480.0]]),
        cam_from_rig=torch.eye(4).repeat(2, 1, 1),
        focal=torch.tensor([[240.0, 241.0], [250.0, 249.0]]),
        principal=torch.tensor([[320.0, 240.0], [318.0, 242.0]]),
        fisheye62=torch.tensor([COEFFS, COEFFS]) if fisheye else None,
    )


def test_fisheye62_matches_simplecv_formula() -> None:
    rng: np.random.Generator = np.random.default_rng(0)
    points: np.ndarray = np.column_stack([rng.uniform(-0.3, 0.3, 50), rng.uniform(-0.3, 0.3, 50), rng.uniform(0.05, 0.8, 50)])
    radius: np.ndarray = np.hypot(points[:, 0], points[:, 1])
    normalized: np.ndarray = points[:, :2] * (np.arctan2(radius, points[:, 2]) / radius)[:, None]
    distorted: np.ndarray = apply_radial_tangential_distortion(KannalaBrandtDistortion(*COEFFS), normalized)
    expected: np.ndarray = distorted * [240.0, 241.0] + [320.0, 240.0]
    got: torch.Tensor = project(_rig(fisheye=True), torch.from_numpy(points).float()[None, None].expand(1, 2, -1, -1))
    np.testing.assert_allclose(got[0, 0].double().numpy(), expected, atol=2e-3)


def test_pinhole_and_gradient_at_optical_axis() -> None:
    points: torch.Tensor = torch.tensor([[[[0.0, 0.0, 0.5], [0.1, -0.05, 0.5]]]]).expand(1, 2, 2, 3).clone().requires_grad_(True)
    pixels: torch.Tensor = project(_rig(fisheye=False), points)
    torch.testing.assert_close(pixels[0, 0, 1], torch.tensor([320.0 + 240.0 * 0.2, 240.0 - 241.0 * 0.1]))
    fisheye_pixels: torch.Tensor = project(_rig(fisheye=True), points)
    fisheye_pixels.sum().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    torch.testing.assert_close(fisheye_pixels[0, 0, 0], torch.tensor([320.0, 240.0]))


def test_world_to_cameras_and_masks() -> None:
    rig: CameraRig = _rig(fisheye=True)
    world_from_rig: torch.Tensor = torch.eye(4)
    world_from_rig[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    points_world: torch.Tensor = torch.tensor([[1.0, 2.0, 3.5], [1.0, 2.0, 2.5]])
    points_cam: torch.Tensor = world_to_cameras(rig, world_from_rig, points_world)
    assert points_cam.shape == (2, 2, 3)
    torch.testing.assert_close(points_cam[0], torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]))
    assert in_front(points_cam).tolist() == [[True, False], [True, False]]
    pixels: torch.Tensor = project(rig, points_cam)
    assert inside_image(rig, pixels)[:, 0].all()
    torch.testing.assert_close(transform_points(torch.eye(4), points_world), points_world)
