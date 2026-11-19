"""Sphere camera maps: round trips, fields, the cube and look-at frames, and view rendering."""

from __future__ import annotations

import math

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from posekit.ops.sphere import (
    EquirectCamera,
    Kb4Camera,
    MeiCamera,
    PinholeCamera,
    cube_face_rotations,
    look_at,
    pixel_grid,
    sample_sphere,
    view_from_rays,
)

# Clip B (Insta360 X4 Air, 1920 px lenses), lens 0 as dataforge writes it.
LENS0: MeiCamera = MeiCamera.from_calibration(
    "mei_v6",
    1835.3690185546875,
    1835.0146484375,
    958.292236328125,
    965.8672485351562,
    [2.3149399757385254, 1.7453292608261108, 0.9386149644851685, -1.6840213537216187, 2.191660165786743, 3.8576207160949707, 0.0,
     -0.0049596200697124004, 0.0015607599634677172, -0.003307220060378313, 0.003974860068410635, 0.008278979919850826,
     -0.0012531500542536378, 0.024164050817489624, -0.0014354699524119496],
)  # fmt: skip
# FullCircle's rear lens (masking/mapping.txt, 2880 px), the synthetic fisheye's model.
FULLCIRCLE_REAR: Kb4Camera = Kb4Camera(774.971, 775.267, 1440.0, 1440.0, (0.0908432, -0.0450016, 0.017596, -0.00323419), radius_px=1440 - 107.5)


def _rays(n: int, seed: int) -> torch.Tensor:
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    rays: torch.Tensor = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    return rays / rays.norm(dim=1, keepdim=True)


def test_mei_axis_projects_to_principal_point() -> None:
    pixels, valid = LENS0.project(torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    assert bool(valid[0])
    np.testing.assert_allclose(pixels[0].numpy(), [LENS0.cx, LENS0.cy], atol=1e-9)


def test_mei_field_is_the_half_field_cone() -> None:
    inside: float = LENS0.half_field_rad - 1e-3
    outside: float = LENS0.half_field_rad + 1e-3
    rays: torch.Tensor = torch.tensor([[math.sin(inside), 0.0, math.cos(inside)], [math.sin(outside), 0.0, math.cos(outside)]], dtype=torch.float64)
    _, valid = LENS0.project(rays)
    assert valid.tolist() == [True, False]


def test_mei_round_trip_over_the_field() -> None:
    rays: torch.Tensor = _rays(20000, 0)
    pixels, valid = LENS0.project(rays)
    back, back_valid = LENS0.unproject(pixels[valid])
    assert bool(back_valid.all())
    angle: torch.Tensor = torch.acos((back * rays[valid]).sum(dim=1).clamp(-1.0, 1.0))
    assert float(angle.max()) < 1e-6
    # The 200-degree lens sees a bit more than half the sphere.
    assert 0.55 < float(valid.double().mean()) < 0.62


@settings(max_examples=50, deadline=None)
@given(st.floats(0.0, 1.7), st.floats(-math.pi, math.pi))
def test_kb4_round_trip(theta: float, phi: float) -> None:
    ray: torch.Tensor = torch.tensor([[math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]], dtype=torch.float64)
    pixels, valid = FULLCIRCLE_REAR.project(ray)
    if not bool(valid[0]):
        return
    back, back_valid = FULLCIRCLE_REAR.unproject(pixels)
    assert bool(back_valid[0])
    np.testing.assert_allclose(back[0].numpy(), ray[0].numpy(), atol=1e-7)


def test_kb4_scaled_keeps_rays() -> None:
    small: Kb4Camera = FULLCIRCLE_REAR.scaled(512 / 2880)
    rays: torch.Tensor = _rays(1000, 1)
    big_px, big_valid = FULLCIRCLE_REAR.project(rays)
    small_px, small_valid = small.project(rays)
    both: torch.Tensor = big_valid & small_valid
    np.testing.assert_allclose(((small_px[both] + 0.5) * 2880 / 512 - 0.5).numpy(), big_px[both].numpy(), atol=1e-6)
    assert int((big_valid ^ small_valid).sum()) <= 2


def test_equirect_and_pinhole_round_trip() -> None:
    for camera in (EquirectCamera(64, 32), PinholeCamera(33, 90.0)):
        pixels: torch.Tensor = pixel_grid(32, 32, "cpu")
        rays, valid = camera.unproject(pixels)
        back, back_valid = camera.project(rays)
        assert bool(valid.all()) and bool(back_valid.all())
        np.testing.assert_allclose(back.numpy(), pixels.numpy(), atol=1e-9)


def test_pixel_grid_scale_centres() -> None:
    grid: torch.Tensor = pixel_grid(1, 2, "cpu", scale=(4.0, 4.0))
    np.testing.assert_allclose(grid.numpy(), [[1.5, 1.5], [5.5, 1.5]])
    anisotropic: torch.Tensor = pixel_grid(2, 2, "cpu", scale=(5.5, 4.0))
    np.testing.assert_allclose(anisotropic.numpy(), [[2.25, 1.5], [7.75, 1.5], [2.25, 5.5], [7.75, 5.5]])


def test_cube_faces_are_rotations_covering_the_sphere() -> None:
    rig_R_faces: np.ndarray = cube_face_rotations()
    for rotation in rig_R_faces:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-12
    axes: np.ndarray = rig_R_faces[:, :, 2]
    np.testing.assert_allclose(axes, [[0, 0, 1], [0, 0, -1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]])
    # Image down is rig +Y on the side faces.
    np.testing.assert_allclose(rig_R_faces[:4, :, 1], np.tile([0.0, 1.0, 0.0], (4, 1)))
    rays: torch.Tensor = _rays(5000, 2)
    face: PinholeCamera = PinholeCamera(64)
    seen: torch.Tensor = torch.zeros(5000, dtype=torch.bool)
    for rotation in rig_R_faces:
        _, valid = face.project(rays @ torch.from_numpy(rotation))
        seen |= valid
    assert bool(seen.all())


@settings(max_examples=50, deadline=None)
@given(st.floats(-1, 1), st.floats(-1, 1), st.floats(-1, 1))
def test_look_at_points_z_along_direction(x: float, y: float, z: float) -> None:
    direction: np.ndarray = np.array([x, y, z])
    if np.linalg.norm(direction) < 1e-3:
        return
    rotation: np.ndarray = look_at(direction, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(rotation[:, 2], direction / np.linalg.norm(direction), atol=1e-9)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-9


def test_view_from_two_back_to_back_lenses_picks_the_nearer_axis() -> None:
    lens: Kb4Camera = FULLCIRCLE_REAR.scaled(64 / 2880)
    rig_R_cams: torch.Tensor = torch.from_numpy(np.stack([np.eye(3), np.diag([-1.0, 1.0, -1.0])]))
    equirect: EquirectCamera = EquirectCamera(32, 16)
    rays, _ = equirect.unproject(pixel_grid(16, 32, "cpu"))
    view = view_from_rays(rays, 16, 32, [lens, lens], rig_R_cams, [(64, 64), (64, 64)])
    # Constant images 10 and 20: the panorama is 10 where lens 0 (+Z) is nearer, 20 where lens 1 (-Z) is.
    images: list[torch.Tensor] = [torch.full((1, 64, 64), 10.0), torch.full((1, 64, 64), 20.0)]
    pano: torch.Tensor = sample_sphere(images, view)[0]
    expected: torch.Tensor = torch.where(rays[:, 2] >= 0, 10.0, 20.0).reshape(16, 32).float()
    torch.testing.assert_close(pano, expected, atol=1e-4, rtol=0.0)
