"""Unit tests for the person-centred virtual cameras on synthetic pinholes (no catalog, no GPU)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from simplecv.sensors.camera.vicon_radial import ViconRadialLens

from open4dgs.catalog_clip import roi_grid, undistortion_grid
from open4dgs.roi_cameras import VirtualCamera, fit_virtual_camera, minimal_rotation_to, virtual_to_source_pixels

SOURCE_K: np.ndarray = np.array([[1800.0, 0.0, 1020.0], [0.0, 1790.0, 760.0], [0.0, 0.0, 1.0]])
SOURCE_SIZE: tuple[int, int] = (2056, 1504)


def _pose(yaw_deg: float, pitch_deg: float, centre: tuple[float, float, float]) -> np.ndarray:
    """A world_T_cam with a camera at ``centre`` turned by yaw (about y) then pitch (about x)."""
    yaw, pitch = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    r_y = np.array([[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]])
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]])
    pose = np.eye(4)
    pose[:3, :3] = r_y @ r_x
    pose[:3, 3] = centre
    return pose


def _project(points: np.ndarray, K: np.ndarray, world_T_cam: np.ndarray) -> np.ndarray:
    cam_T_world = np.linalg.inv(world_T_cam)
    in_cam = points @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
    uvw = in_cam @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def _person(seed: int = 0) -> np.ndarray:
    """A 0.5 x 1.8 x 0.4 m box of points about 4 m in front of the camera and off its axis (a small person in a big frame)."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-0.5, 0.5, size=(4000, 3)) * [0.5, 1.8, 0.4] + [1.1, 0.4, 4.0]).astype(np.float32)


def test_minimal_rotation_maps_axis_and_keeps_up() -> None:
    d = np.array([0.4, -0.2, 1.0])
    rotation = minimal_rotation_to(d)
    np.testing.assert_allclose(rotation @ [0.0, 0.0, 1.0], d / np.linalg.norm(d), atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    # A pure yaw adds no roll: the virtual y axis (image down) stays the source's y axis.
    yaw_only = minimal_rotation_to(np.array([0.5, 0.0, 1.0]))
    np.testing.assert_allclose(yaw_only[:, 1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(minimal_rotation_to(np.array([0.0, 0.0, 2.0])), np.eye(3))


@pytest.mark.parametrize(("yaw", "pitch"), [(0.0, 0.0), (20.0, -10.0), (-35.0, 15.0)])
def test_box_centre_at_image_centre_and_points_inside(yaw: float, pitch: float) -> None:
    pose = _pose(yaw, pitch, (0.3, -0.2, 0.1))
    points = (_person() @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)  # same scene in front of every pose
    camera = fit_virtual_camera(points, SOURCE_K, pose, SOURCE_SIZE, pad=0.15)
    width, height = camera.size
    assert width % 2 == 0 and height % 2 == 0
    world_T_virtual = camera.world_T_virtual(pose)
    uv = _project(points.astype(np.float64), camera.K, world_T_virtual)
    # The points' box is centred on the principal point (the image centre), to a thousandth of a pixel.
    np.testing.assert_allclose((uv.min(axis=0) + uv.max(axis=0)) / 2.0, [(width - 1) / 2.0, (height - 1) / 2.0], atol=1e-3)
    assert (uv >= -0.5).all() and (uv[:, 0] <= width - 0.5).all() and (uv[:, 1] <= height - 0.5).all()
    # The person fills a large part of the image, at the source's focal (no upsampling).
    assert camera.K[0, 0] == pytest.approx(np.sqrt(SOURCE_K[0, 0] * SOURCE_K[1, 1]))
    span = uv.max(axis=0) - uv.min(axis=0)
    assert span[0] / width > 0.75 and span[1] / height > 0.75


def test_virtual_pixel_maps_back_to_source_pixel() -> None:
    pose = _pose(10.0, 5.0, (0.0, 0.0, 0.0))
    points = (_person(1) @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    camera = fit_virtual_camera(points, SOURCE_K, pose, SOURCE_SIZE)
    source_uv = _project(points[:50].astype(np.float64), SOURCE_K, pose)
    virtual_uv = _project(points[:50].astype(np.float64), camera.K, camera.world_T_virtual(pose))
    # A source image whose two channels hold each pixel's own coordinates, sampled through the clip's grid (no lens).
    ys, xs = torch.meshgrid(torch.arange(SOURCE_SIZE[1], dtype=torch.float32), torch.arange(SOURCE_SIZE[0], dtype=torch.float32), indexing="ij")
    coordinates = torch.stack([xs, ys])[None]
    grid = roi_grid(None, camera, torch.device("cpu"))
    virtual_image = F.grid_sample(coordinates, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    width, height = camera.size
    at = torch.from_numpy((virtual_uv + 0.5) / [width, height] * 2.0 - 1.0).float().view(1, 1, -1, 2)
    sampled = F.grid_sample(virtual_image, at, mode="bilinear", align_corners=False)[0, :, 0].T.numpy()
    np.testing.assert_allclose(sampled, source_uv, atol=0.05)
    np.testing.assert_allclose(virtual_to_source_pixels(camera, torch.device("cpu"))[0, 0].numpy(), _source_of(camera, 0.0, 0.0), atol=1e-9)


def _source_of(camera: VirtualCamera, x: float, y: float) -> np.ndarray:
    ray = camera.cam_R_virtual @ np.linalg.inv(camera.sample_K) @ [x, y, 1.0]
    pixel = SOURCE_K @ (ray / ray[2])
    return pixel[:2]


def test_identity_when_points_fill_a_centred_view() -> None:
    width, height, focal = 640, 480, 500.0
    K = np.array([[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    pad, depth = 0.15, 5.0
    # A symmetric box [-a, a] padded by pad * 2a on each side spans exactly the image: a (1 + 2 pad) f = w / 2.
    a, b = width / 2.0 / focal / (1.0 + 2.0 * pad), height / 2.0 / focal / (1.0 + 2.0 * pad)
    grid_ab = np.array([[sa * a, sb * b] for sa in (-1.0, 0.0, 1.0) for sb in (-1.0, 0.0, 1.0)])
    points = (np.concatenate([grid_ab, np.ones((len(grid_ab), 1))], axis=1) * depth).astype(np.float32)
    camera = fit_virtual_camera(points, K, np.eye(4), (width, height), pad=pad)
    np.testing.assert_allclose(camera.cam_R_virtual, np.eye(3), atol=1e-12)
    assert camera.size == (width, height) and camera.sample_size == (width, height)
    np.testing.assert_allclose(camera.K, K, atol=1e-9)
    source = virtual_to_source_pixels(camera, torch.device("cpu")).numpy()
    np.testing.assert_allclose(source[3, 7], [7.0, 3.0], atol=1e-9)


def test_max_side_caps_with_even_sizes_and_centred_principal_point() -> None:
    pose = _pose(0.0, 0.0, (0.0, 0.0, 0.0))
    camera = fit_virtual_camera(_person(), SOURCE_K, pose, SOURCE_SIZE, max_side=300)
    assert max(camera.size) <= 300 and max(camera.size) >= 298
    assert camera.size[0] % 2 == 0 and camera.size[1] % 2 == 0
    assert max(camera.sample_size) > 300
    np.testing.assert_allclose(camera.K[:2, 2], [(camera.size[0] - 1) / 2.0, (camera.size[1] - 1) / 2.0])
    np.testing.assert_allclose(camera.K[0, 0] / camera.sample_K[0, 0], camera.size[0] / camera.sample_size[0])


def test_roi_grid_of_an_identity_camera_equals_the_undistortion_grid() -> None:
    lens = ViconRadialLens(centre=np.array([1033.76, 719.77]), radial=np.array([1.98e-8, -1.2e-14, -3.0e-22]), native_wh=(2056, 1504))
    width, height = 96, 70  # a stored grid smaller than the native one exercises the native <-> stored mapping
    K = np.array([[80.0, 0.0, (width - 1) / 2.0], [0.0, 80.0, (height - 1) / 2.0], [0.0, 0.0, 1.0]])
    camera = VirtualCamera(
        source_K=K, source_size=(width, height), cam_R_virtual=np.eye(3), sample_K=K, sample_size=(width, height), K=K, size=(width, height)
    )
    expected = undistortion_grid(lens, (width, height), torch.device("cpu"))
    torch.testing.assert_close(roi_grid(lens, camera, torch.device("cpu")), expected, atol=1e-6, rtol=0.0)


def test_points_behind_the_camera_are_rejected() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [3.0, 0.0, 0.1]], dtype=np.float32)
    with pytest.raises(ValueError, match="in front"):
        fit_virtual_camera(points, SOURCE_K, np.eye(4), SOURCE_SIZE)
