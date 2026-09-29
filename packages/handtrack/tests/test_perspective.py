import math

import pytest
import torch

from handtrack.geometry.camera import CameraRig, project
from handtrack.geometry.letterbox import letterbox_for
from handtrack.labels.perspective import (
    CROP_CENTRE,
    CROP_MARGIN,
    CropCameras,
    aim,
    crop_cameras,
    jitter,
    local_crop_from_net,
    look_at,
    sample_crops,
    to_crop,
    unproject,
)


def _pinhole(width: int = 640, height: int = 480, focal: float = 300.0) -> CameraRig:
    return CameraRig(names=("/cam",), image_size=torch.tensor([[float(width), float(height)]]), cam_from_rig=torch.eye(4)[None],
                     focal=torch.tensor([[focal, focal]]), principal=torch.tensor([[(width - 1) / 2, (height - 1) / 2]]), fisheye62=None)


def _fisheye() -> CameraRig:
    coefficients = torch.tensor([[0.30, -0.10, 0.05, -0.01, 0.002, 0.0, 1e-4, -2e-4]])
    return CameraRig(names=("/fish",), image_size=torch.tensor([[640.0, 480.0]]), cam_from_rig=torch.eye(4)[None],
                     focal=torch.tensor([[240.0, 240.0]]), principal=torch.tensor([[320.5, 238.0]]), fisheye62=coefficients)


def _hand(centre: tuple[float, float, float], spread: float = 0.05) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.tensor(centre) + spread * (torch.rand(21, 3) - 0.5)


def test_look_at_puts_the_direction_on_the_crop_axis_and_roll_turns_about_it() -> None:
    direction = torch.tensor([[0.3, -0.2, 1.0], [-0.5, 0.4, 0.8]])
    rotation = look_at(direction, torch.tensor([0.0, 0.7]))
    unit = direction / direction.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(torch.einsum('nij,nj->ni', rotation, unit), torch.tensor([[0.0, 0.0, 1.0]] * 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand(2, 3, 3), atol=1e-6, rtol=0)
    plain, rolled = look_at(direction[1:], torch.tensor([0.0])), rotation[1:]
    torch.testing.assert_close(rolled, (plain.transpose(-1, -2) @ torch.tensor([[[math.cos(0.7), -math.sin(0.7), 0.0], [math.sin(0.7), math.cos(0.7), 0.0], [0.0, 0.0, 1.0]]])).transpose(-1, -2),
                               atol=1e-6, rtol=0)


def test_crop_cameras_centre_the_hand_and_fit_it_with_the_margin() -> None:
    points = _hand((0.12, -0.05, 0.45))[None]
    cameras = crop_cameras(points, torch.ones(1, 21, dtype=torch.bool), torch.zeros(1), torch.tensor([False]))
    uv, depth = to_crop(cameras, points)
    assert bool((depth > 0).all())
    extent = (uv - CROP_CENTRE).abs().amax()
    assert float(extent) == pytest.approx(CROP_CENTRE / CROP_MARGIN, rel=1e-4)  # the farthest point sits at 1/1.2 of the half side
    centre = (uv.amin(dim=1) + uv.amax(dim=1)) / 2
    assert float((centre - CROP_CENTRE).abs().max()) < 3.0  # aimed at the 3D box centre: close to the crop centre


def test_mirroring_flips_crop_x_only() -> None:
    points = _hand((0.0, 0.0, 0.5))[None].repeat(2, 1, 1)
    cameras = crop_cameras(points, torch.ones(2, 21, dtype=torch.bool), torch.zeros(2), torch.tensor([False, True]))
    uv, _ = to_crop(cameras, points)
    torch.testing.assert_close(uv[1, :, 0], (96 - 1) - uv[0, :, 0], atol=1e-4, rtol=0)
    torch.testing.assert_close(uv[1, :, 1], uv[0, :, 1], atol=1e-4, rtol=0)


def test_invalid_or_behind_rows_get_no_focal() -> None:
    points = torch.stack([_hand((0.0, 0.0, 0.5)), _hand((0.0, 0.0, -0.5))])
    valid = torch.ones(2, 21, dtype=torch.bool)
    valid[0] = False
    cameras = crop_cameras(points, valid, torch.zeros(2), torch.tensor([False, False]))
    assert not bool(torch.isfinite(cameras.focal).any())


@pytest.mark.parametrize('rig', [_pinhole(), _fisheye()], ids=['pinhole', 'fisheye62'])
def test_a_bright_point_lands_where_its_label_says(rig: CameraRig) -> None:
    point = torch.tensor([[0.07, -0.04, 0.4]])
    pixel = project(rig, point[None, None])[0, 0, 0]
    frame = torch.zeros(1, 480, 640, dtype=torch.uint8)
    x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
    frame[0, y - 1 : y + 2, x - 1 : x + 2] = 255
    exact = unproject(rig, torch.tensor([[float(x), float(y)]])) * 0.4 / unproject(rig, torch.tensor([[float(x), float(y)]]))[:, 2:]
    hand = exact[None] + 0.03 * (torch.rand(1, 21, 3) - 0.5)
    for mirror in (False, True):
        cameras = crop_cameras(hand, torch.ones(1, 21, dtype=torch.bool), torch.tensor([0.4]), torch.tensor([mirror]))
        crop = sample_crops(frame, torch.zeros(1, dtype=torch.int64), cameras, rig)[0, 0]
        uv, _ = to_crop(cameras, exact[None])
        bright = crop > 0.5 * crop.max()  # the 3x3 blob is magnified about 4x: use its centroid, not the first maximum
        ys, xs = torch.nonzero(bright, as_tuple=True)
        centroid = torch.stack([xs.float().mean(), ys.float().mean()])
        assert float((centroid - uv[0, 0]).norm()) < 1.5


@pytest.mark.parametrize('rig', [_pinhole(), _fisheye()], ids=['pinhole', 'fisheye62'])
def test_unproject_inverts_project(rig: CameraRig) -> None:
    pixels = torch.tensor([[20.0, 30.0], [320.0, 240.0], [600.0, 450.0], [100.0, 400.0]])
    rays = unproject(rig, pixels)
    torch.testing.assert_close(project(rig, rays[None, None])[0, 0], pixels, atol=1e-2, rtol=0)
    torch.testing.assert_close(rays.norm(dim=-1), torch.ones(4), atol=1e-6, rtol=0)


def test_aim_moves_the_axis_to_the_requested_pixel_and_jitter_keeps_cameras_valid() -> None:
    points = _hand((0.02, 0.01, 0.5))[None]
    cameras = crop_cameras(points, torch.ones(1, 21, dtype=torch.bool), torch.zeros(1), torch.tensor([False]))
    target = torch.tensor([[20.0, -10.0]])
    moved = CropCameras(aim(cameras.rotation, target, cameras.focal, torch.zeros(1)), cameras.focal, cameras.mirror)
    old_axis = cameras.rotation[0, 2]  # the old crop camera's axis in the camera frame
    new_uv, _ = to_crop(cameras, moved.rotation[:, 2][:, None] * 0.5)
    torch.testing.assert_close(new_uv[0, 0], CROP_CENTRE + target[0], atol=1e-3, rtol=0)
    assert float(old_axis.norm()) == pytest.approx(1.0)
    shaken = jitter(cameras.select(torch.zeros(64, dtype=torch.int64)), torch.Generator().manual_seed(0), 0.5, (0.9, 1.25), 0.1)
    assert bool(torch.isfinite(shaken.focal).all()) and float(shaken.focal.max()) <= float(cameras.focal[0]) / 0.9 + 1e-3
    uv, _ = to_crop(shaken, points.repeat(64, 1, 1))
    # The aim shift is at most 0.1 side per axis (rolled with the crop, so up to sqrt(2) of that on one axis), magnified 1/0.9 at most.
    assert float((uv.mean(dim=1) - CROP_CENTRE).abs().max()) < 0.1 * 96 * math.sqrt(2) / 0.9 + 4.0


def test_local_affine_matches_the_crop_near_its_centre() -> None:
    rig, letterbox = _pinhole(), letterbox_for(640, 480)
    points = _hand((0.05, 0.02, 0.5))[None].repeat(2, 1, 1)
    cameras = crop_cameras(points, torch.ones(2, 21, dtype=torch.bool), torch.zeros(2), torch.tensor([False, True]))
    affine = local_crop_from_net(cameras, rig, letterbox)
    near = cameras.rotation[:, 2][:, None] * 0.5 + 0.002 * torch.tensor([[[1.0, -0.5, 0.0]]])
    uv, _ = to_crop(cameras, near)
    net = letterbox.to_net(project(rig, near[:, None])[:, 0])
    mapped = torch.einsum('nij,nkj->nki', affine[:, :2, :2], net) + affine[:, None, :2, 2]
    torch.testing.assert_close(mapped, uv, atol=0.05, rtol=0)


def test_unproject_survives_a_saturated_lens() -> None:
    far = torch.tensor([[-4000.0, -3000.0], [320.0, 240.0]])  # far outside a fisheye's valid radius: a singular Jacobian
    rays = unproject(_fisheye(), far)
    assert bool(torch.isfinite(rays).all())
    torch.testing.assert_close(project(_fisheye(), rays[None, None])[0, 0, 1], far[1], atol=1e-2, rtol=0)
