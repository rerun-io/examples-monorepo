"""WiLoR hands moved from its virtual camera into the run's pinhole: only the translation changes (synthetic, CPU)."""

import numpy as np
import pytest
import torch

pytest.importorskip("ultralytics", reason="posekit.models.wilor needs the hands env")

from posekit.models.wilor import WilorHands, fit_translation
from posekit.predictions import Keypoints3d
from posekit.skeletons import HAND_21

K = torch.tensor([[565.0, 0.0, 360.0], [0.0, 565.0, 465.0], [0.0, 0.0, 1.0]])


def _project(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    return (points[..., :2] / points[..., 2:]) * torch.stack([intrinsics[0, 0], intrinsics[1, 1]]) + intrinsics[:2, 2]


def _hand_shapes(count: int) -> torch.Tensor:
    """Wrist-relative joints of hand size (about 8 cm), wrist at the origin."""
    generator = torch.Generator().manual_seed(0)
    points = (torch.rand((count, 21, 3), generator=generator) - 0.5) * 0.16
    return points - points[:, :1]


def test_fit_translation_recovers_a_known_translation() -> None:
    shapes = _hand_shapes(3)
    truth = torch.tensor([[0.10, -0.05, 0.60], [-0.20, 0.10, 1.20], [0.0, 0.0, 0.35]])
    uv = _project(shapes + truth[:, None], K)
    assert torch.allclose(fit_translation(shapes, uv, K), truth, atol=1e-4)


def test_in_pinhole_keeps_the_hand_and_its_2d_joints_and_only_moves_it() -> None:
    shapes = _hand_shapes(2)
    virtual = torch.tensor([[0.0, 0.0, 18000.0 / 565.0 * 0.6], [0.1, 0.0, 20.0]])
    virtual_k = torch.tensor([[18000.0, 0.0, 360.0], [0.0, 18000.0, 465.0], [0.0, 0.0, 1.0]])
    xyz = shapes + virtual[:, None]
    xy = _project(xyz, virtual_k)
    vertices = torch.cat([xyz, xyz + 0.01], dim=1)
    hands = WilorHands(
        keypoints=Keypoints3d(xy, xyz[..., 2] - xyz[:, :1, 2], torch.ones((2, 21)), torch.tensor([0, 0]), HAND_21, (0,), xyz),
        sides=("left", "right"),
        vertices=vertices,
        global_orient=torch.zeros((2, 1, 3)),
        hand_pose=torch.zeros((2, 15, 3)),
        betas=torch.zeros((2, 10)),
        cam_t_full=virtual,
        focal_length=torch.full((2,), 18000.0),
    )
    moved = hands.in_pinhole(K.double().numpy())
    assert moved.keypoints.xyz_camera is not None and moved.vertices is not None
    t = moved.cam_t_full
    assert torch.allclose(moved.keypoints.xyz_camera - t[:, None], shapes, atol=1e-5)
    assert torch.allclose(moved.vertices - t[:, None], vertices - virtual[:, None], atol=1e-5)
    assert torch.equal(moved.keypoints.xy, xy)
    # The hand whose 2D size matches 0.6 m in K lands at about 0.6 m (not 19 m out as in the virtual camera).
    assert float(t[0, 2]) == pytest.approx(0.6, rel=0.1)
    assert np.allclose(moved.focal_length.numpy(), 565.0)
