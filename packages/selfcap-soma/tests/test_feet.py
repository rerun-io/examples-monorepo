import numpy as np
import torch

from selfcap_soma.feet import contact_plan, foot_pose_penalty, symmetry_penalty


def test_contact_requires_near_floor_slow_confident_points() -> None:
    times = np.arange(20, dtype=np.float64) * 0.1
    points = np.zeros((20, 6, 3), dtype=np.float64)
    points[..., 1] = 0.5
    scores = np.ones((20, 6), dtype=np.float64)
    points[:, 1, 1] = 0.7  # Above the floor.
    points[:, 2, 0] = times  # Moving at 1 m/s.
    scores[:, 3] = 0.4
    points[:, 4] = np.nan
    plan = contact_plan(times, points, scores)
    assert abs(plan.floor_height_m - 0.5) < 1e-8
    assert plan.active[:, [0, 5]].all()
    assert not plan.active[:, [1, 2, 3, 4]].any()


def test_pose_limits_and_symmetry_have_rest_zero_and_finite_gradients() -> None:
    poses = torch.zeros(2, 2, 3, requires_grad=True)
    axes = torch.eye(3).repeat(2, 1, 1)
    assert foot_pose_penalty(poses, axes).item() == 0.0
    bad = torch.tensor([[[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]], requires_grad=True)
    loss = foot_pose_penalty(bad, axes)
    assert loss.item() > 1.0
    yaw = torch.tensor([[[0.0, 0.3, 0.0], [0.0, 0.3, 0.0]]])
    assert foot_pose_penalty(yaw, axes, toe_joints=True) > foot_pose_penalty(yaw, axes)
    loss.backward()
    assert torch.isfinite(bad.grad).all()
    assert symmetry_penalty(torch.tensor([[0.4, 0.4, 0.3, 0.3, 0.45, 0.45, 0.4, 0.4]])).item() == 0.0
    unequal = torch.tensor([[0.35, 0.45, 0.3, 0.3, 0.45, 0.45, 0.4, 0.4]], requires_grad=True)
    penalty = symmetry_penalty(unequal)
    assert penalty.item() > 1.0
    penalty.backward()
    assert torch.isfinite(unequal.grad).all()


def test_missing_or_unbracketed_points_never_create_contact() -> None:
    times = np.array([0.0, 1.0, 2.0])
    points = np.full((3, 6, 3), np.nan)
    scores = np.ones((3, 6))
    missing = contact_plan(times, points, scores)
    assert missing.floor_height_m is None
    assert not missing.active.any()
    points[:] = [0.0, 0.5, 0.0]
    gapped = contact_plan(times, points, scores, 0.5)
    assert not gapped.active.any()
