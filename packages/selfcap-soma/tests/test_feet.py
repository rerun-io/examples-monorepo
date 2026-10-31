import torch

from selfcap_soma.feet import foot_pose_penalty, symmetry_penalty


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
