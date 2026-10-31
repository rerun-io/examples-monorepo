import torch

from selfcap_soma.losses import robust_points


def test_robust_points_masks_nan_and_has_finite_gradient():
    predicted = torch.tensor([[[0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]])
    loss = robust_points(predicted, target, torch.ones(1, 2), 0.01)
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    torch.testing.assert_close(predicted.grad[0, 1], torch.zeros(3))


def test_outlier_growth_is_linear_and_empty_source_is_zero():
    target = torch.zeros(1, 1, 3)
    point = torch.tensor([[[0.04, 0.0, 0.0]]])
    torch.testing.assert_close(robust_points(point, target, torch.ones(1, 1), 0.01), torch.tensor(6.0))
    torch.testing.assert_close(robust_points(point, target, torch.zeros(1, 1), 0.01), torch.tensor(0.0))


def test_body_lengths_include_widths_and_midpoint_torso():
    from selfcap_soma.losses import body_lengths

    body = torch.zeros(1, 12, 3)
    body[0, 0] = torch.tensor([1.0, 3.0, 0.0])
    body[0, 1] = torch.tensor([-1.0, 3.0, 0.0])
    body[0, 6] = torch.tensor([0.5, 0.0, 0.0])
    body[0, 7] = torch.tensor([-0.5, 0.0, 0.0])
    torch.testing.assert_close(body_lengths(body)[0, :3], torch.tensor([2.0, 1.0, 3.0]))


def test_rotation_temporal_term_penalizes_jitter_but_allows_steady_turns():
    from selfcap_soma.losses import rotation_temporal_penalty

    def rotations(angles):
        a = torch.tensor(angles)
        z = torch.zeros_like(a)
        o = torch.ones_like(a)
        return torch.stack((torch.cos(a), -torch.sin(a), z, torch.sin(a), torch.cos(a), z, z, z, o), dim=-1).reshape(-1, 1, 3, 3).requires_grad_()

    dt = torch.full((4,), 0.04)
    steady = rotations([0.0, 0.01, 0.02, 0.03, 0.04])
    jitter = rotations([0.0, 0.02, 0.0, 0.02, 0.0])
    cost = rotation_temporal_penalty(jitter, dt)
    assert cost > rotation_temporal_penalty(steady, dt) * 100
    cost.backward()
    assert torch.isfinite(jitter.grad).all()
    assert rotation_temporal_penalty(rotations([0.0] * 5), dt) == 0


def test_body_acceleration_uses_timestamps_and_does_not_bridge_gaps():
    from selfcap_soma.losses import body_acceleration_penalty

    times = torch.tensor([0.0, 0.04, 0.09, 0.13])
    points = times[:, None, None].expand(-1, 2, 3).clone().requires_grad_()
    assert body_acceleration_penalty(points, times[1:] - times[:-1]) < 1e-10
    jitter = points.detach().clone()
    jitter[1, 0, 0] += 0.02
    jitter.requires_grad_()
    cost = body_acceleration_penalty(jitter, times[1:] - times[:-1])
    assert cost > 1.0
    cost.backward()
    assert torch.isfinite(jitter.grad).all()
    assert body_acceleration_penalty(jitter, torch.full((3,), 0.2)) == 0
