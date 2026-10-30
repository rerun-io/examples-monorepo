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
