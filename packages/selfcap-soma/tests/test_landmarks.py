import torch

from selfcap_soma.landmarks import ear_vertices, rotation_delta


def test_ear_skinning_applies_weighted_rigid_transform_and_keeps_gradients():
    rest = torch.zeros(1, 4505, 3)
    weights = torch.ones(4505, 1)
    inverse = torch.eye(4)[None, None]
    transform = torch.eye(4)[None, None].clone().requires_grad_()
    with torch.no_grad():
        transform[0, 0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    ears = ear_vertices(rest, weights, inverse, transform)
    torch.testing.assert_close(ears, torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]))
    ears.sum().backward()
    assert torch.isfinite(transform.grad).all()


def test_attachment_rotation_is_proper_and_differentiable_at_zero():
    vector = torch.zeros(3, requires_grad=True)
    rotation = rotation_delta(vector)
    torch.testing.assert_close(rotation, torch.eye(3))
    rotation[0, 1].backward()
    assert torch.isfinite(vector.grad).all()
    torch.testing.assert_close(vector.grad, torch.tensor([0.0, 0.0, -1.0]))
