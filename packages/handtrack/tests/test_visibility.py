import torch

from handtrack.hand.pose import generic_hand_model
from handtrack.labels.visibility import OTHER_HAND_MARGIN_M, flesh_margin, keypoints_hidden, ray_blocked


def _square(z: float, half: float = 0.05) -> torch.Tensor:
    """Two triangles covering [-half, half]^2 at depth z (camera looks down +z)."""
    a, b, c, d = [-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]
    return torch.tensor([[a, b, c], [a, c, d]], dtype=torch.float32)


def test_ray_blocked_only_by_triangles_in_front_within_the_limit() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.5], [0.2, 0.0, 0.5]]])
    wall = _square(0.3)[None]
    blocked = ray_blocked(points, torch.full((1, 2), 0.49), wall)
    assert blocked.tolist() == [[True, False]]  # the second ray passes beside the square
    assert not ray_blocked(points, torch.full((1, 2), 0.25), wall).any()  # the square lies beyond the limit
    assert not ray_blocked(points, torch.full((1, 2), 0.49), _square(0.6)[None]).any()  # behind the point


def test_other_hand_hides_and_missing_hands_hide_nothing() -> None:
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    square = _square(0.3)[[0, 0, 0, 1]].reshape(-1, 3)[[0, 1, 2, 5]]  # the four corners a, b, c, d
    points = torch.full((1, 2, 21, 3), float("nan"))
    points[0, 0] = torch.tensor([0.0, 0.0, 0.5])  # every left keypoint straight behind the right hand's square
    vertices = torch.full((1, 2, 4, 3), float("nan"))
    vertices[0, 1] = square
    margin = torch.full((21,), 0.01)
    hidden = keypoints_hidden(points, vertices, faces, margin)
    assert hidden[0, 0].all() and not hidden[0, 1].any()  # the right hand has no keypoints: no statement
    vertices[0, 1, :, 2] = 0.5 - OTHER_HAND_MARGIN_M / 2  # touching: closer than the margin
    assert not keypoints_hidden(points, vertices, faces, margin).any()
    vertices[0, 1] = float("nan")
    assert not keypoints_hidden(points, vertices, faces, margin).any()


def test_flesh_margin_is_bounded_and_largest_at_the_wrist() -> None:
    margin = flesh_margin(generic_hand_model())
    assert margin.shape == (21,) and bool((margin >= 0.005).all()) and bool((margin <= 0.035).all())
    assert int(margin.argmax()) == 5  # the wrist sits deepest in the flesh
