"""mamma's triangulation policy over the shared kernel: hypothesis properties (unit lane, CPU)."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from mamma_fakes import K_ENGINE, camera_ring, project

from mamma.ops.triangulation import triangulate_gated, triangulate_landmarks

POINTS = arrays(np.float64, (12, 3), elements=st.floats(-0.8, 0.8))
LOG_VARIANCE = arrays(np.float32, (4, 12), elements=st.floats(-12.0, -6.0, width=32))


def inputs(points: np.ndarray, views: int) -> tuple[torch.Tensor, torch.Tensor]:
    cam_T_world: np.ndarray = camera_ring(views)
    P: torch.Tensor = torch.as_tensor(K_ENGINE @ cam_T_world[:, :3], dtype=torch.float32)
    return torch.as_tensor(project(points, cam_T_world), dtype=torch.float32), P


@settings(max_examples=40, deadline=None)
@given(points=POINTS, log_variance=LOG_VARIANCE, views=st.integers(2, 4))
def test_recovers_points_projected_into_two_or_more_views(points: np.ndarray, log_variance: np.ndarray, views: int) -> None:
    xy, P = inputs(points, views)
    found, valid = triangulate_landmarks(xy, torch.as_tensor(log_variance[:views]), torch.ones(views, 12), P)
    assert bool(valid.all())
    np.testing.assert_allclose(found.numpy(), points, atol=2e-3)


@settings(max_examples=40, deadline=None)
@given(points=POINTS, seen=arrays(np.bool_, (4, 12)))
def test_points_seen_by_fewer_than_two_views_come_back_as_finite_zeros(points: np.ndarray, seen: np.ndarray) -> None:
    xy, P = inputs(points, 4)
    found, valid = triangulate_landmarks(xy, torch.full((4, 12), -9.0), torch.as_tensor(seen, dtype=torch.float32), P)
    np.testing.assert_array_equal(valid.numpy(), seen.sum(axis=0) >= 2)
    assert bool(torch.isfinite(found).all()) and bool((found[~valid] == 0).all())


@settings(max_examples=30, deadline=None)
@given(points=POINTS, swapped=st.integers(0, 5), side_m=st.floats(0.5, 1.0) | st.floats(-1.0, -0.5))
def test_gate_drops_a_view_that_sees_the_other_person_and_keeps_the_agreeing_ones(points: np.ndarray, swapped: int, side_m: float) -> None:
    """The swapped view's mask jumped to a person standing ``side_m`` beside the subject, across its line of sight.

    Six views: with four at 90 deg the one view that also sees that direction disagrees just as much (no majority to find).
    """
    xy, P = inputs(points, 6)
    angle: float = 2.0 * np.pi * swapped / 6
    other: np.ndarray = points + side_m * np.array([-np.sin(angle), np.cos(angle), 0.0])
    xy[swapped] = torch.as_tensor(project(other, camera_ring(6))[swapped], dtype=torch.float32)
    visibility = torch.ones(6, 12)
    gated, found, valid = triangulate_gated(xy, torch.full((6, 12), -9.0), visibility, P)
    assert bool((gated[swapped] == 0).all())
    np.testing.assert_array_equal(np.delete(gated.numpy(), swapped, axis=0), 1.0)
    np.testing.assert_allclose(found.numpy(), points, atol=2e-3)


@settings(max_examples=30, deadline=None)
@given(points=POINTS)
def test_gate_is_a_no_op_when_every_view_agrees(points: np.ndarray) -> None:
    xy, P = inputs(points, 4)
    visibility = torch.ones(4, 12)
    gated, found, valid = triangulate_gated(xy, torch.full((4, 12), -9.0), visibility, P)
    plain, plain_valid = triangulate_landmarks(xy, torch.full((4, 12), -9.0), visibility, P)
    assert torch.equal(gated, visibility) and torch.equal(found, plain) and torch.equal(valid, plain_valid)
