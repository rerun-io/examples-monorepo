"""Properties of the shared weighted DLT kernel (SPEC 5.1): recovery, weight-scale and view-order invariance, the <min_views policy."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from simplecv.ops.triangulate import triangulate_weighted


def ring_projections(n_views: int, dtype: torch.dtype) -> torch.Tensor:
    """Projection matrices K [R | t] of cameras on a 4 m ring around the origin, each looking at it (RDF)."""
    K = torch.tensor([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    projections: list[torch.Tensor] = []
    for angle in torch.arange(n_views, dtype=torch.float64) * (2.0 * torch.pi / n_views):
        center = torch.stack((4.0 * torch.cos(angle), 4.0 * torch.sin(angle), torch.tensor(0.3, dtype=torch.float64)))
        forward = -center / center.norm()
        right = torch.linalg.cross(forward, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
        right = right / right.norm()
        rotation = torch.stack((right, torch.linalg.cross(forward, right), forward))
        projections.append(K @ torch.cat((rotation, (-rotation @ center)[:, None]), dim=1))
    return torch.stack(projections).to(dtype)


def project(points: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.einsum("vij,nj->vni", P, torch.cat((points, torch.ones_like(points[:, :1])), dim=1))
    return homogeneous[..., :2] / homogeneous[..., 2:]


POINTS = arrays(np.float64, st.tuples(st.integers(1, 12), st.just(3)), elements=st.floats(-0.8, 0.8))


@pytest.mark.parametrize(("dtype", "rtol"), [(torch.float64, 1e-6), (torch.float32, 1e-3)])
@settings(max_examples=60, deadline=None)
@given(points=POINTS, n_views=st.integers(3, 8), weights=st.floats(0.05, 4.0))  # two ring cameras face each other: no depth on their baseline
def test_recovers_projected_points(dtype: torch.dtype, rtol: float, points: np.ndarray, n_views: int, weights: float) -> None:
    P = ring_projections(n_views, dtype)
    truth = torch.as_tensor(points, dtype=dtype)
    xy = project(truth, P)
    xyz, valid = triangulate_weighted(xy, torch.full(xy.shape[:2], weights, dtype=dtype), P)
    assert xyz.dtype == dtype and bool(valid.all())
    torch.testing.assert_close(xyz, truth, rtol=rtol, atol=rtol * 4.0)


@settings(max_examples=60, deadline=None)
@given(points=POINTS, scale=st.floats(0.01, 100.0), seed=st.integers(0, 2**16))
def test_invariant_to_weight_scale_and_view_order(points: np.ndarray, scale: float, seed: int) -> None:
    P = ring_projections(5, torch.float64)
    generator = torch.Generator().manual_seed(seed)
    xy = project(torch.as_tensor(points), P) + torch.randn((5, len(points), 2), generator=generator, dtype=torch.float64)
    weight = torch.rand((5, len(points)), generator=generator, dtype=torch.float64) + 0.1
    order = torch.randperm(5, generator=generator)
    base, _ = triangulate_weighted(xy, weight, P)
    scaled, _ = triangulate_weighted(xy, weight * scale, P)
    permuted, _ = triangulate_weighted(xy[order], weight[order], P[order])
    torch.testing.assert_close(scaled, base, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(permuted, base, rtol=1e-6, atol=1e-8)


@settings(max_examples=40, deadline=None)
@given(points=POINTS, keep=st.integers(0, 1), min_views=st.integers(2, 3))
def test_fewer_than_min_views_is_nan_and_invalid(points: np.ndarray, keep: int, min_views: int) -> None:
    P = ring_projections(4, torch.float64)
    xy = project(torch.as_tensor(points), P)
    weight = torch.zeros(xy.shape[:2], dtype=torch.float64)
    weight[: keep + min_views - 1] = 1.0  # min_views - 1 usable views (or min_views when keep == 1); at most views 0..2
    xy[-1], weight[-1] = float("nan"), 1.0  # a non-finite pixel drops its observation even with a weight
    xyz, valid = triangulate_weighted(xy, weight, P, min_views=min_views)
    if keep:
        assert bool(valid.all()) and bool(torch.isfinite(xyz).all())
    else:
        assert not bool(valid.any()) and bool(torch.isnan(xyz).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_an_empty_batch_returns_no_points(dtype: torch.dtype) -> None:
    xyz, valid = triangulate_weighted(torch.empty((3, 0, 2), dtype=dtype), torch.empty((3, 0), dtype=dtype), ring_projections(3, dtype))
    assert xyz.shape == (0, 3) and valid.shape == (0,) and xyz.dtype == dtype
