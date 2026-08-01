"""Unit tests for gaussian init, plus a live gt-mesh fetch gate."""

import numpy as np
import pytest
from conftest import PROBE_CATALOG_URL, catalog_reachable
from jaxtyping import Float32, UInt32
from numpy import ndarray

from rerun_gsplat.apis.mesh_init import ColoredPoints, GaussianInit, gaussians_from_points, load_gt_mesh, unpack_rgba32
from rerun_gsplat.apis.segment_views import SegmentViewsConfig


def test_unpack_rgba32() -> None:
    packed: UInt32[ndarray, "2"] = np.array([0xBFBFBEFF, 0xFF0000FF], dtype=np.uint32)
    rgb: Float32[ndarray, "2 3"] = unpack_rgba32(packed)
    assert np.allclose(rgb[0], [0xBF / 255.0, 0xBF / 255.0, 0xBE / 255.0])
    assert np.allclose(rgb[1], [1.0, 0.0, 0.0])


def test_gaussians_from_points_subsamples_and_scales() -> None:
    rng: np.random.Generator = np.random.default_rng(seed=1)
    verts: Float32[ndarray, "n 3"] = rng.uniform(-1.0, 1.0, size=(5000, 3)).astype(np.float32)
    colors: Float32[ndarray, "n 3"] = rng.uniform(0.0, 1.0, size=(5000, 3)).astype(np.float32)
    init: GaussianInit = gaussians_from_points(ColoredPoints(verts_n3=verts, rgbs_n3=colors), max_points=1000)
    assert init.means_n3.shape == (1000, 3)
    assert init.quats_n4.shape == (1000, 4)
    assert bool((init.quats_n4[:, 0] == 1.0).all())
    scales: Float32[ndarray, "n 3"] = init.log_scales_n3.exp().numpy()
    assert scales.min() > 0.0 and scales.max() < 1.0
    assert np.allclose(init.logit_opacities_n.sigmoid().numpy(), 0.5)


@pytest.mark.skipif(not catalog_reachable(), reason="ephemeral catalog server on :51299 not running")
def test_load_gt_mesh_live() -> None:
    config: SegmentViewsConfig = SegmentViewsConfig(catalog_url=PROBE_CATALOG_URL)
    points: ColoredPoints = load_gt_mesh(config)
    verts: Float32[ndarray, "n 3"] = points.verts_n3
    colors: Float32[ndarray, "n 3"] = points.rgbs_n3
    assert len(verts) > 100_000
    assert verts.shape == colors.shape
    extent: Float32[ndarray, "3"] = verts.max(axis=0) - verts.min(axis=0)
    assert (extent > 1.0).all() and (extent < 30.0).all()
    assert colors.min() >= 0.0 and colors.max() <= 1.0
