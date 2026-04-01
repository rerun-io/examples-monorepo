"""Tests for the Gaussians3D dataclass and component batch generation."""

from __future__ import annotations

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from jaxtyping import Float32

from gsplat_rust_renderer.gaussians3d import Gaussians3D


def _make_gaussians(n: int, *, with_sh: bool = False) -> Gaussians3D:
    """Helper to create a valid Gaussians3D instance with *n* splats."""
    rng: np.random.Generator = np.random.default_rng(42)
    sh: Float32[np.ndarray, "n coeffs 3"] | None = None
    if with_sh:
        sh = rng.standard_normal((n, 4, 3)).astype(np.float32)
    return Gaussians3D(
        centers=rng.standard_normal((n, 3)).astype(np.float32),
        quaternions_xyzw=rng.standard_normal((n, 4)).astype(np.float32),
        scales=rng.standard_normal((n, 3)).astype(np.float32),
        opacities=rng.random(n).astype(np.float32),
        colors_dc=rng.random((n, 3)).astype(np.float32),
        sh_coefficients=sh,
    )


def test_construction_validates_and_normalizes() -> None:
    """Verify Gaussians3D validates shapes and normalizes quaternions."""
    n: int = 10
    gaussians: Gaussians3D = _make_gaussians(n)
    assert gaussians.centers.shape == (n, 3)
    assert gaussians.quaternions_xyzw.shape == (n, 4)
    # Quaternions should be unit-length after normalization.
    norms: Float32[np.ndarray, "n"] = np.linalg.norm(gaussians.quaternions_xyzw, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    # Scales must be >= 1e-6.
    assert np.all(gaussians.scales >= 1e-6)
    # Opacities must be in [0, 1].
    assert np.all(gaussians.opacities >= 0.0)
    assert np.all(gaussians.opacities <= 1.0)


def test_component_batches_without_sh() -> None:
    """Verify as_component_batches returns 5 batches when SH is absent."""
    gaussians: Gaussians3D = _make_gaussians(5)
    batches: list = gaussians.as_component_batches()
    assert len(batches) == 5


def test_component_batches_with_sh() -> None:
    """Verify as_component_batches returns 6 batches when SH is present."""
    gaussians: Gaussians3D = _make_gaussians(5, with_sh=True)
    batches: list = gaussians.as_component_batches()
    assert len(batches) == 6


def test_shape_mismatch_raises() -> None:
    """Verify mismatched array lengths raise ValueError."""
    with pytest.raises(ValueError):
        Gaussians3D(
            centers=np.zeros((10, 3), dtype=np.float32),
            quaternions_xyzw=np.zeros((5, 4), dtype=np.float32),
            scales=np.ones((10, 3), dtype=np.float32),
            opacities=np.ones(10, dtype=np.float32),
            colors_dc=np.ones((10, 3), dtype=np.float32),
        )


def test_bad_shape_raises() -> None:
    """Verify wrong array shapes raise ValueError or beartype violation.

    In dev mode beartype catches the shape mismatch at the ``__init__``
    boundary before ``__post_init__`` runs, raising
    ``BeartypeCallHintParamViolation`` instead of ``ValueError``.
    """
    with pytest.raises((ValueError, BeartypeCallHintParamViolation)):
        Gaussians3D(
            centers=np.zeros((10, 2), dtype=np.float32),  # wrong: 2 not 3
            quaternions_xyzw=np.zeros((10, 4), dtype=np.float32),
            scales=np.ones((10, 3), dtype=np.float32),
            opacities=np.ones(10, dtype=np.float32),
            colors_dc=np.ones((10, 3), dtype=np.float32),
        )
