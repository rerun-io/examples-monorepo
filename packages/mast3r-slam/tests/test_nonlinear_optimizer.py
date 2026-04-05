"""Unit tests for mast3r_slam.nonlinear_optimizer."""

import torch
from jaxtyping import Float

from mast3r_slam.nonlinear_optimizer import check_convergence, huber


def test_huber_small_residuals() -> None:
    """Residuals below threshold should get weight 1."""
    r: Float[torch.Tensor, "3"] = torch.tensor([0.1, 0.5, 1.0])
    w: Float[torch.Tensor, "3"] = huber(r, k=1.345)
    torch.testing.assert_close(w, torch.ones(3))


def test_huber_large_residuals() -> None:
    """Residuals above threshold should get weight k/|r|."""
    k: float = 1.345
    r: Float[torch.Tensor, "2"] = torch.tensor([5.0, 10.0])
    w: Float[torch.Tensor, "2"] = huber(r, k=k)
    expected: Float[torch.Tensor, "2"] = torch.tensor([k / 5.0, k / 10.0])
    torch.testing.assert_close(w, expected)


def test_huber_symmetric() -> None:
    """Huber weights should be symmetric: huber(r) == huber(-r)."""
    r: Float[torch.Tensor, "3"] = torch.tensor([-2.0, 0.0, 2.0])
    w: Float[torch.Tensor, "3"] = huber(r)
    torch.testing.assert_close(w, huber(-r))


def test_check_convergence_converged() -> None:
    """Should report convergence when both criteria are met."""
    delta: Float[torch.Tensor, "3"] = torch.tensor([1e-10, 1e-10, 1e-10])
    result: bool = check_convergence(
        iter=5,
        rel_error_threshold=1e-3,
        delta_norm_threshold=1e-3,
        old_cost=1.0,
        new_cost=0.9999,
        delta=delta,
    )
    assert result is True


def test_check_convergence_not_converged() -> None:
    """Should report not converged when cost change and delta are large."""
    delta: Float[torch.Tensor, "3"] = torch.tensor([1.0, 1.0, 1.0])
    result: bool = check_convergence(
        iter=1,
        rel_error_threshold=1e-6,
        delta_norm_threshold=1e-6,
        old_cost=10.0,
        new_cost=5.0,
        delta=delta,
    )
    assert result is False
