"""Behavior contracts for complementary GausSurf geometry supervision."""

from __future__ import annotations

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from gauss_surf.gaussurf_losses import (
    _masked_mean,
    compute_gaussurf_losses,
    linear_ramp_weight,
    normal_contradiction_gate,
    pgsr_edge_weights,
    prepare_gaussurf_targets,
)


@settings(max_examples=48, deadline=None)
@given(
    values=st.lists(st.integers(-1_000, 1_000), min_size=8, max_size=8),
    mask=st.lists(st.booleans(), min_size=8, max_size=8),
    perturbations=st.lists(
        st.integers(-1_000_000, 1_000_000), min_size=8, max_size=8
    ),
)
def test_masked_mean_is_bit_identical_after_outside_mask_perturbations(
    values: list[int], mask: list[bool], perturbations: list[int]
) -> None:
    """Pixels outside a loss mask have no numerical effect on that loss."""
    assume(any(mask))
    values_n: torch.Tensor = torch.tensor(
        values, dtype=torch.float64, device="cpu"
    )
    mask_n: torch.Tensor = torch.tensor(mask, dtype=torch.bool, device="cpu")
    perturbations_n: torch.Tensor = torch.tensor(
        perturbations, dtype=torch.float64, device="cpu"
    )
    perturbed_n: torch.Tensor = torch.where(
        mask_n, values_n, perturbations_n
    )

    original_loss: torch.Tensor = _masked_mean(values_n, mask_n)
    perturbed_loss: torch.Tensor = _masked_mean(perturbed_n, mask_n)

    assert torch.equal(perturbed_loss, original_loss)


def test_masked_out_nonfinite_values_do_not_poison_loss_or_gradients() -> None:
    """A non-finite value outside the mask contributes exact zero."""
    values_n: torch.Tensor = torch.tensor(
        [3.0, float("inf")],
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )
    mask_n: torch.Tensor = torch.tensor(
        [True, False], dtype=torch.bool, device="cpu"
    )

    loss: torch.Tensor = _masked_mean(values_n, mask_n)
    loss.backward()

    assert torch.equal(
        loss, torch.tensor(3.0, dtype=torch.float64, device="cpu")
    )
    assert values_n.grad is not None
    assert torch.isfinite(values_n.grad).all()


def test_linear_ramp_weight_delays_and_saturates() -> None:
    assert linear_ramp_weight(step=9, start_step=10, ramp_steps=4) == 0.0
    assert linear_ramp_weight(step=10, start_step=10, ramp_steps=4) == 0.25
    assert linear_ramp_weight(step=11, start_step=10, ramp_steps=4) == 0.5
    assert linear_ramp_weight(step=13, start_step=10, ramp_steps=4) == 1.0
    assert linear_ramp_weight(step=20, start_step=10, ramp_steps=4) == 1.0


def test_target_resize_keeps_masks_discrete_and_normals_unit_length() -> None:
    targets = prepare_gaussurf_targets(
        prior_normal_3hw=torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ),
        prior_valid_1hw=torch.tensor([[[True, False], [False, True]]]),
        height=4,
        width=4,
    )

    assert targets.prior_normal_hw3.shape == (4, 4, 3)
    assert targets.prior_valid_hw1.dtype == torch.bool
    assert set(targets.prior_valid_hw1.unique().tolist()) == {False, True}
    torch.testing.assert_close(
        torch.linalg.vector_norm(targets.prior_normal_hw3, dim=-1),
        torch.ones((4, 4)),
    )


def test_neuris_contradiction_gate_rejects_normals_beyond_thirty_degrees() -> None:
    rendered = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    prior = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    gate = normal_contradiction_gate(rendered, prior, valid)

    torch.testing.assert_close(gate, torch.tensor([[[True], [False]]]))
    assert not gate.requires_grad


def test_normal_prior_uses_all_valid_surface_pixels() -> None:
    direct_normal = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    depth_normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    prior_normal = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    prior_valid = torch.ones((1, 2, 1), dtype=torch.bool)
    surface_valid = torch.ones_like(prior_valid)

    result = compute_gaussurf_losses(
        direct_normal_hw3=direct_normal,
        depth_normal_hw3=depth_normal,
        prior_normal_hw3=prior_normal,
        prior_valid_hw1=prior_valid,
        surface_valid_hw1=surface_valid,
    )

    torch.testing.assert_close(result.normal_prior_cosine, torch.tensor(1.0))
    torch.testing.assert_close(result.depth_normal_cosine, torch.tensor(0.5))
    torch.testing.assert_close(result.normal_prior_mask_fraction, torch.tensor(1.0))


def test_neuris_gate_can_veto_prior_without_changing_depth_mask() -> None:
    normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    prior_valid = torch.ones((1, 2, 1), dtype=torch.bool)
    contradiction_gate = torch.tensor([[[True], [False]]])

    result = compute_gaussurf_losses(
        direct_normal_hw3=normals,
        depth_normal_hw3=normals,
        prior_normal_hw3=normals,
        prior_valid_hw1=prior_valid,
        surface_valid_hw1=torch.ones_like(prior_valid),
        normal_prior_gate_hw1=contradiction_gate,
    )

    torch.testing.assert_close(result.normal_prior_mask_fraction, torch.tensor(0.5))
    torch.testing.assert_close(result.normal_prior_cosine, torch.tensor(0.0))


def test_empty_masks_produce_finite_differentiable_zero_losses() -> None:
    normal = torch.tensor(
        [[[0.0, 0.0, 1.0]]], device="cpu", requires_grad=True
    )
    false_mask = torch.zeros((1, 1, 1), dtype=torch.bool, device="cpu")

    result = compute_gaussurf_losses(
        direct_normal_hw3=normal,
        depth_normal_hw3=normal,
        prior_normal_hw3=normal.detach(),
        prior_valid_hw1=false_mask,
        surface_valid_hw1=false_mask,
    )
    loss_terms: tuple[torch.Tensor, ...] = (
        result.normal_prior_cosine,
        result.depth_normal_cosine,
    )
    for loss_term in loss_terms:
        assert torch.equal(loss_term, torch.zeros_like(loss_term))
        assert torch.isfinite(loss_term)
    total: torch.Tensor = torch.stack(loss_terms).sum()
    total.backward()

    assert torch.equal(total, torch.tensor(0.0, device="cpu"))
    assert normal.grad is not None and torch.isfinite(normal.grad).all()


def test_pgsr_edge_weights_keep_smooth_interiors_and_suppress_borders() -> None:
    rgb = torch.full((5, 6, 3), 0.5, requires_grad=True)

    weights = pgsr_edge_weights(rgb)

    assert weights.shape == (5, 6, 1)
    torch.testing.assert_close(weights[1:-1, 1:-1], torch.ones((3, 4, 1)))
    assert weights[0].eq(0).all() and weights[-1].eq(0).all()
    assert weights[:, 0].eq(0).all() and weights[:, -1].eq(0).all()
    assert not weights.requires_grad


def test_pgsr_edge_weights_suppress_a_vertical_rgb_step() -> None:
    rgb = torch.zeros((5, 7, 3))
    rgb[:, 3:] = 1.0

    weights = pgsr_edge_weights(rgb)

    assert weights[2, 2, 0].item() == 0.0
    assert weights[2, 3, 0].item() == 0.0
    assert weights[2, 1, 0].item() == 1.0
    assert weights[2, 5, 0].item() == 1.0

    floored = pgsr_edge_weights(rgb, minimum_weight=0.25)
    assert floored[2, 2, 0].item() == 0.25
    assert floored[2, 3, 0].item() == 0.25
    assert floored[0].eq(0).all()


def test_consistency_weights_normalize_by_active_weight() -> None:
    direct = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], requires_grad=True
    )
    depth_normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    false_mask = torch.zeros((1, 2, 1), dtype=torch.bool)

    losses = compute_gaussurf_losses(
        direct_normal_hw3=direct,
        depth_normal_hw3=depth_normal,
        prior_normal_hw3=depth_normal,
        prior_valid_hw1=false_mask,
        surface_valid_hw1=torch.ones_like(false_mask),
        consistency_weight_hw1=torch.tensor([[[0.0], [0.25]]]),
    )

    torch.testing.assert_close(losses.depth_normal_cosine, torch.tensor(0.0))
    torch.testing.assert_close(losses.consistency_weight_mean, torch.tensor(0.125))
    torch.testing.assert_close(
        losses.consistency_weight_below_half_fraction, torch.tensor(1.0)
    )
    losses.depth_normal_cosine.backward()
    assert direct.grad is not None and torch.isfinite(direct.grad).all()


def test_consistency_weights_can_match_pgsr_full_image_mean() -> None:
    direct = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], requires_grad=True
    )
    depth_normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
    false_mask = torch.zeros((1, 2, 1), dtype=torch.bool)

    losses = compute_gaussurf_losses(
        direct_normal_hw3=direct,
        depth_normal_hw3=depth_normal,
        prior_normal_hw3=depth_normal,
        prior_valid_hw1=false_mask,
        surface_valid_hw1=torch.ones_like(false_mask),
        consistency_weight_hw1=torch.full((1, 2, 1), 0.25),
        normalize_consistency_weights=False,
    )

    torch.testing.assert_close(losses.depth_normal_cosine, torch.tensor(0.125))
