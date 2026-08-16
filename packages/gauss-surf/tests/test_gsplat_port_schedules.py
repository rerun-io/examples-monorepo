"""Schedule tests shared by the version-control and direct-gsplat stages."""

from gauss_surf.apis.train_gsplat import Config
from gauss_surf.gsplat_schedule import (
    downscale_factor,
    opacity_reset_callback_step,
    sh_degree,
    should_refine,
    should_reset_opacity,
)

EXPECTED_REFINE_STEPS: tuple[int, ...] = (
    1_400,
    1_500,
    1_600,
    1_700,
    1_800,
    1_900,
    2_000,
    2_100,
    2_200,
    2_300,
    2_400,
    2_500,
    2_600,
    2_700,
    2_800,
    2_900,
    4_400,
    4_500,
    4_600,
    4_700,
    4_800,
    4_900,
    5_000,
    5_100,
    5_200,
    5_300,
    5_400,
)


def test_production_trainer_keeps_opacity_reset_gated_off() -> None:
    """Stage 2 remains selectable, but the production port keeps reset off."""
    assert Config(video_id="segment").opacity_reset_every == 1_000_000_000


def test_opacity_reset_gate_preserves_the_refinement_cycle() -> None:
    """Only the fixed gsplat reset branch is bypassed at historical reset steps."""
    callback_steps: tuple[int, ...] = tuple(
        opacity_reset_callback_step(
            step=step,
            refinement_cycle_every=3_000,
            opacity_reset_every=1_000_000_000,
        )
        for step in (2_999, 3_000, 3_001, 6_000)
    )

    assert callback_steps == (2_999, 3_001, 3_001, 6_001)


def test_direct_trainer_refinement_steps_match_the_golden() -> None:
    """The public predicate derives its pause from all 1,221 training images."""
    actual_steps: tuple[int, ...] = tuple(
        step
        for step in range(7_000)
        if should_refine(
            step=step,
            num_train_frames=1_221,
            refine_every=100,
            reset_every=3_000,
            refine_start=500,
            refine_stop=5_500,
        )
    )

    assert actual_steps == EXPECTED_REFINE_STEPS


def test_fixed_opacity_reset_stops_with_refinement() -> None:
    """Match installed gsplat 1.6: step 3000 resets, but step 6000 is past refine stop."""
    actual_steps: tuple[int, ...] = tuple(
        step
        for step in range(7_000)
        if should_reset_opacity(step=step, reset_every=3_000, refine_stop=5_500)
    )

    assert actual_steps == (3_000,)


def test_direct_trainer_resolution_and_sh_boundaries_match_the_goldens() -> None:
    """Progressive resolution and SH degree change at exact iteration boundaries."""
    boundary_steps: tuple[int, ...] = (0, 999, 1_000, 1_999, 2_000, 2_999, 3_000, 5_999, 6_000, 6_999)

    assert tuple(downscale_factor(step) for step in boundary_steps) == (4, 4, 4, 4, 4, 4, 2, 2, 1, 1)
    assert tuple(sh_degree(step) for step in boundary_steps) == (0, 0, 1, 1, 2, 2, 3, 3, 3, 3)
