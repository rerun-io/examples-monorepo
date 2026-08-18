"""Pure schedule helpers shared by gsplat training paths."""


def downscale_factor(step: int) -> int:
    """Return the splatfacto quarter-to-half-to-full resolution factor."""
    if step < 0:
        raise ValueError("step must be non-negative")
    return 2 ** max(2 - step // 3_000, 0)


def sh_degree(step: int) -> int:
    """Return the active spherical-harmonic degree for one training step."""
    if step < 0:
        raise ValueError("step must be non-negative")
    return min(step // 1_000, 3)


def should_refine(
    *,
    step: int,
    num_train_frames: int,
    refine_every: int,
    reset_every: int,
    refine_start: int,
    refine_stop: int,
) -> bool:
    """Return splatfacto's integer refinement predicate.

    The pause after each reset cycle is derived from the number of training
    images across all camera streams plus one refinement interval.
    """
    if step < 0 or num_train_frames <= 0:
        raise ValueError("step must be non-negative and the train split non-empty")
    if refine_every <= 0 or reset_every <= 0:
        raise ValueError("schedule intervals must be positive")
    if refine_start < 0 or refine_stop <= refine_start:
        raise ValueError("refinement bounds are invalid")
    pause_after_reset: int = num_train_frames + refine_every
    return (
        step > refine_start
        and step < refine_stop
        and step % refine_every == 0
        and step % reset_every >= pause_after_reset
    )


def should_reset_opacity(*, step: int, reset_every: int, refine_stop: int) -> bool:
    """Return gsplat 1.6's opacity-reset predicate including its early stop."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if reset_every <= 0 or refine_stop <= 0:
        raise ValueError("reset interval and refinement stop must be positive")
    return step < refine_stop and step > 0 and step % reset_every == 0


def opacity_reset_callback_step(
    *,
    step: int,
    refinement_cycle_every: int,
    opacity_reset_every: int,
) -> int:
    """Return the strategy step that preserves refinement while gating resets.

    Gsplat's default strategy uses one cycle for both its refinement pause and
    opacity reset. The historical gauss-surf recipe needs the 3,000-step pause
    cycle, but its 1.5.3 opacity reset branch never ran. Moving a reset-cycle
    callback to the following non-refinement step preserves that old behavior.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if refinement_cycle_every <= 0 or opacity_reset_every <= 0:
        raise ValueError("schedule intervals must be positive")
    suppress_reset: bool = (
        step > 0
        and step % refinement_cycle_every == 0
        and step % opacity_reset_every != 0
    )
    return step + 1 if suppress_reset else step
