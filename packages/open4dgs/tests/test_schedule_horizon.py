"""TrainConfig.schedule_steps: a run that stops before the end of the schedule it follows (the LR decay, the relocation
stop and the MoGe-2 prior schedule keep the horizon of a longer run)."""

import pytest

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def test_schedule_steps_sets_the_lr_horizon_and_defaults_to_steps() -> None:
    from open4dgs.trigs_adapter import TrainConfig

    stopped = TrainConfig(steps=20_000, schedule_steps=25_000)
    assert stopped.horizon == 25_000
    trigs = stopped.to_trigs(30.0, 3, ())
    assert (trigs.n_steps, trigs.lr_decay_steps) == (20_000, 25_000)
    full = TrainConfig(steps=20_000)
    assert full.horizon == 20_000 and full.to_trigs(30.0, 3, ()).lr_decay_steps == 0  # TRiGS: 0 = decay over n_steps


def test_schedule_steps_shorter_than_steps_is_rejected() -> None:
    from open4dgs.trigs_adapter import TrainConfig

    with pytest.raises(ValueError, match="schedule_steps"):
        TrainConfig(steps=20_000, schedule_steps=10_000)
