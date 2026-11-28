"""Open4dgsTrainer.step_config: the MoGe-2 prior's ``renders_depth`` decides RGB+ED per train step; renders outside the
train loop keep the depth."""

import dataclasses

import pytest

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")


def test_step_config_drops_depth_only_on_rgb_train_steps() -> None:
    from trigs.core.trainer import TrigsTrainingConfig

    from open4dgs.trigs_trainer import Open4dgsTrainer

    trainer = Open4dgsTrainer(device="cpu")
    config = dataclasses.replace(TrigsTrainingConfig(), render_depth=True)
    assert trainer.step_config(config, 3) is config  # no decider: every step renders depth
    trainer.depth_at_step = lambda step: step % 3 == 0
    assert trainer.step_config(config, 3) is config
    rgb = trainer.step_config(config, 4)
    assert not rgb.render_depth and dataclasses.replace(rgb, render_depth=True) == config
    assert trainer.step_config(config, 5) is rgb  # built once
    assert trainer.step_config(config, None) is config  # eval / preview renders
    plain = TrigsTrainingConfig()
    assert trainer.step_config(plain, 4) is plain
