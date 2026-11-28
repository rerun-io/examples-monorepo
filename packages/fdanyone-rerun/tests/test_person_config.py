"""PersonConfig's step budget: the CLI default stops a 25,000-step schedule at 20,000 steps; the app's presets span theirs."""

import pytest

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (fdanyone-rerun envs)")


def test_default_trains_20k_steps_of_the_25k_schedule_and_with_steps_spans_the_new_budget() -> None:
    from fdanyone_rerun.person4d import PersonConfig

    default = PersonConfig()
    assert (default.train.steps, default.train.horizon) == (20_000, 25_000)
    assert default.prior_config.at_steps(default.train.horizon).contradiction_start_step == 18_750
    app = PersonConfig().with_steps(5_000)
    assert (app.train.steps, app.train.horizon) == (5_000, 5_000)
    assert PersonConfig().with_steps(40_000).train.horizon == 40_000
