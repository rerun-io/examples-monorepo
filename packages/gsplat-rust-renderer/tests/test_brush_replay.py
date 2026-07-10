"""Behavior tests for pure Brush PLY snapshot replay."""

import pytest
import rerun as rr
from simplecv.rerun_log_utils import RerunTyroConfig

from gsplat_rust_renderer.apis.visualize_brush_training import (
    VisualizeBrushTrainingConfig,
    estimate_replay_size_bytes,
    set_iteration_time,
    should_retain_snapshot,
)


def test_retention_keeps_first_every_thousand_and_final_snapshot() -> None:
    """50-step exports become a bounded 31-snapshot 30k replay."""
    export_iterations: tuple[int, ...] = tuple(range(50, 30001, 50))
    retained_iterations: tuple[int, ...] = tuple(
        iteration
        for index, iteration in enumerate(export_iterations)
        if should_retain_snapshot(
            iteration=iteration,
            is_first=index == 0,
            is_final=iteration == 30000,
            export_every=50,
            snapshot_stride=20,
        )
    )

    assert retained_iterations == (50, *range(1000, 30001, 1000))
    assert len(retained_iterations) == 31


def test_retention_budget_stays_below_two_gb_at_one_million_splats() -> None:
    """The documented worst-case replay schedule remains under the RRD cap."""
    estimated_bytes: int = estimate_replay_size_bytes(max_splats=1_000_000, snapshot_count=31, final_has_sh=True)

    assert estimated_bytes <= 2_000_000_000


def test_replay_logs_brush_iterations_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retained snapshots use Brush's plural iterations timeline."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-replay-test", headless=True)
    )
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(rr, "set_time", lambda timeline, *, sequence: calls.append((timeline, sequence)))

    set_iteration_time(config, 1000)

    assert calls == [("iterations", 1000), ("step", 20)]


def test_config_defaults_to_pure_spinning_replay() -> None:
    """Default visualization never depends on Brush's embedded Rerun SDK."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-replay-defaults", headless=True)
    )

    assert config.replay_only is True
    assert config.brush_native is False
    assert config.spin_speed > 0.0
    assert config.step_stride == 50
    assert config.snapshot_stride == 20
