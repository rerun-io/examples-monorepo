"""Behavior tests for pure Brush PLY snapshot replay."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig

import gsplat_rust_renderer.apis.visualize_brush_training as brush_replay
from gsplat_rust_renderer.apis.visualize_brush_training import (
    BrushLogParser,
    BrushMetric,
    VisualizeBrushTrainingConfig,
    estimate_replay_size_bytes,
    log_camera_frustum,
    remove_processed_export,
    set_iteration_time,
    should_retain_snapshot,
)


def test_brush_log_parser_extracts_training_and_eval_metrics() -> None:
    """Pure-trainer stdout becomes typed samples on Brush's iteration clock."""
    parser: BrushLogParser = BrushLogParser()

    samples: list[BrushMetric] = parser.feed(
        "[INFO brush_cli] Train iter 5: loss 0.123456\n"
        "[INFO brush_cli] Refine iter 200, 12345 splats.\n"
        "[INFO brush_cli] Eval iter 1000: PSNR 28.75, ssim 0.9125\n",
        final=True,
    )

    assert samples == [
        BrushMetric(iteration=5, name="loss", value=0.123456),
        BrushMetric(iteration=200, name="num_splats", value=12345.0),
        BrushMetric(iteration=1000, name="psnr", value=28.75),
        BrushMetric(iteration=1000, name="ssim", value=0.9125),
    ]


def test_brush_log_parser_waits_for_a_complete_appended_line() -> None:
    """A polling read split through a metric value cannot drop or corrupt it."""
    parser: BrushLogParser = BrushLogParser()

    first_samples: list[BrushMetric] = parser.feed("Train iter 10: loss 0.0")
    second_samples: list[BrushMetric] = parser.feed("625\n")

    assert first_samples == []
    assert second_samples == [BrushMetric(iteration=10, name="loss", value=0.0625)]


def test_camera_frustum_attaches_a_static_jpeg_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every timeless training frustum carries a bounded encoded GT image."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-camera-test", headless=True),
        plane_thumb_px=160,
        image_jpeg_quality=80,
    )
    camera: PinholeParameters = PinholeParameters(
        name="train_000",
        extrinsics=Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RUB",
            fl_x=200.0,
            fl_y=200.0,
            cx=160.0,
            cy=120.0,
            height=240,
            width=320,
        ),
    )
    log_calls: list[tuple[str, object, bool]] = []
    monkeypatch.setattr(brush_replay, "load_rgb_composited", lambda *_args, **_kwargs: np.full((240, 320, 3), 127, dtype=np.uint8))
    monkeypatch.setattr(brush_replay, "log_pinhole", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rr, "log", lambda path, archetype, *, static=False: log_calls.append((path, archetype, static)))

    log_camera_frustum("world/cameras/train_000", camera, Path("unused.png"), config, "RUB")

    assert len(log_calls) == 1
    path, archetype, static = log_calls[0]
    assert path == "world/cameras/train_000/pinhole/image"
    assert isinstance(archetype, rr.EncodedImage)
    assert static is True


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


def test_config_defaults_to_rich_pure_trainer_replay() -> None:
    """Default replay adds cameras and metrics without Brush's embedded Rerun."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-replay-defaults", headless=True)
    )

    assert config.replay_only is False
    assert config.brush_native is False
    assert config.spin_speed > 0.0
    assert config.step_stride == 50
    assert config.snapshot_stride == 20


def test_live_replay_deletes_processed_intermediates_but_keeps_final(tmp_path: Path) -> None:
    """Disk-bounded live replay removes each consumed PLY except the final export."""
    intermediate: Path = tmp_path / "export_50.ply"
    final: Path = tmp_path / "export_30000.ply"
    intermediate.touch()
    final.touch()

    assert remove_processed_export(intermediate, delete_processed=True, is_final=False) is True
    assert remove_processed_export(final, delete_processed=True, is_final=True) is False
    assert not intermediate.exists()
    assert final.exists()
