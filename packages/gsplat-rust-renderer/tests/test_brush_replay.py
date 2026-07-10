"""Behavior tests for pure Brush PLY snapshot replay."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig

import gsplat_rust_renderer.apis.visualize_brush_training as brush_replay
from gsplat_rust_renderer.apis.visualize_brush_training import (
    BrushLogParser,
    BrushMetric,
    VisualizeBrushTrainingConfig,
    estimate_replay_size_bytes,
    log_brush_metrics,
    log_camera_frustum,
    ready_export_plys,
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


def test_brush_metrics_use_brush_entity_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure-trainer scalars populate the same paths as Brush's own recording."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-metric-paths", headless=True)
    )
    paths: list[str] = []
    monkeypatch.setattr(rr, "set_time", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rr, "log", lambda path, _archetype: paths.append(path))

    log_brush_metrics(
        config,
        [
            BrushMetric(iteration=50, name="loss", value=0.1),
            BrushMetric(iteration=500, name="psnr", value=20.0),
            BrushMetric(iteration=500, name="ssim", value=0.8),
            BrushMetric(iteration=100, name="num_splats", value=1000.0),
        ],
        set(),
    )

    assert paths == ["loss/total", "psnr/eval", "ssim/eval", "splats/num_splats"]


def test_camera_frustum_attaches_a_static_jpeg_thumbnail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every timeless training frustum carries a black-composited bounded GT image."""
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
    composite_backgrounds: list[float] = []

    def load_composited(_path: Path, *, background: float) -> UInt8[ndarray, "h w 3"]:
        composite_backgrounds.append(background)
        return np.full((240, 320, 3), 127, dtype=np.uint8)

    monkeypatch.setattr(brush_replay, "load_rgb_composited", load_composited)
    monkeypatch.setattr(brush_replay, "log_pinhole", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rr, "log", lambda path, archetype, *, static=False: log_calls.append((path, archetype, static)))

    log_camera_frustum("world/cameras/train_000", camera, Path("unused.png"), config, "RUB")

    assert len(log_calls) == 1
    path, archetype, static = log_calls[0]
    assert path == "world/cameras/train_000/pinhole/image"
    assert isinstance(archetype, rr.EncodedImage)
    assert static is True
    assert composite_backgrounds == [0.0]


def test_eval_render_is_jpeg_logged_on_brush_view_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Brush-native eval PNG becomes a bounded render beside its GT view."""
    render_path: Path = tmp_path / "r_0.png"
    Image.fromarray(np.full((24, 32, 3), 64, dtype=np.uint8)).save(render_path)
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-eval-render", headless=True),
        image_jpeg_quality=80,
    )
    log_calls: list[tuple[str, object]] = []
    monkeypatch.setattr(rr, "log", lambda path, archetype: log_calls.append((path, archetype)))

    brush_replay.log_eval_render(render_path, view_index=0, config=config)

    assert len(log_calls) == 1
    path, archetype = log_calls[0]
    assert path == "eval/view_0/render"
    assert isinstance(archetype, rr.EncodedImage)


def test_eval_ground_truth_is_static_jpeg_on_brush_view_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The eval GT is timeless and shares Brush's render-vs-GT hierarchy."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-eval-gt", headless=True),
        image_jpeg_quality=80,
    )
    log_calls: list[tuple[str, object, bool]] = []
    monkeypatch.setattr(brush_replay, "load_rgb_composited", lambda *_args, **_kwargs: np.full((24, 32, 3), 128, dtype=np.uint8))
    monkeypatch.setattr(rr, "log", lambda path, archetype, *, static=False: log_calls.append((path, archetype, static)))

    brush_replay.log_eval_ground_truth(Path("r_0.png"), view_index=0, config=config)

    assert len(log_calls) == 1
    path, archetype, static = log_calls[0]
    assert path == "eval/view_0/ground_truth"
    assert isinstance(archetype, rr.EncodedImage)
    assert static is True


def test_eval_directory_waits_for_every_brush_render_to_settle(tmp_path: Path) -> None:
    """Live cleanup waits for Brush to finish the complete eval split."""
    eval_dir: Path = tmp_path / "eval_500"
    eval_dir.mkdir()
    (eval_dir / "r_0.png").write_bytes(b"first")
    signatures: dict[Path, tuple[int, int]] = {}

    assert brush_replay.eval_dir_is_stable(eval_dir, expected_files=2, pending_signatures=signatures) is False
    (eval_dir / "r_1.png").write_bytes(b"second")
    assert brush_replay.eval_dir_is_stable(eval_dir, expected_files=2, pending_signatures=signatures) is False
    assert brush_replay.eval_dir_is_stable(eval_dir, expected_files=2, pending_signatures=signatures) is True

    (eval_dir / "r_1.png").write_bytes(b"second-and-growing")
    assert brush_replay.eval_dir_is_stable(eval_dir, expected_files=2, pending_signatures=signatures) is False


def test_eval_render_lookup_requires_one_exact_nested_filename(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Suffix variants and duplicate exact names remain pending instead of being guessed."""
    eval_dir: Path = tmp_path / "eval_500"
    (eval_dir / "nested").mkdir(parents=True)
    (eval_dir / "r_98.png.tmp.png").touch()
    assert brush_replay.find_eval_render(eval_dir, "r_98") is None
    assert "expected exactly one" in capsys.readouterr().err

    (eval_dir / "r_98.png").touch()
    assert brush_replay.find_eval_render(eval_dir, "r_98") == eval_dir / "r_98.png"
    (eval_dir / "nested" / "r_98.png").touch()
    assert brush_replay.find_eval_render(eval_dir, "r_98") is None
    assert "found 2" in capsys.readouterr().err


def test_ready_exports_drains_a_large_stable_backlog_in_iteration_order(tmp_path: Path) -> None:
    """A completed dense trainer backlog settles once, then drains in one ordered scan."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-export-backlog", headless=True),
        export_dir=tmp_path,
        follow=True,
    )
    for iteration in range(1, 1001):
        (tmp_path / f"export_{iteration}.ply").write_bytes(b"stable")
    pending_sizes: dict[Path, int] = {}

    first_scan: list[tuple[int, Path]] = list(ready_export_plys(config, {}, pending_sizes))
    second_scan: list[tuple[int, Path]] = list(ready_export_plys(config, {}, pending_sizes))

    assert first_scan == []
    assert [iteration for iteration, _ in second_scan] == list(range(1, 1001))


def test_truncated_checkpoint_retries_then_deletes_after_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stable-size but incomplete PLY remains until a later parse succeeds."""
    ply_path: Path = tmp_path / "export_50.ply"
    ply_path.write_bytes(b"truncated")
    attempts: int = 0

    def parse_then_succeed(_path: Path, _with_sh: bool) -> int:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("truncated PLY")
        return 42

    monkeypatch.setattr(brush_replay, "log_checkpoint", parse_then_succeed)
    retries: dict[Path, int] = {}
    assert brush_replay.try_log_checkpoint(ply_path, False, retries) == (None, False)
    assert ply_path.exists()
    assert brush_replay.try_log_checkpoint(ply_path, False, retries) == (42, False)
    remove_processed_export(ply_path, delete_processed=True, is_final=False)
    assert not ply_path.exists()


def test_corrupt_checkpoint_is_skipped_after_retry_cap_and_retained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A permanently malformed PLY cannot abort replay and is never deleted."""
    ply_path: Path = tmp_path / "export_50.ply"
    ply_path.write_bytes(b"corrupt")
    monkeypatch.setattr(brush_replay, "log_checkpoint", lambda *_args: (_ for _ in ()).throw(ValueError("corrupt PLY")))
    retries: dict[Path, int] = {}

    outcomes: list[tuple[int | None, bool]] = [
        brush_replay.try_log_checkpoint(ply_path, False, retries) for _ in range(brush_replay.ARTIFACT_RETRY_LIMIT)
    ]

    assert outcomes[-1] == (None, True)
    assert ply_path.exists()


def test_retention_keeps_first_every_thousand_and_final_snapshot() -> None:
    """The default 50-step exports become a bounded eight-snapshot 7k replay."""
    export_iterations: tuple[int, ...] = tuple(range(50, 7001, 50))
    retained_iterations: tuple[int, ...] = tuple(
        iteration
        for index, iteration in enumerate(export_iterations)
        if should_retain_snapshot(
            iteration=iteration,
            is_first=index == 0,
            is_final=iteration == 7000,
            export_every=50,
            snapshot_stride=20,
        )
    )

    assert retained_iterations == (50, *range(1000, 7001, 1000))
    assert len(retained_iterations) == 8


def test_retention_budget_stays_below_two_gb_at_one_million_splats() -> None:
    """The documented worst-case replay schedule remains under the RRD cap."""
    estimated_bytes: int = estimate_replay_size_bytes(max_splats=1_000_000, snapshot_count=8, final_has_sh=True)

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
    assert config.total_iters == 7000
    assert config.eval_views_logged == 4
    assert config.step_stride == 50
    assert config.snapshot_stride == 20


def test_standalone_blueprint_lays_out_every_eval_pair_and_logged_metric() -> None:
    """The rich replay exposes four comparisons and only populated graphs."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-blueprint-test", headless=True)
    )

    blueprint = brush_replay.standalone_blueprint(config, train_camera_paths=["world/cameras/train_000"])

    pending: list[object] = [blueprint.root_container]
    view_origins: set[str] = set()
    names: set[str] = set()
    while pending:
        item: object = pending.pop()
        attributes: dict[str, object] = vars(item)
        origin: object | None = attributes.get("origin")
        name: object | None = attributes.get("name")
        if isinstance(origin, str):
            view_origins.add(origin)
        if isinstance(name, str):
            names.add(name)
        contents: object = attributes.get("contents", ())
        if isinstance(contents, tuple):
            pending.extend(contents)

    assert view_origins >= {
        *(f"eval/view_{index}/ground_truth" for index in range(4)),
        *(f"eval/view_{index}/render" for index in range(4)),
        "loss",
        "psnr",
        "ssim",
        "splats",
    }
    assert names >= {"Scene", "Eval views", "Quality", "Loss", "PSNR", "SSIM", "Splats"}


def test_video_blueprint_is_flat_dark_and_shows_all_four_eval_pairs() -> None:
    """Video recordings bake a chrome-free flat layout showing all four logged views."""
    config: VisualizeBrushTrainingConfig = VisualizeBrushTrainingConfig(
        rr_config=RerunTyroConfig(application_id="brush-video-blueprint-test", headless=True),
        video_layout=True,
    )

    blueprint = brush_replay.standalone_blueprint(config, train_camera_paths=["world/cameras/train_000"])

    pending: list[object] = [blueprint.root_container]
    view_origins: set[str] = set()
    container_types: set[str] = set()
    background_kinds: list[int] = []
    while pending:
        item: object = pending.pop()
        container_types.add(type(item).__name__)
        attributes: dict[str, object] = vars(item)
        origin: object | None = attributes.get("origin")
        if isinstance(origin, str):
            view_origins.add(origin)
        properties: object = attributes.get("properties")
        if isinstance(properties, dict) and "Background" in properties:
            background_kinds.extend(properties["Background"].kind.as_arrow_array().to_pylist())
        contents: object = attributes.get("contents", ())
        if isinstance(contents, tuple):
            pending.extend(contents)

    eval_origins: set[str] = {origin for origin in view_origins if origin.startswith("eval/view_")}
    assert config.eval_views_logged == 4
    assert eval_origins == {
        "eval/view_0/ground_truth",
        "eval/view_0/render",
        "eval/view_1/ground_truth",
        "eval/view_1/render",
        "eval/view_2/ground_truth",
        "eval/view_2/render",
        "eval/view_3/ground_truth",
        "eval/view_3/render",
    }
    assert view_origins >= {"loss", "psnr", "ssim", "splats"}
    assert "Tabs" not in container_types
    assert background_kinds == [1]


def test_live_replay_deletes_processed_intermediates_but_keeps_final(tmp_path: Path) -> None:
    """Disk-bounded live replay removes each consumed PLY except the final export."""
    intermediate: Path = tmp_path / "export_50.ply"
    final: Path = tmp_path / "export_7000.ply"
    intermediate.touch()
    final.touch()

    assert remove_processed_export(intermediate, delete_processed=True, is_final=False) is True
    assert remove_processed_export(final, delete_processed=True, is_final=True) is False
    assert not intermediate.exists()
    assert final.exists()
