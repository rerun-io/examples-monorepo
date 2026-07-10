"""Behavior tests for nerfbaselines scene evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import UInt8
from PIL import Image

from gsplat_rust_renderer.apis.evaluate_nerfbaselines import Config, quality_guard_failures, selected_scenes
from gsplat_rust_renderer.evaluation import (
    evaluate_checkpoint_predictions,
    evaluate_prediction_directory,
    evaluate_predictions_against_checkpoint,
    render_test_split,
)
from gsplat_rust_renderer.metrics import load_image_rgb, psnr, ssim
from gsplat_rust_renderer.nerfbaselines import BLENDER_SCENES, scene_pretrained_dir


def _save_rgb(path: Path, rgb_hwc: UInt8[np.ndarray, "h w 3"]) -> None:
    """Save one RGB fixture image, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_hwc, mode="RGB").save(path)


def test_evaluate_prediction_directory_averages_per_image_metrics(tmp_path: Path) -> None:
    """Evaluation pairs relative paths and averages metrics over every image."""
    rendered_dir: Path = tmp_path / "rendered"
    ground_truth_dir: Path = tmp_path / "ground-truth"
    black_hwc: UInt8[np.ndarray, "h w 3"] = np.zeros((16, 16, 3), dtype=np.uint8)
    gray_hwc: UInt8[np.ndarray, "h w 3"] = np.full((16, 16, 3), 26, dtype=np.uint8)

    _save_rgb(rendered_dir / "test" / "r_0.png", black_hwc)
    _save_rgb(ground_truth_dir / "test" / "r_0.png", black_hwc)
    _save_rgb(rendered_dir / "test" / "r_1.png", black_hwc)
    _save_rgb(ground_truth_dir / "test" / "r_1.png", gray_hwc)

    result = evaluate_prediction_directory(rendered_dir, ground_truth_dir)
    rendered_0 = load_image_rgb(rendered_dir / "test" / "r_0.png")
    ground_truth_0 = load_image_rgb(ground_truth_dir / "test" / "r_0.png")
    rendered_1 = load_image_rgb(rendered_dir / "test" / "r_1.png")
    ground_truth_1 = load_image_rgb(ground_truth_dir / "test" / "r_1.png")

    assert result.image_count == 2
    np.testing.assert_allclose(result.psnr, (psnr(rendered_0, ground_truth_0) + psnr(rendered_1, ground_truth_1)) / 2.0)
    np.testing.assert_allclose(result.ssim, (ssim(rendered_0, ground_truth_0) + ssim(rendered_1, ground_truth_1)) / 2.0)


def test_evaluate_prediction_directory_rejects_incomplete_splits(tmp_path: Path) -> None:
    """An extra or missing frame fails instead of silently changing the aggregate."""
    rendered_dir: Path = tmp_path / "rendered"
    ground_truth_dir: Path = tmp_path / "ground-truth"
    black_hwc: UInt8[np.ndarray, "h w 3"] = np.zeros((16, 16, 3), dtype=np.uint8)
    _save_rgb(rendered_dir / "test" / "r_0.png", black_hwc)
    _save_rgb(ground_truth_dir / "test" / "r_0.png", black_hwc)
    _save_rgb(ground_truth_dir / "test" / "r_1.png", black_hwc)

    with pytest.raises(ValueError, match="image sets differ"):
        evaluate_prediction_directory(rendered_dir, ground_truth_dir)


def test_evaluate_checkpoint_predictions_compares_published_results(tmp_path: Path) -> None:
    """A checkpoint report includes published values and signed deltas."""
    checkpoint_dir: Path = tmp_path / "checkpoint"
    rendered_dir: Path = checkpoint_dir / "predictions" / "color"
    ground_truth_dir: Path = checkpoint_dir / "predictions" / "gt-color"
    black_hwc: UInt8[np.ndarray, "h w 3"] = np.zeros((16, 16, 3), dtype=np.uint8)
    gray_hwc: UInt8[np.ndarray, "h w 3"] = np.full((16, 16, 3), 26, dtype=np.uint8)
    _save_rgb(rendered_dir / "test" / "r_0.png", black_hwc)
    _save_rgb(ground_truth_dir / "test" / "r_0.png", gray_hwc)
    (checkpoint_dir / "results.json").write_text(json.dumps({"render_dataset_metadata": {"scene": "fixture"}, "metrics": {"psnr": 20.0, "ssim": 0.5}}))

    result = evaluate_checkpoint_predictions(checkpoint_dir)

    assert result.scene == "fixture"
    assert result.image_count == 1
    assert result.published_psnr == 20.0
    assert result.published_ssim == 0.5
    np.testing.assert_allclose(result.psnr_delta, result.measured_psnr - result.published_psnr)
    np.testing.assert_allclose(result.ssim_delta, result.measured_ssim - result.published_ssim)


def test_render_test_split_invokes_standalone_all_frame_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Python orchestration calls the no-Rerun binary once for the full split."""
    capture_path: Path = tmp_path / "args.txt"
    binary_path: Path = tmp_path / "fake-gsplat-render"
    binary_path.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n")
    binary_path.chmod(0o755)
    monkeypatch.setenv("CAPTURE_PATH", str(capture_path))
    ply_path: Path = tmp_path / "point_cloud.ply"
    camera_path: Path = tmp_path / "transforms_test.json"
    output_dir: Path = tmp_path / "renders"

    render_test_split(
        render_binary=binary_path,
        ply_path=ply_path,
        camera_path=camera_path,
        output_dir=output_dir,
        width=800,
        height=600,
    )

    assert capture_path.read_text().splitlines() == [
        "--ply",
        str(ply_path),
        "--camera",
        str(camera_path),
        "--output-dir",
        str(output_dir),
        "--width",
        "800",
        "--height",
        "600",
        "--background",
        "1,1,1",
    ]


def test_evaluate_external_renders_against_checkpoint_ground_truth(tmp_path: Path) -> None:
    """Standalone renders use checkpoint GT/results without bundled predictions."""
    checkpoint_dir: Path = tmp_path / "checkpoint"
    rendered_dir: Path = tmp_path / "standalone-renders"
    ground_truth_dir: Path = checkpoint_dir / "predictions" / "gt-color"
    black_hwc: UInt8[np.ndarray, "h w 3"] = np.zeros((16, 16, 3), dtype=np.uint8)
    _save_rgb(rendered_dir / "test" / "r_0.png", black_hwc)
    _save_rgb(ground_truth_dir / "test" / "r_0.png", black_hwc)
    (checkpoint_dir / "results.json").write_text(json.dumps({"render_dataset_metadata": {"scene": "fixture"}, "metrics": {"psnr": 100.0, "ssim": 1.0}}))

    result = evaluate_predictions_against_checkpoint(rendered_dir, checkpoint_dir)

    assert result.scene == "fixture"
    assert result.image_count == 1
    assert result.measured_psnr == 100.0
    np.testing.assert_allclose(result.measured_ssim, 1.0)


def test_evaluation_cli_selects_one_or_all_blender_scenes() -> None:
    """The harness defaults to all eight scenes and supports a focused scene."""
    assert selected_scenes(Config()) == BLENDER_SCENES
    assert selected_scenes(Config(scene="mic")) == ("mic",)


def test_quality_guard_accepts_cross_backend_deltas_within_thresholds() -> None:
    """Expected Metal-vs-published drift passes the configured quality guard."""
    report = pytest.importorskip("gsplat_rust_renderer.evaluation").CheckpointEvaluation(
        scene="lego", image_count=200, measured_psnr=32.05, published_psnr=32.0, psnr_delta=0.05,
        measured_ssim=0.95002, published_ssim=0.95, ssim_delta=0.00002,
    )

    assert quality_guard_failures([report], Config()) == []


def test_quality_guard_reports_every_threshold_breach() -> None:
    """A scene exceeding either absolute delta makes the quality guard fail."""
    report = pytest.importorskip("gsplat_rust_renderer.evaluation").CheckpointEvaluation(
        scene="lego", image_count=200, measured_psnr=31.7, published_psnr=32.0, psnr_delta=-0.3,
        measured_ssim=0.951, published_ssim=0.95, ssim_delta=0.001,
    )

    assert quality_guard_failures([report], Config()) == [
        "lego: PSNR delta -0.30000000 exceeds 0.15000000",
        "lego: SSIM delta +0.001000000 exceeds 0.000500000",
    ]


def test_lego_checkpoint_predictions_match_published_full_split() -> None:
    """Bundled checkpoint renders reproduce published metrics over all 200 views."""
    checkpoint_dir: Path = scene_pretrained_dir("lego")
    if not (checkpoint_dir / "results.json").exists():
        pytest.skip("lego nerfbaselines checkpoint is not downloaded")

    result = evaluate_checkpoint_predictions(checkpoint_dir)

    assert result.image_count == 200
    np.testing.assert_allclose(result.measured_psnr, result.published_psnr, atol=5e-6)
    np.testing.assert_allclose(result.measured_ssim, result.published_ssim, atol=5e-6)


def test_hotdog_checkpoint_predictions_match_published_full_split() -> None:
    """Hotdog's bundled renders reproduce both published metrics over 200 views."""
    checkpoint_dir: Path = scene_pretrained_dir("hotdog")
    if not (checkpoint_dir / "results.json").exists():
        pytest.skip("hotdog nerfbaselines checkpoint is not downloaded")

    result = evaluate_checkpoint_predictions(checkpoint_dir)

    assert result.image_count == 200
    np.testing.assert_allclose(result.measured_psnr, result.published_psnr, atol=5e-6)
    np.testing.assert_allclose(result.measured_ssim, result.published_ssim, atol=5e-6)
