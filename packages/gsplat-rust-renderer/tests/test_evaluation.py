"""Behavior tests for nerfbaselines scene evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsplat_rust_renderer.apis.evaluate_nerfbaselines import Config, quality_guard_failures, selected_scenes
from gsplat_rust_renderer.evaluation import (
    evaluate_prediction_directory,
    evaluate_predictions_against_checkpoint,
    render_test_split,
)
from gsplat_rust_renderer.nerfbaselines import BLENDER_SCENES, scene_pretrained_dir


def test_evaluation_delegates_published_metrics_to_rust(tmp_path: Path) -> None:
    """The Python boundary preserves the Rust result and metric convention."""
    binary: Path = tmp_path / "evaluator"
    capture: Path = tmp_path / "arguments"
    payload: str = json.dumps({
        "views": [{"name": "test/r_0.png", "psnr": 12.5, "ssim": 0.75, "lpips": None}],
        "mean": {"psnr": 12.5, "ssim": 0.75, "lpips": None},
        "convention": "published",
        "provenance": {"crate_version": "test", "source_sha": "test", "cargo_lock_sha256": "test", "brush": "test"},
    })
    binary.write_text(
        "#!/usr/bin/env python3\nimport pathlib, sys\n"
        f"pathlib.Path({str(capture)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        f"pathlib.Path(sys.argv[sys.argv.index('--out') + 1]).write_text({payload!r})\n"
    )
    binary.chmod(0o755)
    result = evaluate_prediction_directory(tmp_path / "render", tmp_path / "gt", eval_binary=binary)
    assert (len(result.views), result.mean.psnr, result.mean.ssim) == (1, 12.5, 0.75)
    arguments: list[str] = capture.read_text().splitlines()
    assert arguments[:7] == ["eval", "--render", str(tmp_path / "render"), "--gt", str(tmp_path / "gt"), "--convention", "published"]


def test_render_test_split_invokes_standalone_all_frame_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Python orchestration invokes the render command once for the full split."""
    capture_path: Path = tmp_path / "args.txt"
    binary_path: Path = tmp_path / "fake-gsplat-render"
    binary_path.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n")
    binary_path.chmod(0o755)
    monkeypatch.setenv("CAPTURE_PATH", str(capture_path))
    ply_path: Path = tmp_path / "point_cloud.ply"
    camera_path: Path = tmp_path / "transforms_test.json"
    output_dir: Path = tmp_path / "renders"

    render_test_split(
        render_binary=binary_path, ply_path=ply_path, camera_path=camera_path,
        output_dir=output_dir, width=800, height=600,
    )
    assert capture_path.read_text().splitlines() == [
        "render", "--ply", str(ply_path), "--camera", str(camera_path), "--output-dir", str(output_dir),
        "--width", "800", "--height", "600", "--background", "1,1,1",
    ]



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


@pytest.mark.golden
@pytest.mark.parametrize("scene", ["lego", "hotdog"])
def test_checkpoint_predictions_match_published_full_split(scene: str) -> None:
    """Bundled checkpoint renders reproduce published metrics over all 200 views."""
    checkpoint_dir: Path = scene_pretrained_dir(scene)
    if not (checkpoint_dir / "results.json").exists():
        pytest.skip(f"{scene} nerfbaselines checkpoint is not downloaded")

    result = evaluate_predictions_against_checkpoint(checkpoint_dir / "predictions/color", checkpoint_dir)

    assert result.image_count == 200
    np.testing.assert_allclose(result.measured_psnr, result.published_psnr, atol=5e-6)
    np.testing.assert_allclose(result.measured_ssim, result.published_ssim, atol=5e-6)
