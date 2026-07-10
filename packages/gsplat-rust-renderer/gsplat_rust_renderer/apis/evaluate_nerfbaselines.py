"""Render and evaluate full nerfbaselines Blender test splits."""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from gsplat_rust_renderer.evaluation import (
    CheckpointEvaluation,
    evaluate_checkpoint_predictions,
    evaluate_predictions_against_checkpoint,
    render_test_split,
)
from gsplat_rust_renderer.nerfbaselines import BLENDER_SCENES, scene_data_dir, scene_ply_path, scene_pretrained_dir

SceneChoice: TypeAlias = Literal["all", "lego", "hotdog", "chair", "drums", "ficus", "materials", "mic", "ship"]


@dataclass
class Config:
    """Command-line configuration for full-split evaluation."""

    scene: SceneChoice = "all"
    """One Blender scene to evaluate, or ``all`` for the full benchmark."""
    checkpoint_only: bool = False
    """Validate bundled checkpoint predictions without running the renderer."""
    render_binary: Path = Path("target/release/gsplat-render")
    """Standalone renderer executable built without Rerun."""
    output_root: Path = Path("data/evaluation/standalone")
    """Root directory for standalone per-scene renders."""
    report: Path = Path("data/evaluation/metrics.json")
    """JSON report written after every selected scene succeeds."""
    width: int = 800
    """Render width matching the Blender benchmark images."""
    height: int = 800
    """Render height matching the Blender benchmark images."""


def selected_scenes(config: Config) -> tuple[str, ...]:
    """Resolve a one-scene or all-scene CLI choice.

    Args:
        config: Evaluation command-line configuration.

    Returns:
        Ordered scene names to evaluate.
    """
    if config.scene == "all":
        return BLENDER_SCENES
    return (config.scene,)


def main(config: Config) -> None:
    """Run checkpoint validation or standalone rendering plus evaluation.

    Args:
        config: Evaluation command-line configuration.
    """
    reports: list[CheckpointEvaluation] = []
    scenes: tuple[str, ...] = selected_scenes(config)
    for scene in scenes:
        checkpoint_dir: Path = scene_pretrained_dir(scene)
        if config.checkpoint_only:
            report: CheckpointEvaluation = evaluate_checkpoint_predictions(checkpoint_dir)
        else:
            rendered_dir: Path = config.output_root / scene
            render_test_split(
                render_binary=config.render_binary,
                ply_path=scene_ply_path(scene),
                camera_path=scene_data_dir(scene) / "transforms_test.json",
                output_dir=rendered_dir,
                width=config.width,
                height=config.height,
            )
            report = evaluate_predictions_against_checkpoint(rendered_dir, checkpoint_dir)
        reports.append(report)
        print(
            f"{report.scene}: {report.image_count} images, "
            f"PSNR {report.measured_psnr:.8f} / {report.published_psnr:.5f} "
            f"(delta {report.psnr_delta:+.8f}), "
            f"SSIM {report.measured_ssim:.9f} / {report.published_ssim:.5f} "
            f"(delta {report.ssim_delta:+.9f})"
        )

    config.report.parent.mkdir(parents=True, exist_ok=True)
    report_data = [dataclasses.asdict(report) for report in reports]
    config.report.write_text(json.dumps(report_data, indent=2) + "\n")
    print(f"Wrote {config.report}")
