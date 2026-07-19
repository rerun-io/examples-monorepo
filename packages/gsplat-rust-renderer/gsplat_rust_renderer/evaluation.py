"""Full-scene image-quality evaluation for nerfbaselines checkpoints."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float32

from gsplat_rust_renderer.metrics import load_image_rgb, psnr, ssim


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate image metrics for one evaluated scene."""

    image_count: int
    """Number of paired images included in the aggregate."""
    psnr: float
    """Arithmetic mean of per-image PSNR values in decibels."""
    ssim: float
    """Arithmetic mean of per-image SSIM values."""


@dataclass(frozen=True, slots=True)
class CheckpointEvaluation:
    """Measured checkpoint metrics compared with its published results."""

    scene: str
    """Scene name recorded in the checkpoint metadata."""
    image_count: int
    """Number of paired test images included in the aggregate."""
    measured_psnr: float
    """PSNR recomputed from the checkpoint prediction images."""
    published_psnr: float
    """PSNR stored in the checkpoint's ``results.json``."""
    psnr_delta: float
    """Signed measured-minus-published PSNR difference."""
    measured_ssim: float
    """SSIM recomputed from the checkpoint prediction images."""
    published_ssim: float
    """SSIM stored in the checkpoint's ``results.json``."""
    ssim_delta: float
    """Signed measured-minus-published SSIM difference."""


def render_test_split(
    *,
    render_binary: Path,
    ply_path: Path,
    camera_path: Path,
    output_dir: Path,
    width: int,
    height: int,
) -> None:
    """Render every camera in a NeRF test split with ``gsplat-render``.

    The standalone binary is invoked once so GPU initialization, uploaded
    splats, and renderer scratch buffers are reused across all frames.

    Args:
        render_binary: Path to the standalone ``gsplat-render`` executable.
        ply_path: Path to the scene's pretrained Gaussian PLY.
        camera_path: Path to ``transforms_test.json``.
        output_dir: Directory below which relative frame paths are written.
        width: Render width in pixels.
        height: Render height in pixels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command: list[str] = [
        str(render_binary),
        "--ply",
        str(ply_path),
        "--camera",
        str(camera_path),
        "--output-dir",
        str(output_dir),
        "--width",
        str(width),
        "--height",
        str(height),
        "--background",
        "1,1,1",
    ]
    subprocess.run(command, check=True)


def evaluate_prediction_directory(rendered_dir: Path, ground_truth_dir: Path) -> EvaluationMetrics:
    """Evaluate matching PNGs below two directories.

    Images are paired by their relative paths. Aggregates are arithmetic means
    over per-image metrics, matching nerfbaselines' scene-level convention.

    Args:
        rendered_dir: Root containing rendered PNG images.
        ground_truth_dir: Root containing ground-truth PNG images.

    Returns:
        The number of image pairs and their mean PSNR and SSIM.
    """
    rendered_relative_paths: list[Path] = sorted(
        (path.relative_to(rendered_dir) for path in rendered_dir.rglob("*.png")), key=Path.as_posix
    )
    ground_truth_relative_paths: list[Path] = sorted(
        (path.relative_to(ground_truth_dir) for path in ground_truth_dir.rglob("*.png")), key=Path.as_posix
    )
    if not rendered_relative_paths:
        raise ValueError(f"no PNG images found below {rendered_dir}")
    if rendered_relative_paths != ground_truth_relative_paths:
        rendered_set: set[Path] = set(rendered_relative_paths)
        ground_truth_set: set[Path] = set(ground_truth_relative_paths)
        missing_rendered: list[str] = sorted(path.as_posix() for path in ground_truth_set - rendered_set)
        missing_ground_truth: list[str] = sorted(path.as_posix() for path in rendered_set - ground_truth_set)
        raise ValueError(
            "image sets differ: "
            f"missing rendered={missing_rendered[:5]}, missing ground truth={missing_ground_truth[:5]}"
        )

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for relative_path in rendered_relative_paths:
        rendered_path: Path = rendered_dir / relative_path
        ground_truth_path: Path = ground_truth_dir / relative_path
        rendered_rgb: Float32[np.ndarray, "h w 3"] = load_image_rgb(rendered_path)
        ground_truth_rgb: Float32[np.ndarray, "h w 3"] = load_image_rgb(ground_truth_path)
        psnr_values.append(psnr(rendered_rgb, ground_truth_rgb))
        ssim_values.append(ssim(rendered_rgb, ground_truth_rgb))

    image_count: int = len(rendered_relative_paths)
    return EvaluationMetrics(
        image_count=image_count,
        psnr=sum(psnr_values) / image_count,
        ssim=sum(ssim_values) / image_count,
    )


def evaluate_predictions_against_checkpoint(rendered_dir: Path, checkpoint_dir: Path) -> CheckpointEvaluation:
    """Evaluate a rendered split against a nerfbaselines checkpoint.

    Args:
        rendered_dir: Root containing rendered images at checkpoint-relative
            paths such as ``test/r_0.png``.
        checkpoint_dir: Extracted checkpoint directory containing
            ``results.json`` and ``predictions/gt-color``.

    Returns:
        Recomputed and published metrics with signed differences.
    """
    results_data: dict[str, object] = json.loads((checkpoint_dir / "results.json").read_text())
    metrics_data: object = results_data.get("metrics")
    metadata: object = results_data.get("render_dataset_metadata")
    if not isinstance(metrics_data, dict) or not isinstance(metadata, dict):
        raise ValueError(f"{checkpoint_dir / 'results.json'} is missing metrics or render_dataset_metadata")

    scene_value: object = metadata.get("scene")
    published_psnr_value: object = metrics_data.get("psnr")
    published_ssim_value: object = metrics_data.get("ssim")
    if not isinstance(scene_value, str) or not isinstance(published_psnr_value, (int, float)) or not isinstance(published_ssim_value, (int, float)):
        raise ValueError(f"{checkpoint_dir / 'results.json'} has invalid scene, PSNR, or SSIM values")

    measured: EvaluationMetrics = evaluate_prediction_directory(
        rendered_dir,
        checkpoint_dir / "predictions" / "gt-color",
    )
    published_psnr: float = float(published_psnr_value)
    published_ssim: float = float(published_ssim_value)
    return CheckpointEvaluation(
        scene=scene_value,
        image_count=measured.image_count,
        measured_psnr=measured.psnr,
        published_psnr=published_psnr,
        psnr_delta=measured.psnr - published_psnr,
        measured_ssim=measured.ssim,
        published_ssim=published_ssim,
        ssim_delta=measured.ssim - published_ssim,
    )


def evaluate_checkpoint_predictions(checkpoint_dir: Path) -> CheckpointEvaluation:
    """Evaluate a nerfbaselines checkpoint's bundled prediction images.

    Args:
        checkpoint_dir: Extracted checkpoint directory containing
            ``results.json`` and ``predictions/{color,gt-color}``.

    Returns:
        Recomputed and published metrics with signed differences.
    """
    return evaluate_predictions_against_checkpoint(
        checkpoint_dir / "predictions" / "color",
        checkpoint_dir,
    )
