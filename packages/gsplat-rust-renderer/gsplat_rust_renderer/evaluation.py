"""Full-scene image-quality evaluation for nerfbaselines checkpoints."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from serde import SerdeError, serde
from serde.json import from_json


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Metrics:
    """Mean image metrics from the Rust evaluator."""
    psnr: float
    """Peak signal-to-noise ratio in dB."""
    ssim: float
    """Structural similarity."""
    lpips: float | None = None
    """Optional perceptual distance."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ViewMetrics:
    """One image's Rust evaluation."""
    name: str
    """Relative PNG path."""
    psnr: float
    """Peak signal-to-noise ratio in dB."""
    ssim: float
    """Structural similarity."""
    lpips: float | None = None
    """Optional perceptual distance."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Provenance:
    """Runtime source and dependency identity from the Rust evaluator."""
    crate_version: str
    """Evaluator package version."""
    source_sha: str | None
    """Explicit GSPLAT_SOURCE_SHA or the current checkout revision."""
    cargo_lock_sha256: str | None
    """Fingerprint of the workspace's resolved dependencies."""
    brush: str
    """Pinned upstream source and observer patch."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Evaluation:
    """Rust evaluator report."""
    views: list[ViewMetrics]
    """Per-image measurements."""
    mean: Metrics
    """Arithmetic per-image mean."""
    convention: Literal["brush", "published"]
    """Metric convention name."""
    provenance: Provenance
    """Runtime source and dependency identity."""


@serde(deny_unknown_fields=True)
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
    """Render every camera in a NeRF test split with ``gsplat render``.

    The standalone binary is invoked once so GPU initialization, uploaded
    splats, and renderer scratch buffers are reused across all frames.

    Args:
        render_binary: Path to the ``gsplat`` executable.
        ply_path: Path to the scene's pretrained Gaussian PLY.
        camera_path: Path to ``transforms_test.json``.
        output_dir: Directory below which relative frame paths are written.
        width: Render width in pixels.
        height: Render height in pixels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    command: list[str] = [
        str(render_binary),
        "render",
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


def evaluate_prediction_directory(
    rendered_dir: Path, ground_truth_dir: Path, *, eval_binary: Path | None = None,
) -> Evaluation:
    """Delegate published white-background metrics to the Rust evaluator."""
    binary: Path = eval_binary or Path(os.environ.get(
        "GSPLAT_BIN", str(Path(__file__).resolve().parents[1] / "target/release/gsplat"),
    ))
    with TemporaryDirectory(prefix="gsplat-eval-") as directory:
        report_path: Path = Path(directory) / "evaluation.json"
        subprocess.run([
            str(binary.resolve()), "eval", "--render", str(rendered_dir), "--gt", str(ground_truth_dir),
            "--convention", "published", "--out", str(report_path),
        ], check=True)
        report: Evaluation = from_json(Evaluation, report_path.read_text())
    return report


@serde
@dataclass(frozen=True, slots=True)
class PublishedMetrics:
    """Metrics read from third-party checkpoint metadata."""
    psnr: float
    """Published peak signal-to-noise ratio."""
    ssim: float
    """Published structural similarity."""


@serde
@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Scene identity from a third-party checkpoint."""
    scene: str
    """Benchmark scene name."""


@serde
@dataclass(frozen=True, slots=True)
class CheckpointResults:
    """The checkpoint metadata fields used by the quality guard."""
    metrics: PublishedMetrics
    """Published image metrics."""
    render_dataset_metadata: DatasetMetadata
    """Source scene metadata."""


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
    source: Path = checkpoint_dir / "results.json"
    try:
        published: CheckpointResults = from_json(CheckpointResults, source.read_text())
    except (SerdeError, JSONDecodeError) as error:
        raise ValueError(f"Invalid checkpoint metadata {source}: {error}") from error

    measured: Evaluation = evaluate_prediction_directory(
        rendered_dir,
        checkpoint_dir / "predictions" / "gt-color",
    )
    published_psnr: float = published.metrics.psnr
    published_ssim: float = published.metrics.ssim
    return CheckpointEvaluation(
        scene=published.render_dataset_metadata.scene,
        image_count=len(measured.views),
        measured_psnr=measured.mean.psnr,
        published_psnr=published_psnr,
        psnr_delta=measured.mean.psnr - published_psnr,
        measured_ssim=measured.mean.ssim,
        published_ssim=published_ssim,
        ssim_delta=measured.mean.ssim - published_ssim,
    )
