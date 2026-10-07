"""Full-scene image-quality evaluation for nerfbaselines checkpoints."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from tempfile import TemporaryDirectory

from serde import SerdeError, serde
from serde.json import from_json


@serde
@dataclass(frozen=True, slots=True)
class Metrics:
    """Read the required fields from Rust-owned image metrics."""
    psnr: float
    ssim: float


@serde
@dataclass(frozen=True, slots=True)
class ViewMetrics:
    """Read the required fields from one Rust-owned image result."""
    name: str
    psnr: float
    ssim: float


@serde
@dataclass(frozen=True, slots=True)
class Evaluation:
    """Read Rust-owned evaluation results without mirroring its provenance schema."""
    views: list[ViewMetrics]
    mean: Metrics


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
    """Render every NeRF test camera in one process, reusing GPU resources."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command: list[str] = [
        str(render_binary), "render", "--ply", str(ply_path), "--camera", str(camera_path),
        "--output-dir", str(output_dir), "--width", str(width), "--height", str(height), "--background", "1,1,1",
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
    """Compare a rendered split with the published checkpoint metrics."""
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
