"""CLI orchestration for direct gsplat training and fused publication."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gsplat
import torch

from gauss_surf.catalog import DEFAULT_CATALOG_URL, DEFAULT_DATASET_NAME, SegmentReader
from gauss_surf.contracts import LAYERS, SPLAT_DEPTH_LAYER, SPLAT_LAYER, SPLAT_TRIAGE_LAYER
from gauss_surf.train_gsplat.cache import load_gpu_cache
from gauss_surf.train_gsplat.evaluation import evaluate_holdout
from gauss_surf.train_gsplat.metadata import MetadataStats, metadata_files_complete, prepare_training_metadata
from gauss_surf.train_gsplat.publish import PublishStats, publish_in_process
from gauss_surf.train_gsplat.trainer import ITERATIONS, metric_training_constants, train


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for direct metric-world gsplat training."""

    video_id: str
    """Catalog segment identifier."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """Existing Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset read by the cache and targeted by publication."""
    bundle_dir: Path | None = None
    """Camera/seed metadata directory; defaults to a fresh per-segment catalog-derived path."""
    output_root: Path = Path("data/splat_runs")
    """Root for direct metric-world runs."""
    timestamp: str = "run"
    """Unique training run name."""
    splat_output_dir: Path = Path(LAYERS[SPLAT_LAYER]).parent
    """Destination for the in-process static Gaussian-splat RRD."""
    depth_output_dir: Path = Path(LAYERS[SPLAT_DEPTH_LAYER]).parent
    """Destination for the in-process two-camera depth RRD."""
    triage_output_dir: Path = Path(LAYERS[SPLAT_TRIAGE_LAYER]).parent
    """Destination for the in-process normal, RGB, and diagnostic RRD."""
    publish_batch_size: int = 8
    """Full-grid products encoded and sent per bounded-memory batch."""
    encoder_workers: int = 8
    """CPU worker threads used for PNG and JPEG encoding."""
    seed: int = 42
    """Training and camera-sampling seed."""
    opacity_reset_every: int = 1_000_000_000
    """Opacity reset interval; 3000 selects the measured Stage 2 path."""
    fused_training: bool = False
    """Opt into fused plane gradients for a future quality round; publication is always fused."""


def _run_dir(config: Config) -> Path:
    return config.output_root / f"{config.video_id}-gaussurf" / "gsplat-direct" / config.timestamp


def main(config: Config) -> None:
    """Run metric-world training, fused evaluation, and direct RRD publication."""
    if not torch.cuda.is_available():
        raise SystemExit("direct gsplat training requires CUDA")
    segment_started_at: float = time.perf_counter()
    device: torch.device = torch.device("cuda")
    reader: SegmentReader = SegmentReader.open(config.catalog_url, config.dataset_name, config.video_id)
    bundle_dir: Path = config.bundle_dir if config.bundle_dir is not None else Path("data/training_metadata") / config.video_id
    if not metadata_files_complete(bundle_dir, allow_legacy=config.bundle_dir is not None):
        metadata_stats: MetadataStats = prepare_training_metadata(reader, bundle_dir)
        print(
            f"generated training metadata: {bundle_dir} wide={metadata_stats.wide_frames} "
            f"ultrawide={metadata_stats.ultrawide_frames} holdout={metadata_stats.holdout_frames} "
            f"render_wide={metadata_stats.render_wide_frames} "
            f"render_ultrawide={metadata_stats.render_ultrawide_frames} "
            f"seed_vertices={metadata_stats.seed_vertices}",
            flush=True,
        )
    cache = load_gpu_cache(reader, bundle_dir, device)
    result = train(
        cache,
        bundle_dir=bundle_dir,
        run_dir=_run_dir(config),
        seed=config.seed,
        opacity_reset_every=config.opacity_reset_every,
        fused_training=config.fused_training,
    )
    holdout_metrics: dict[str, float] = evaluate_holdout(result.splats, cache)
    publish_stats: PublishStats = publish_in_process(
        result.splats,
        reader=reader,
        video_id=config.video_id,
        bundle_dir=bundle_dir,
        splat_output_dir=config.splat_output_dir,
        depth_output_dir=config.depth_output_dir,
        triage_output_dir=config.triage_output_dir,
        batch_size=config.publish_batch_size,
        encoder_workers=config.encoder_workers,
    )
    segment_wall_seconds: float = time.perf_counter() - segment_started_at
    summary: dict[str, Any] = {
        "seed": config.seed,
        "iterations": ITERATIONS,
        "dataset_read": config.dataset_name,
        "holdout_sha256": cache.holdout_sha256,
        "holdout_count": len(cache.holdout_indices),
        "train_camera_count": len(cache.train_indices),
        "versions": {
            "gsplat": gsplat.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "scene_scale_m": cache.scene_scale,
        "metric_training_constants": asdict(metric_training_constants(cache.scene_scale)),
        "opacity_reset_every": config.opacity_reset_every,
        "fused_training": config.fused_training,
        "metrics": holdout_metrics,
        "gaussian_count": result.gaussian_count,
        "exported_gaussian_count": result.exported_gaussian_count,
        "training_wall_seconds": result.training_wall_seconds,
        "train_evaluate_publish_wall_seconds": result.training_wall_seconds + holdout_metrics["wall_seconds"] + publish_stats.wall_seconds,
        "segment_command_internal_wall_seconds": segment_wall_seconds,
        "publication": asdict(publish_stats),
        "refinement_steps": result.refinement_steps,
        "reset_steps": result.reset_steps,
        "strategy_trajectory": [asdict(event) for event in result.strategy_trajectory],
        "final_losses": result.final_losses,
    }
    (_run_dir(config) / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"completed direct gsplat run: {_run_dir(config)}")
