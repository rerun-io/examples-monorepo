"""Render a published 30k checkpoint with upstream CUDA: 16-bit test renders, paper-protocol FPS, the
shared scorer, and upstream's own direct-float evaluator for the same checkpoint.

Writes `runs/<dataset>/<scene>/hexels-<v>-30k-pub-cuda/{model.ply, cameras.json, renders/test/,
metrics.json (shared-png16), upstream_float_metrics.json, upstream-eval/, published_metrics.json, log.txt}`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from hexels_bench.apis.export_published import published_variant, stage_published
from hexels_bench.apis.score import score_run
from hexels_bench.cameras import CamerasFile
from hexels_bench.checkpoint import find_checkpoint
from hexels_bench.layout import Dataset, Roots, read_json
from hexels_bench.methods import Method
from hexels_bench.metrics import MetricsFile, ViewMetrics, summarize, write_metrics_file
from hexels_bench.upstream import UpstreamRenderer, load_renderer, measure_fps, run_upstream_evaluator, write_test_renders


@dataclass
class Config:
    """Render published checkpoints with upstream CUDA and score them twice (shared PNG and upstream float)."""

    dataset: Dataset
    """Run-layout dataset name."""
    scenes: list[str]
    """Upstream scene names."""
    methods: list[Method] = field(default_factory=lambda: ["hexels-v1-30k-pub-cuda", "hexels-v2-30k-pub-cuda", "hexels-v3-30k-pub-cuda"])
    """Published CUDA methods."""
    upstream_evaluator: bool = True
    """Also run upstream `scripts.evaluate_nvs_campaign evaluate` (direct-float metrics, fly-around video)."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def upstream_float_metrics(result: dict, method: Method, dataset: Dataset, scene: str, shared: MetricsFile) -> MetricsFile:
    """Upstream's evaluator result in our metrics.json schema (scorer upstream-float)."""
    rows: list[ViewMetrics] = [
        ViewMetrics(name=row["image_name"], psnr=row["psnr_db"], ssim=row["ssim"], lpips_vgg=row["lpips_vgg"]) for row in result["per_view"]
    ]
    return summarize(
        rows,
        method=method,
        dataset=dataset,
        scene=scene,
        scorer="upstream-float",
        fps=float(result["benchmark"]["median_fps"]),
        deploy_mb=shared.deploy_mb,
        ply_mb=shared.ply_mb,
    )


def render_published(roots: Roots, dataset: Dataset, scene: str, method: Method, upstream_evaluator: bool) -> MetricsFile:
    run_dir: Path = stage_published(roots, dataset, scene, method)
    model_dir: Path = roots.published_model(dataset, scene, published_variant(method))
    cameras: CamerasFile = read_json(CamerasFile, run_dir / "cameras.json")
    renderer: UpstreamRenderer = load_renderer(model_dir, find_checkpoint(model_dir), roots.scene_dir(dataset, scene))
    write_test_renders(renderer, [camera.name for camera in cameras.cameras], run_dir / "renders" / "test")
    fps: float = measure_fps(renderer)
    del renderer
    torch.cuda.empty_cache()
    shared: MetricsFile = score_run(roots, dataset, scene, method, fps, torch.device("cuda"))
    if upstream_evaluator:
        result: dict = run_upstream_evaluator(roots, model_dir, run_dir / "upstream-eval" / "metrics.json", run_dir / "log.txt")
        upstream: MetricsFile = upstream_float_metrics(result, method, dataset, scene, shared)
        write_metrics_file(run_dir / "upstream_float_metrics.json", upstream)
        print(
            f"{dataset}/{scene}/{method}: shared-png16 {shared.psnr:.4f} dB vs upstream-float {upstream.psnr:.4f} dB (delta {shared.psnr - upstream.psnr:+.4f})"
        )
    published: MetricsFile = read_json(MetricsFile, run_dir / "published_metrics.json")
    print(f"{dataset}/{scene}/{method}: published {published.psnr:.4f} dB, ours {shared.psnr:.4f} dB, FPS {fps:.1f}")
    return shared


def main(config: Config) -> None:
    for scene in config.scenes:
        for method in config.methods:
            render_published(config.roots, config.dataset, scene, method, config.upstream_evaluator)
