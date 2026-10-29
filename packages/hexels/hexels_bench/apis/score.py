"""Score one run directory with the shared scorer: `renders/test/*.png` vs the scene's `gt/test/*.png`."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from hexels_bench.cameras import CamerasFile, load_cameras_file
from hexels_bench.layout import Dataset, Roots
from hexels_bench.metrics import (
    MetricsFile,
    ViewMetrics,
    bytes_per_primitive,
    load_metrics_file,
    ply_vertex_count,
    score_views,
    summarize,
    write_metrics_file,
)


@dataclass
class Config:
    """Write `metrics.json` (scorer shared-png16) for `runs/<dataset>/<scene>/<method>/`."""

    dataset: Dataset
    """Run-layout dataset name."""
    scene: str
    """Scene name."""
    method: str
    """Run method directory name, e.g. brush-7k or hexels-v1-7k."""
    fps: float | None = None
    """Frames per second measured by the run's renderer, if any; kept from an existing metrics.json otherwise."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""
    device: str = "cuda"
    """Torch device for SSIM and LPIPS."""


def score_run(
    roots: Roots, dataset: Dataset, scene: str, method: str, fps: float | None, device: torch.device, with_lpips: bool = True
) -> MetricsFile:
    run_dir: Path = roots.run_dir(dataset, scene, method)
    cameras: CamerasFile = load_cameras_file(run_dir / "cameras.json")
    rows: list[ViewMetrics] = score_views(
        run_dir / "renders" / "test", roots.gt_dir(dataset, scene), [camera.name for camera in cameras.cameras], device, with_lpips
    )
    if fps is None and (run_dir / "metrics.json").is_file():
        fps = load_metrics_file(run_dir / "metrics.json").fps
    ply: Path = run_dir / "model.ply"
    size: int | None = bytes_per_primitive(method)
    deploy_mb: float | None = ply_vertex_count(ply) * size / 1e6 if ply.is_file() and size is not None else None
    ply_mb: float | None = ply.stat().st_size / 1e6 if ply.is_file() else None
    metrics: MetricsFile = summarize(
        rows, method=method, dataset=dataset, scene=scene, scorer="shared-png16", fps=fps, deploy_mb=deploy_mb, ply_mb=ply_mb
    )
    write_metrics_file(run_dir / "metrics.json", metrics)
    return metrics


def main(config: Config) -> None:
    metrics: MetricsFile = score_run(config.roots, config.dataset, config.scene, config.method, config.fps, torch.device(config.device))
    lpips_text: str = "n/a" if metrics.lpips_vgg is None else f"{metrics.lpips_vgg:.4f}"
    print(
        f"{config.dataset}/{config.scene}/{config.method}: {metrics.views} views, PSNR {metrics.psnr:.4f} dB, SSIM {metrics.ssim:.4f}, LPIPS {lpips_text}"
    )
