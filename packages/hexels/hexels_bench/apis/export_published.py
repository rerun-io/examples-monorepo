"""Stage a published 30k checkpoint (HF jhu-ccvl/Hexels) for the comparison, without the CUDA build:
`model.ply` and `cameras.json` in `runs/<dataset>/<scene>/hexels-<v>-30k-pub-cuda/`, plus the scene's
shared ground truth `runs/<dataset>/<scene>/gt/test/*.png` (bit-exact upstream targets, checked
against the camera contract digest), and `published_metrics.json` (the authors' scores from HF
evaluation.json). `render-published` adds the upstream CUDA renders and our measured metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path

from hexels_bench.apis.export import export_run
from hexels_bench.cameras import CameraContract, CamerasFile, cameras_file, write_cameras_file, write_targets
from hexels_bench.checkpoint import UpstreamRunConfig, load_run_config
from hexels_bench.hexel_ply import HexelPly, HexelVariant
from hexels_bench.layout import Dataset, Roots, read_json
from hexels_bench.methods import METHODS, Method, MethodSpec, primitive_bytes
from hexels_bench.metrics import MetricsFile, ViewMetrics, ply_vertex_count, summarize, write_metrics_file
from hexels_bench.upstream import PublishedEvaluation


@dataclass
class Config:
    """Stage published checkpoints: hexel PLY, cameras.json and ground truth."""

    dataset: Dataset
    """Run-layout dataset name."""
    scenes: list[str]
    """Upstream scene names, e.g. scan24 garden."""
    methods: list[Method] = field(default_factory=lambda: ["hexels-v1-30k-pub-cuda", "hexels-v2-30k-pub-cuda", "hexels-v3-30k-pub-cuda"])
    """Published CUDA methods to stage."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def published_variant(method: Method) -> HexelVariant:
    """Appearance variant of a published checkpoint rendered by upstream CUDA (`hexels-v*-30k-pub-cuda`); raises for any other method."""
    spec: MethodSpec = METHODS[method]
    if spec.variant is None or not spec.published or spec.renders_from is not None:
        raise ValueError(f"{method} is not a published checkpoint rendered by upstream CUDA")
    return spec.variant


def stage_ground_truth(roots: Roots, dataset: Dataset, scene: str, contract: CameraContract) -> Path:
    gt_dir: Path = roots.gt_dir(dataset, scene)
    expected: list[Path] = [gt_dir / f"{camera.name}.png" for camera in contract.geometry.test]
    if not all(path.is_file() for path in expected):
        write_targets(contract, roots.scene_dir(dataset, scene), gt_dir)
    return gt_dir


def stage_published(roots: Roots, dataset: Dataset, scene: str, method: Method) -> Path:
    """Write model.ply and cameras.json of one published checkpoint; returns the run directory."""
    variant: HexelVariant = published_variant(method)
    contract: CameraContract = read_json(CameraContract, roots.contract(dataset, scene))
    stage_ground_truth(roots, dataset, scene, contract)
    model_dir: Path = roots.published_model(dataset, scene, variant)
    run_config: UpstreamRunConfig = load_run_config(model_dir / "config.json")
    run_dir: Path = roots.run_dir(dataset, scene, method)
    ply: HexelPly = export_run(model_dir, run_dir / "model.ply")
    if ply.cloud.variant != variant:
        raise ValueError(f"{model_dir}: config says {ply.cloud.variant}, expected {variant}")
    background: list[float] = [1.0, 1.0, 1.0] if run_config.white_background else [0.0, 0.0, 0.0]
    cameras: CamerasFile = cameras_file(contract, dataset, scene, background)
    write_cameras_file(run_dir / "cameras.json", cameras)
    write_metrics_file(run_dir / "published_metrics.json", published_metrics(model_dir, run_dir, dataset, scene, method))
    return run_dir


def published_metrics(model_dir: Path, run_dir: Path, dataset: Dataset, scene: str, method: Method) -> MetricsFile:
    """The authors' published direct-float scores (HF evaluation.json) in metrics.json form.
    Not re-measured: our own measurements go to metrics.json (shared-png16) and the upstream evaluator
    run by `render-published`."""
    evaluation: PublishedEvaluation = read_json(PublishedEvaluation, model_dir / "evaluation.json")
    rows: list[ViewMetrics] = [
        ViewMetrics(name=view.image_name, psnr=view.psnr_db, ssim=view.ssim, lpips_vgg=view.lpips_vgg) for view in evaluation.per_view
    ]
    ply: Path = run_dir / "model.ply"
    deploy_mb: float = ply_vertex_count(ply) * primitive_bytes(METHODS[method].variant) / 1e6
    return summarize(
        rows, method=method, dataset=dataset, scene=scene, scorer="upstream-float", fps=None, deploy_mb=deploy_mb, ply_mb=ply.stat().st_size / 1e6
    )


def main(config: Config) -> None:
    for scene in config.scenes:
        for method in config.methods:
            run_dir: Path = stage_published(config.roots, config.dataset, scene, method)
            print(f"staged {run_dir}")
