"""Score fused meshes with the paper's evaluators, run unchanged as subprocesses from the patched 2DGS
tree that `packages/hexels-src` installs: DTU Chamfer (`scripts/eval_dtu/evaluate_single_scene.py`:
mask culling, 0.2 mm sampling, ObsMask/plane crop, accuracy and completeness below 20 mm) and T&T
precision/recall/F1 (`scripts/eval_tnt/run.py`: trajectory alignment, ICP, crop, tau per scene) in the
`tnt-eval` env (Python 3.8, Open3D 0.10, upstream environment-tnt-eval.yml), through `pixi run --frozen` from
the workspace root, so Pixi installs that env from the lock or reports why it cannot.

Writes `<surface dir>/{dtu_eval/, tnt_eval/, surface_metrics.json}`.
"""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from serde import serde
from serde.json import from_json, to_json

from hexels_bench.apis.surface_run import SurfaceMethod, surface_out_dir
from hexels_bench.layout import WORKSPACE, Dataset, Roots
from hexels_bench.surface import DepthMode
from hexels_bench.surface_records import SurfaceMetrics

TWO_DGS: Path = Path(sys.prefix) / "share" / "hexels-src" / "2d-gaussian-splatting"
"""Patched 2DGS tree (hexels-src output) in this env."""


@dataclass
class Config:
    """Score fuse_post.ply of surface runs: DTU Chamfer distance or T&T F1."""

    dataset: Dataset
    """dtu or tnt."""
    scenes: list[str]
    """Upstream scene names."""
    methods: list[SurfaceMethod] = field(default_factory=lambda: ["hexels-v3-surf-30k-pub-wgsl"])
    """Surface runs to score."""
    depth_mode: DepthMode | None = None
    """The depth-mode override the run was fused with (surface-<mode>/), if any."""
    two_dgs: Path = TWO_DGS
    """Patched 2DGS checkout holding scripts/eval_dtu and scripts/eval_tnt."""
    workers: int = 8
    """Processes for the DTU evaluator's triangle sampling (its SLURM_CPUS_PER_TASK knob)."""
    force: bool = False
    """Re-score runs that already have surface_metrics.json."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


@serde
@dataclass(frozen=True, slots=True)
class DtuResults:
    """`results.json` of the 2DGS DTU evaluator (mm)."""

    mean_d2s: float
    """Accuracy: mean data-to-STL distance below max_dist."""
    mean_s2d: float
    """Completeness: mean STL-to-data distance below max_dist."""
    overall: float
    """Chamfer distance (mean_d2s + mean_s2d) / 2."""


def evaluate_dtu(config: Config, scene: str, method: str, mesh: Path, out_dir: Path, log: Path) -> SurfaceMetrics:
    scan_id: str = scene.removeprefix("scan")
    command: list[str] = [
        sys.executable, str(config.two_dgs / "scripts" / "eval_dtu" / "evaluate_single_scene.py"),
        "--input_mesh", str(mesh), "--scan_id", scan_id, "--output_dir", str(out_dir),
        "--mask_dir", str(config.roots.data / "dtu"), "--DTU", str(config.roots.data / "dtu_official"),
    ]  # fmt: skip
    _run(command, log, {"SLURM_CPUS_PER_TASK": str(config.workers)}, None)
    results: DtuResults = from_json(DtuResults, (out_dir / "results.json").read_text())
    return SurfaceMetrics(
        dataset="dtu", scene=scene, method=method, evaluator="2dgs-eval_dtu",
        cd_mm=results.overall, acc_mm=results.mean_d2s, comp_mm=results.mean_s2d, precision=None, recall=None, f1=None, tau=None,
    )  # fmt: skip


def evaluate_tnt(config: Config, scene: str, method: str, mesh: Path, out_dir: Path, log: Path) -> SurfaceMetrics:
    command: list[str] = [
        "pixi", "run", "-e", "tnt-eval", "--frozen", "python", str(config.two_dgs / "scripts" / "eval_tnt" / "run.py"),
        "--dataset-dir", str(config.roots.data / "tnt_official" / scene),
        "--traj-path", str(config.roots.data / "tanks_and_temples" / scene / f"{scene}_COLMAP_SfM.log"),
        "--ply-path", str(mesh), "--out-dir", str(out_dir),
    ]  # fmt: skip
    _run(command, log, {}, WORKSPACE)
    values: list[float] = [float(value) for value in (out_dir / f"{scene}.prf_tau_plotstr.txt").read_text().split()]
    return SurfaceMetrics(
        dataset="tnt", scene=scene, method=method, evaluator="2dgs-eval_tnt",
        cd_mm=None, acc_mm=None, comp_mm=None, precision=100 * values[0], recall=100 * values[1], f1=100 * values[2], tau=values[3],
    )  # fmt: skip


def _run(command: list[str], log: Path, env: dict[str, str], cwd: Path | None) -> None:
    with log.open("a") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, **env}, cwd=cwd)


def evaluate(config: Config, scene: str, method: SurfaceMethod) -> SurfaceMetrics | None:
    surface_dir: Path = surface_out_dir(config.roots, config.dataset, scene, method, config.depth_mode)
    target: Path = surface_dir / "surface_metrics.json"
    if target.is_file() and not config.force:
        print(f"skip {config.dataset}/{scene}/{method}: {target} exists")
        return None
    mesh: Path = surface_dir / "fuse_post.ply"
    log: Path = surface_dir / "log.txt"
    if config.dataset == "dtu":
        metrics: SurfaceMetrics = evaluate_dtu(config, scene, method, mesh, surface_dir / "dtu_eval", log)
    elif config.dataset == "tnt":
        metrics = evaluate_tnt(config, scene, method, mesh, surface_dir / "tnt_eval", log)
    else:
        raise ValueError(f"{config.dataset} has no surface ground truth")
    target.write_text(to_json(metrics) + "\n")
    print(f"{config.dataset}/{scene}/{method}: {to_json(metrics)}")
    return metrics


def main(config: Config) -> None:
    for scene in config.scenes:
        for method in config.methods:
            evaluate(config, scene, method)
