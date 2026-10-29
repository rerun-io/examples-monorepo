"""Train Hexels with upstream's recipe under our protocol (run-layout.md) and write a full run directory.

Upstream `scripts.train_scene` resolves the paper's fixed-population recipe for the scene (images and
resolution per dataset, every 8th view held out); we run that command ourselves so the trainer is our
direct child (peak VRAM by NVML on its pid). The 30k-tied schedules are rescaled to the step count:
`lr_decay_milestones` 15001/20001/25001 and `lr_means3d_max_steps` 30000 scale by steps/30000, the same
policy as upstream's short smoke recipe (`scripts/prepare_nvs_smoke.py`); otherwise a 7k run would stop
before any learning-rate decay.
"""

import json
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from serde.json import to_json

from hexels_bench.apis.export import export_run
from hexels_bench.apis.export_published import stage_ground_truth
from hexels_bench.apis.score import score_run
from hexels_bench.brush import read_sfm_points
from hexels_bench.cameras import CameraContract, cameras_file, write_cameras_file
from hexels_bench.checkpoint import find_checkpoint
from hexels_bench.gpu_monitor import Measured, gpu_name, run_measured
from hexels_bench.hexel_ply import HexelPly, HexelVariant
from hexels_bench.layout import UPSTREAM_DATASET, Dataset, Roots, read_json
from hexels_bench.methods import primitive_bytes
from hexels_bench.metrics import TrainRecord
from hexels_bench.upstream import UPSTREAM_HEXELS_REV, UpstreamRenderer, load_renderer, measure_fps, write_test_renders

UPSTREAM_STEPS: int = 30000
UPSTREAM_MILESTONES: tuple[int, ...] = (15001, 20001, 25001)


def scaled_schedule(steps: int) -> tuple[list[int], int]:
    """lr_decay_milestones and lr_means3d_max_steps for a run of `steps` (identity at 30k)."""
    milestones: list[int] = [round((milestone - 1) * steps / UPSTREAM_STEPS) + 1 for milestone in UPSTREAM_MILESTONES]
    return milestones, steps


def apply_schedule(command: list[str], steps: int) -> list[str]:
    """Replace the 30k-tied schedule values in upstream's resolved training command."""
    milestones, means_max_steps = scaled_schedule(steps)
    result: list[str] = list(command)
    if "--lr_decay_milestones" in result:
        index: int = result.index("--lr_decay_milestones")
        end: int = index + 1
        while end < len(result) and not result[end].startswith("--"):
            end += 1
        result[index + 1 : end] = [str(value) for value in milestones]
    else:
        result += ["--lr_decay_milestones", *(str(value) for value in milestones)]
    if "--lr_means3d_max_steps" in result:
        result[result.index("--lr_means3d_max_steps") + 1] = str(means_max_steps)
    else:
        result += ["--lr_means3d_max_steps", str(means_max_steps)]
    return result


@dataclass
class Config:
    """Train one scene with upstream Hexels under our protocol."""

    dataset: Dataset
    """Run-layout dataset name."""
    scene: str
    """Upstream scene name."""
    variant: HexelVariant = "v1"
    """Appearance variant."""
    steps: int = 7000
    """Optimisation steps."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def prepare_command(scene_dir: Path, model_dir: Path, dataset: Dataset, variant: HexelVariant, steps: int) -> tuple[list[str], Path]:
    """Upstream's resolved training command (`scripts.train_scene --prepare`) with our schedule."""
    output: str = subprocess.run(
        [sys.executable, "-m", "scripts.train_scene", "--source", str(scene_dir), "--output", str(model_dir),
         "--dataset", UPSTREAM_DATASET[dataset], "--variant", variant, "--iterations", str(steps), "--prepare"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    prepared: dict = json.loads(output)
    return apply_schedule([str(part) for part in prepared["command"]], steps), Path(prepared["cwd"])


def train(roots: Roots, dataset: Dataset, scene: str, variant: HexelVariant, steps: int) -> Path:
    method: str = f"hexels-{variant}-{steps // 1000}k" if steps % 1000 == 0 else f"hexels-{variant}-{steps}"
    run_dir: Path = roots.run_dir(dataset, scene, method)
    model_dir: Path = run_dir / "upstream"
    if model_dir.exists():
        raise FileExistsError(f"{model_dir} exists; upstream needs a fresh output directory")
    scene_dir: Path = roots.scene_dir(dataset, scene)
    run_dir.mkdir(parents=True, exist_ok=True)
    command, cwd = prepare_command(scene_dir, model_dir, dataset, variant, steps)
    measured: Measured = run_measured(command, cwd, run_dir / "log.txt")
    if measured.returncode != 0:
        raise RuntimeError(f"upstream training exited with {measured.returncode}; see {run_dir / 'log.txt'}")
    summary: dict = json.loads((model_dir / "summary.json").read_text())

    ply: HexelPly = export_run(model_dir, run_dir / "model.ply")
    contract: CameraContract = read_json(CameraContract, roots.contract(dataset, scene))
    stage_ground_truth(roots, dataset, scene, contract)
    write_cameras_file(run_dir / "cameras.json", cameras_file(contract, dataset, scene, [0.0, 0.0, 0.0]))
    renderer: UpstreamRenderer = load_renderer(model_dir, find_checkpoint(model_dir), scene_dir)
    write_test_renders(renderer, [camera.name for camera in contract.geometry.test], run_dir / "renders" / "test")
    fps: float = measure_fps(renderer)
    del renderer
    torch.cuda.empty_cache()
    score_run(roots, dataset, scene, method, primitive_bytes(variant), fps, torch.device("cuda"))

    record = TrainRecord(
        method=method,
        dataset=dataset,
        scene=scene,
        steps=steps,
        loop_seconds=float(summary["train_seconds"]),
        total_seconds=measured.seconds,
        peak_vram_gib=measured.vram.peak_gib(),
        primitives=ply.cloud.count,
        sfm_points=len(read_sfm_points(scene_dir / "sparse" / "0").xyz),
        gpu=gpu_name(),
        host=socket.gethostname(),
        source_sha=UPSTREAM_HEXELS_REV,
        command=" ".join(command),
    )
    (run_dir / "train.json").write_text(to_json(record) + "\n")
    return run_dir


def main(config: Config) -> None:
    run_dir: Path = train(config.roots, config.dataset, config.scene, config.variant, config.steps)
    print(f"wrote {run_dir}")
