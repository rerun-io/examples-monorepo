"""Train Brush on the Hexels NVS protocol and write run directories (model.ply, 16-bit test renders, cameras.json, train.json).

The camera contract fixes the cameras, the every-8th held-out split and the target size (`hexels_bench.brush`
prepares the scene once). Training starts from the SfM points and caps the splat count at that number; all
other Brush settings keep their defaults. `gsplat-train` comes from the gsplat-rust-renderer package build.
"""

import shlex
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from serde import serde
from serde.json import to_json

from hexels_bench.brush import PreparedScene, prepare_scene
from hexels_bench.cameras import CameraContract, cameras_file, write_cameras_file
from hexels_bench.gpu_monitor import Measured, gpu_lock, gpu_name, run_measured
from hexels_bench.layout import GSPLAT_RELEASE, PROTOCOL_SCENES, Dataset, Roots, read_json
from hexels_bench.metrics import TrainRecord


@dataclass(frozen=True, slots=True)
class Config:
    """Scenes of one dataset, one method: prepare the protocol inputs, train, render the held-out views."""

    dataset: Dataset
    """Run-layout dataset name."""
    scenes: tuple[str, ...] = ()
    """Upstream scene names (garden, scan24, Truck); empty runs every protocol scene of the dataset."""
    steps: int = 7000
    """Training steps."""
    method: str = "brush-7k"
    """Run-layout method directory."""
    roots: Roots = field(default_factory=Roots)
    """Data (scenes, camera contracts), runs and Brush-prepared roots."""
    binary: Path = GSPLAT_RELEASE / "gsplat-train"
    """Built trainer executable."""
    gpu_lock: Path | None = None
    """File held with flock during each training, so GPU jobs sharing it do not overlap."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrainerSummary:
    """The trainer's ``--summary`` document."""

    steps: int
    """Steps trained."""
    loop_seconds: float
    """Brush's summed step time (no loading, evaluation or export)."""
    wall_seconds: float
    """End of loading to end of training, evaluation and export included."""
    training_end_unix_seconds: float
    """Wall clock when training, Brush's eval and export finished, before the test renders (the VRAM peak's cutoff)."""
    splats: int
    """Final splat count."""
    brush_eval_psnr: float | None
    """Brush's own last-step eval PSNR (8-bit render roundtrip)."""
    brush_eval_ssim: float | None
    """Brush's own last-step eval SSIM."""
    renderer: str
    """Renderer that produced the test renders."""


def source_sha() -> str:
    """Commit of the trainer's package (gsplat-rust-renderer), marked dirty when its tracked files differ."""
    package: Path = GSPLAT_RELEASE.parents[1]
    git: list[str] = ["git", "-C", str(package)]
    sha: str = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty: str = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no", "--", "."], capture_output=True, text=True, check=True).stdout
    return f"{sha}-dirty" if dirty.strip() else sha


def main(config: Config, extra_flags: list[str]) -> None:
    """Run every selected scene; a failing scene is reported at the end and does not stop the others."""
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    forwarded: list[str] = extra_flags[1:] if extra_flags[:1] == ["--"] else extra_flags
    scenes: tuple[str, ...] = config.scenes or PROTOCOL_SCENES[config.dataset]
    failures: list[str] = []
    for scene in scenes:
        try:
            run_scene(config, scene, forwarded)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"FAILED {config.dataset}/{scene}: {error}")
            failures.append(scene)
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(scenes)} scenes failed: {', '.join(failures)}")


def run_scene(config: Config, scene: str, forwarded: list[str]) -> None:
    """Prepare one scene, train with Brush defaults capped at the SfM count, and write its run directory."""
    roots: Roots = config.roots
    contract: Path = roots.contract(config.dataset, scene)
    prepared_dir: Path = roots.brush_prepared / config.dataset / scene
    prepared: PreparedScene = prepare_scene(roots.scene_dir(config.dataset, scene), contract, prepared_dir)
    run: Path = roots.run_dir(config.dataset, scene, config.method)
    if (run / "train.json").is_file():
        print(f"Complete, skipped: {run}")
        return
    run.mkdir(parents=True, exist_ok=True)
    summary_path: Path = run / "brush-summary.json"
    steps: str = str(config.steps)
    command: list[str] = [
        str(config.binary.resolve()), str(prepared_dir.resolve()),
        "--total-train-iters", steps, "--max-splats", str(prepared.sfm_points), "--eval-every", steps, "--export-every", steps,
        "--export-path", str(run.resolve()), "--export-name", "model.ply",
        "--test-renders", str((run / "renders/test").resolve()), "--summary", str(summary_path.resolve()), *forwarded,
    ]  # fmt: skip
    if config.gpu_lock is not None:
        print(f"Waiting for {config.gpu_lock}")
    with gpu_lock(config.gpu_lock):
        print(f"Training {config.dataset}/{scene} -> {run}")
        measured: Measured = run_measured(command, None, run / "log.txt")
    summary: TrainerSummary = read_json(TrainerSummary, summary_path)
    renders: int = len(list((run / "renders/test").glob("*.png")))
    if renders != prepared.test_views:
        raise RuntimeError(f"{run}: {renders} test renders for {prepared.test_views} held-out views")
    write_cameras_file(run / "cameras.json", cameras_file(read_json(CameraContract, contract), config.dataset, scene, [0.0, 0.0, 0.0]))
    report: TrainRecord = TrainRecord(
        method=config.method,
        dataset=config.dataset,
        scene=scene,
        steps=summary.steps,
        loop_seconds=summary.loop_seconds,
        total_seconds=measured.seconds,
        peak_vram_gib=measured.vram.peak_gib(until_unix_s=summary.training_end_unix_seconds),
        primitives=summary.splats,
        sfm_points=prepared.sfm_points,
        gpu=gpu_name(),
        host=socket.gethostname(),
        source_sha=source_sha(),
        command=shlex.join(command),
    )
    (run / "train.json").write_text(to_json(report) + "\n")
    print(
        f"{run}: loop {summary.loop_seconds:.1f} s, total {measured.seconds:.1f} s, {summary.splats} splats, Brush eval PSNR {summary.brush_eval_psnr}"
    )
