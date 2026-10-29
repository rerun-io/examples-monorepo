"""Train Brush on the Hexels NVS protocol and write run directories (model.ply, 16-bit test renders, train.json)."""

import shlex
import shutil
import socket
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from serde import serde
from serde.json import to_json

from gsplat_rust_renderer.bench import TrainRecord, gpu_lock
from gsplat_rust_renderer.nerfbaselines import DATA_ROOT
from gsplat_rust_renderer.nvs_protocol import (
    DATA_DIRS,
    PROTOCOL_SCENES,
    UPSTREAM_DATASETS,
    CameraContract,
    ContractCamera,
    Dataset,
    PreparedScene,
    prepare_scene,
    read_json,
)


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
    data_root: Path = DATA_ROOT.parent
    """Holds mipnerf360/, dtu/ and tnt_nvs/ scene directories."""
    contracts_root: Path = DATA_ROOT.parent / "hexels-pretrained/cameras/nvs"
    """Upstream camera contracts: <upstream dataset>/<scene>/contract.json."""
    prepared_root: Path = DATA_ROOT.parent / "brush-protocol"
    """Brush-ready scene directories, reused across runs."""
    runs_root: Path = DATA_ROOT.parent / "hexels-runs"
    """Run outputs: <dataset>/<scene>/<method>/."""
    binary: Path = Path("target/release/gsplat-train")
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
    """Wall clock when training, Brush's eval and export finished, before the test renders."""
    splats: int
    """Final splat count."""
    brush_eval_psnr: float | None
    """Brush's own last-step eval PSNR (8-bit render roundtrip)."""
    brush_eval_ssim: float | None
    """Brush's own last-step eval SSIM."""
    renderer: str
    """Renderer that produced the test renders."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RenderCameras:
    """``cameras.json`` of the run-layout contract: the upstream records of the rendered views."""

    format: str
    """Schema name."""
    dataset: str
    """Run-layout dataset name."""
    scene: str
    """Scene name."""
    split: str
    """Rendered split."""
    background: list[float]
    """Render background RGB."""
    cameras: list[ContractCamera]
    """Upstream camera records in render order."""


def process_memory_sample(line: str, pid: int) -> tuple[float, float] | None:
    """(Unix seconds, used MiB) of ``pid`` from one ``nvidia-smi --query-compute-apps=timestamp,pid,used_memory`` CSV line."""
    fields: list[str] = [field.strip() for field in line.split(",")]
    if len(fields) != 3 or fields[1] != str(pid):
        return None
    try:
        return datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(fields[2])
    except ValueError:
        return None


def wait_with_vram_samples(process: subprocess.Popen[bytes]) -> list[tuple[float, float]]:
    """Wait for ``process`` while nvidia-smi samples its GPU memory about every 5 ms; empty without nvidia-smi."""
    samples: list[tuple[float, float]] = []
    if shutil.which("nvidia-smi") is None:
        process.wait()
        return samples
    query: list[str] = ["nvidia-smi", "--query-compute-apps=timestamp,pid,used_memory", "--format=csv,noheader,nounits", "-lms", "5"]
    sampler: subprocess.Popen[str] = subprocess.Popen(query, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def read() -> None:
        assert sampler.stdout is not None
        for line in sampler.stdout:
            sample: tuple[float, float] | None = process_memory_sample(line, process.pid)
            if sample is not None:
                samples.append(sample)

    reader: threading.Thread = threading.Thread(target=read, daemon=True)
    reader.start()
    process.wait()
    sampler.terminate()
    sampler.wait()
    reader.join()
    return samples


def peak_training_vram_mib(samples: list[tuple[float, float]], training_end: float) -> float | None:
    """Largest sample taken before training ended; the test renders afterwards are evaluation, not training."""
    return max((used for when, used in samples if when <= training_end), default=None)


def gpu_name() -> str | None:
    """First GPU's name, or None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    output: str = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout
    return output.splitlines()[0].strip()


def source_sha() -> str:
    """Commit of this package's checkout, marked dirty when tracked files under the package differ."""
    package: Path = Path(__file__).resolve().parents[2]
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
    contract: Path = config.contracts_root / UPSTREAM_DATASETS[config.dataset] / scene / "contract.json"
    prepared_dir: Path = config.prepared_root / config.dataset / scene
    prepared: PreparedScene = prepare_scene(config.data_root / DATA_DIRS[config.dataset] / scene, contract, prepared_dir)
    run: Path = config.runs_root / config.dataset / scene / config.method
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
    ]
    if config.gpu_lock is not None:
        print(f"Waiting for {config.gpu_lock}")
    with (run / "log.txt").open("wb") as log, gpu_lock(config.gpu_lock) if config.gpu_lock is not None else nullcontext():
        print(f"Training {config.dataset}/{scene} -> {run}")
        start: float = time.perf_counter()
        trainer: subprocess.Popen[bytes] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        samples: list[tuple[float, float]] = wait_with_vram_samples(trainer)
        total_seconds: float = time.perf_counter() - start
    if trainer.returncode != 0:
        raise RuntimeError(f"gsplat-train exited with {trainer.returncode}; see {run / 'log.txt'}")
    summary: TrainerSummary = read_json(TrainerSummary, summary_path)
    renders: int = len(list((run / "renders/test").glob("*.png")))
    if renders != prepared.test_views:
        raise RuntimeError(f"{run}: {renders} test renders for {prepared.test_views} held-out views")
    peak_mib: float | None = peak_training_vram_mib(samples, summary.training_end_unix_seconds)
    cameras: RenderCameras = RenderCameras(
        format="hexels-cameras-1", dataset=config.dataset, scene=scene, split="test", background=[0.0, 0.0, 0.0],
        cameras=read_json(CameraContract, contract).geometry.test,
    )
    (run / "cameras.json").write_text(to_json(cameras))
    report: TrainRecord = TrainRecord(
        method=config.method, dataset=config.dataset, scene=scene, steps=summary.steps,
        loop_seconds=summary.loop_seconds, total_seconds=total_seconds,
        peak_vram_gib=None if peak_mib is None else peak_mib / 1024.0,
        primitives=summary.splats, sfm_points=prepared.sfm_points, gpu=gpu_name(), host=socket.gethostname(),
        source_sha=source_sha(), command=shlex.join(command),
    )
    (run / "train.json").write_text(to_json(report))
    print(f"{run}: loop {summary.loop_seconds:.1f} s, total {total_seconds:.1f} s, {summary.splats} splats, Brush eval PSNR {summary.brush_eval_psnr}")
