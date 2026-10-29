"""The Hexels 34-scene benchmark queue (~/hexels-work/contracts/run-layout.md).

An item is one (dataset, scene, method) run directory. Each method runs through its tool in its
own Pixi environment: the Brush protocol runner and the WGSL `gsplat render` in `gsplat-rust-renderer`,
upstream Hexels training, published-checkpoint CUDA renders and the shared scorer in `hexels`. An item is
complete when its `metrics.json` scores every held-out view (and the method's other records exist);
complete items are skipped, so a stopped queue resumes where it left off.
"""

import fcntl
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias

from serde import serde
from serde.json import from_json, to_json

from gsplat_rust_renderer.nvs_protocol import PROTOCOL_SCENES, Dataset, read_json

Method: TypeAlias = Literal[
    "brush-7k",
    "brush-7k-wgsl",
    "hexels-v1-7k",
    "hexels-v3-7k",
    "hexels-v1-30k-pub-cuda",
    "hexels-v2-30k-pub-cuda",
    "hexels-v3-30k-pub-cuda",
    "hexels-v1-30k-pub-wgsl",
    "hexels-v2-30k-pub-wgsl",
    "hexels-v3-30k-pub-wgsl",
]
METHODS: tuple[Method, ...] = (
    "brush-7k",
    "brush-7k-wgsl",
    "hexels-v1-7k",
    "hexels-v3-7k",
    "hexels-v1-30k-pub-cuda",
    "hexels-v2-30k-pub-cuda",
    "hexels-v3-30k-pub-cuda",
    "hexels-v1-30k-pub-wgsl",
    "hexels-v2-30k-pub-wgsl",
    "hexels-v3-30k-pub-wgsl",
)
"""Queue order within a scene: a WGSL re-render follows the run it re-renders."""
REPRESENTATIVE_SCENES: dict[Dataset, tuple[str, ...]] = {"dtu": ("scan24", "scan114"), "mipnerf360": ("garden", "counter"), "tnt": ("Barn", "Truck")}
"""Scope cut (contracts/scope.md): new compute runs on these scenes only, upstream order."""
TRAINED: frozenset[str] = frozenset({"brush-7k", "hexels-v1-7k", "hexels-v3-7k"})
"""Methods that train, so their run also needs `train.json`."""
WGSL_SOURCE: dict[str, str] = {
    "brush-7k-wgsl": "brush-7k",
    "hexels-v1-30k-pub-wgsl": "hexels-v1-30k-pub-cuda",
    "hexels-v2-30k-pub-wgsl": "hexels-v2-30k-pub-cuda",
    "hexels-v3-30k-pub-wgsl": "hexels-v3-30k-pub-cuda",
}
"""WGSL re-render -> the run whose model.ply and cameras.json it renders."""
Env: TypeAlias = Literal["gsplat-rust-renderer", "hexels"]


@serde
@dataclass(frozen=True, slots=True)
class CameraName:
    """The name of one record in `cameras.json` (other camera fields are not read here)."""

    name: str
    """GT image stem; the render is `renders/test/<name>.png`."""


@serde
@dataclass(frozen=True, slots=True)
class CameraNames:
    """The camera names of a run's `cameras.json` (schema hexels-cameras-1)."""

    cameras: list[CameraName]
    """Held-out views in render order."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ViewScore:
    """One held-out view of `metrics.json`."""

    name: str
    """Image stem."""
    psnr: float
    """PSNR in dB."""
    ssim: float
    """SSIM."""
    lpips_vgg: float | None
    """LPIPS-VGG, null when not scored."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunMetrics:
    """`metrics.json` of a run directory (run-layout.md)."""

    method: str
    """Run method."""
    dataset: str
    """Run-layout dataset."""
    scene: str
    """Scene name."""
    views: int
    """Scored views."""
    psnr: float
    """Mean per-view PSNR, dB."""
    ssim: float
    """Mean SSIM."""
    lpips_vgg: float | None
    """Mean LPIPS-VGG."""
    fps: float | None
    """Paper-protocol frames per second, null when not measured."""
    deploy_mb: float | None
    """Primitives x FP32 bytes per primitive / 1e6."""
    ply_mb: float | None
    """model.ply size, 1e6 bytes."""
    scorer: Literal["shared-png16", "upstream-float"]
    """Scorer that produced the numbers."""
    per_view: list[ViewScore]
    """Per-view scores in render order."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TrainRecord:
    """`train.json` of a trained run (run-layout.md; both writers, Brush and Hexels)."""

    method: str
    """Run method."""
    dataset: str
    """Run-layout dataset."""
    scene: str
    """Scene name."""
    steps: int
    """Optimisation steps."""
    loop_seconds: float
    """Optimisation loop wall time."""
    total_seconds: float
    """Training process wall time."""
    peak_vram_gib: float | None
    """Peak GPU memory of the training process, GiB."""
    primitives: int
    """Primitives in the exported model."""
    sfm_points: int | None
    """SfM points of the scene (the primitive cap)."""
    gpu: str | None
    """GPU name."""
    host: str
    """Host name."""
    source_sha: str
    """Source commit."""
    command: str
    """Training command line."""


@serde
@dataclass(frozen=True, slots=True)
class FpsReport:
    """`fps.json` written by `gsplat render --fps` (read partially; the CLI owns the schema)."""

    median_fps: float
    """Median of the trial frame rates."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FailureRecord:
    """One line of `runs/FAILURES.jsonl`."""

    time: str
    """Local ISO time of the failure."""
    dataset: str
    """Run-layout dataset."""
    scene: str
    """Scene name."""
    method: str
    """Run method."""
    error: str
    """What failed."""
    log: str
    """The item's log file."""


@dataclass(frozen=True, slots=True)
class Item:
    """One run directory of the queue."""

    dataset: Dataset
    """Run-layout dataset."""
    scene: str
    """Upstream scene name."""
    method: Method
    """Run method."""


@dataclass(frozen=True, slots=True)
class BenchPaths:
    """Where the queue reads and writes."""

    repo: Path = Path(__file__).resolve().parents[3]
    """Pixi workspace root (holds pixi.toml and packages/)."""
    runs: Path = Path.home() / "hexels-work" / "runs"
    """Run directories: <dataset>/<scene>/<method>/."""
    data: Path = Path.home() / "gsplat-modern-data"
    """Datasets and hexels-pretrained/ (camera contracts, published checkpoints)."""
    brush_prepared: Path = Path.home() / "hexels-work" / "brush-prepared"
    """Brush-ready protocol scenes (the Brush runner prepares missing ones)."""
    gpu_lock: Path = Path.home() / "hexels-work" / "gpu.lock"
    """flock file shared by every GPU-heavy job on the host."""

    @property
    def package(self) -> Path:
        return self.repo / "packages" / "gsplat-rust-renderer"

    @property
    def gsplat_binary(self) -> Path:
        """Release `gsplat` CLI (pixi run -e gsplat-rust-renderer gsplat-build)."""
        return self.package / "target" / "release" / "gsplat"

    def run_dir(self, item: Item) -> Path:
        return self.runs / item.dataset / item.scene / item.method


@dataclass(frozen=True, slots=True)
class Step:
    """One command of an item, run as `pixi run -e <env> --frozen <args>` in `cwd`."""

    name: str
    """Short step name for logs."""
    env: Env
    """Pixi environment."""
    args: tuple[str, ...]
    """Command and arguments (a Pixi task name or an executable)."""
    cwd: Path
    """Working directory."""


class StepFailed(RuntimeError):
    """A step exited with a nonzero status."""


Runner: TypeAlias = Callable[[Step, Path], None]
"""Runs a step and appends its output to the given log file; raises StepFailed on a nonzero exit."""


@dataclass(slots=True)
class QueueSummary:
    """What one pass over the queue did."""

    done: list[Item] = field(default_factory=list)
    """Items run to completion."""
    skipped: list[Item] = field(default_factory=list)
    """Items already complete."""
    failed: list[Item] = field(default_factory=list)
    """Items that failed (see FAILURES.jsonl)."""


def build_queue(datasets: tuple[Dataset, ...], scenes: tuple[str, ...] = (), methods: tuple[Method, ...] = METHODS) -> list[Item]:
    """Items dataset by dataset, scenes in upstream order, every method of a scene before the next scene.

    Args:
        datasets: Datasets in queue order.
        scenes: Scene filter; empty keeps every protocol scene.
        methods: Methods per scene, in this order.
    """
    known: set[str] = {scene for dataset in datasets for scene in PROTOCOL_SCENES[dataset]}
    unknown: list[str] = [scene for scene in scenes if scene not in known]
    if unknown:
        raise ValueError(f"scenes {unknown} are not protocol scenes of {datasets}")
    return [
        Item(dataset=dataset, scene=scene, method=method)
        for dataset in datasets
        for scene in PROTOCOL_SCENES[dataset]
        if not scenes or scene in scenes
        for method in methods
    ]


def incomplete_reason(run_dir: Path, method: str) -> str | None:
    """None when the run directory is a finished, scored run of `method`; otherwise what is missing."""
    if not (run_dir / "metrics.json").is_file():
        return "no metrics.json"
    if not (run_dir / "cameras.json").is_file():
        return "no cameras.json"
    try:
        metrics: RunMetrics = read_json(RunMetrics, run_dir / "metrics.json")
        names: list[str] = [camera.name for camera in read_json(CameraNames, run_dir / "cameras.json").cameras]
        if method in TRAINED:
            read_json(TrainRecord, run_dir / "train.json")
    except FileNotFoundError as error:
        return f"no {Path(str(error.filename)).name}"
    except ValueError as error:
        return f"unreadable record: {error}"
    renders: list[str] = sorted(path.stem for path in (run_dir / "renders" / "test").glob("*.png"))
    if metrics.method != method or metrics.scorer != "shared-png16":
        return f"metrics.json is for method {metrics.method}, scorer {metrics.scorer}"
    if renders != sorted(names):
        return f"{len(renders)} renders for {len(names)} cameras"
    if [view.name for view in metrics.per_view] != names or metrics.views != len(names):
        return f"metrics.json scores {metrics.views} views, cameras.json has {len(names)}"
    if (run_dir / "model.ply").exists() and (run_dir / "model.ply").stat().st_mtime > (run_dir / "metrics.json").stat().st_mtime:
        return "metrics.json is older than model.ply (retrained or re-exported since scoring)"
    if metrics.lpips_vgg is None:
        return "no LPIPS"
    if method in WGSL_SOURCE and metrics.fps is None:
        return "no FPS"
    if method.endswith("-pub-cuda") and not (run_dir / "upstream_float_metrics.json").is_file():
        return "no upstream_float_metrics.json"
    return None


def load_failures(path: Path) -> list[FailureRecord]:
    if not path.is_file():
        return []
    return [from_json(FailureRecord, line) for line in path.read_text().splitlines() if line.strip()]


@contextmanager
def gpu_lock(path: Path) -> Iterator[None]:
    """Hold `flock(path)` (exclusive) for the duration of the block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def to_attic(paths: BenchPaths, item: Item, partials: tuple[Path, ...]) -> None:
    """Move partial outputs of an item out of the run layout into one `runs/_attic/` directory, kept for inspection."""
    existing: list[Path] = [path for path in partials if path.exists()]
    if not existing:
        return
    stamp: str = datetime.now().strftime("%Y%m%dT%H%M%S-%f")
    attic: Path = paths.runs / "_attic" / f"{item.dataset}-{item.scene}-{item.method}-{stamp}"
    for path in existing:
        target: Path = attic if path == paths.run_dir(item) else attic / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
        print(f"moved partial {path} -> {target}")


def score_step(paths: BenchPaths, item: Item, fps: float | None) -> Step:
    fps_args: tuple[str, ...] = () if fps is None else ("--fps", repr(fps))
    args: tuple[str, ...] = ("hexels-score", "--dataset", item.dataset, "--scene", item.scene, "--method", item.method, *fps_args)
    return Step(name="score", env="hexels", args=args, cwd=paths.repo)


def brush_train_step(paths: BenchPaths, item: Item) -> Step:
    """The Brush protocol runner for one scene, without --gpu-lock (the conductor already holds it)."""
    args: tuple[str, ...] = (
        "python", "tools/train_protocol.py", "--dataset", item.dataset, "--scenes", item.scene, "--method", item.method,
        "--data-root", str(paths.data), "--contracts-root", str(paths.data / "hexels-pretrained" / "cameras" / "nvs"),
        "--prepared-root", str(paths.brush_prepared), "--runs-root", str(paths.runs),
        "--binary", str(paths.package / "target" / "release" / "gsplat-train"),
    )  # fmt: skip
    return Step(name="brush-train", env="gsplat-rust-renderer", args=args, cwd=paths.package)


def wgsl_render_step(paths: BenchPaths, run_dir: Path) -> Step:
    args: tuple[str, ...] = (
        str(paths.gsplat_binary), "render", "--ply", str(run_dir / "model.ply"), "--camera", str(run_dir / "cameras.json"),
        "--output-dir", str(run_dir / "renders" / "test"), "--background", "0,0,0", "--png16", "--fps", str(run_dir / "fps.json"),
    )  # fmt: skip
    return Step(name="wgsl-render", env="gsplat-rust-renderer", args=args, cwd=paths.package)


def stage_wgsl(paths: BenchPaths, item: Item) -> Path:
    """Link the source run's model.ply and copy its cameras.json into the WGSL run directory."""
    source: Path = paths.runs / item.dataset / item.scene / WGSL_SOURCE[item.method]
    if not (source / "model.ply").is_file() or not (source / "cameras.json").is_file():
        raise RuntimeError(f"source run {source} has no model.ply/cameras.json yet")
    if item.method == "brush-7k-wgsl" and not (source / "train.json").is_file():
        raise RuntimeError(f"source run {source} has not finished training (no train.json)")
    run_dir: Path = paths.run_dir(item)
    run_dir.mkdir(parents=True, exist_ok=True)
    link: Path = run_dir / "model.ply"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("..") / source.name / "model.ply")
    shutil.copyfile(source / "cameras.json", run_dir / "cameras.json")
    return run_dir


def execute(paths: BenchPaths, item: Item, run: Runner) -> None:
    """Run one incomplete item to completion; raises when a step fails or the result is still incomplete."""
    run_dir: Path = paths.run_dir(item)
    log: Path = run_dir / "log.txt"
    if item.method in WGSL_SOURCE:
        stage_wgsl(paths, item)
        run(wgsl_render_step(paths, run_dir), log)
        run(score_step(paths, item, read_json(FpsReport, run_dir / "fps.json").median_fps), log)
    elif item.method == "brush-7k":
        if not (run_dir / "train.json").is_file():
            run(brush_train_step(paths, item), log)
        run(score_step(paths, item, None), log)
    elif item.method in TRAINED:
        to_attic(paths, item, (run_dir,))
        variant: str = item.method.split("-")[1]
        args: tuple[str, ...] = ("hexels-train", "--dataset", item.dataset, "--scene", item.scene, "--variant", variant, "--steps", "7000")
        run(Step(name="hexels-train", env="hexels", args=args, cwd=paths.repo), log)
    else:
        to_attic(paths, item, (run_dir / "upstream-eval", run_dir / "upstream-eval-job"))
        variant = item.method.split("-")[1]
        args = ("hexels-render-published", "--dataset", item.dataset, "--scenes", item.scene, "--variants", variant)
        run(Step(name="hexels-render-published", env="hexels", args=args, cwd=paths.repo), log)
    reason: str | None = incomplete_reason(run_dir, item.method)
    if reason is not None:
        raise RuntimeError(f"steps finished but the run is incomplete: {reason}")


def record_failure(paths: BenchPaths, item: Item, error: str) -> None:
    record = FailureRecord(
        time=datetime.now().isoformat(timespec="seconds"),
        dataset=item.dataset,
        scene=item.scene,
        method=item.method,
        error=error,
        log=str(paths.run_dir(item) / "log.txt"),
    )
    paths.runs.mkdir(parents=True, exist_ok=True)
    with (paths.runs / "FAILURES.jsonl").open("a") as stream:
        stream.write(to_json(record) + "\n")


def run_queue(items: list[Item], paths: BenchPaths, run: Runner) -> QueueSummary:
    """Run every incomplete item under the GPU lock; a failure is logged and the queue continues."""
    summary = QueueSummary()
    for index, item in enumerate(items):
        label: str = f"[{index + 1}/{len(items)}] {item.dataset}/{item.scene}/{item.method}"
        if incomplete_reason(paths.run_dir(item), item.method) is None:
            summary.skipped.append(item)
            print(f"{label}: complete, skipped")
            continue
        started: float = time.perf_counter()
        with gpu_lock(paths.gpu_lock):
            reason: str | None = incomplete_reason(paths.run_dir(item), item.method)
            if reason is None:
                summary.skipped.append(item)
                print(f"{label}: completed by another job while waiting, skipped")
                continue
            print(f"{label}: running ({reason})", flush=True)
            try:
                execute(paths, item, run)
            except (StepFailed, OSError, RuntimeError, ValueError) as error:
                summary.failed.append(item)
                record_failure(paths, item, f"{type(error).__name__}: {error}")
                print(f"{label}: FAILED after {time.perf_counter() - started:.0f} s: {error}", flush=True)
                continue
        summary.done.append(item)
        print(f"{label}: done in {time.perf_counter() - started:.0f} s", flush=True)
    return summary


def pixi_runner(step: Step, log: Path) -> None:
    """Run a step through `pixi run --frozen`, echo its output, and append it to `log`."""
    command: list[str] = ["pixi", "run", "-e", step.env, "--frozen", *step.args]
    started: float = time.perf_counter()
    lines: list[str] = []
    with subprocess.Popen(command, cwd=step.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    seconds: float = time.perf_counter() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"== bench {step.name}: {shlex.join(command)} (cwd {step.cwd})\n")
        stream.writelines(lines)
        stream.write(f"== bench {step.name}: exit {process.returncode} after {seconds:.1f} s\n")
    if process.returncode != 0:
        raise StepFailed(f"{step.name} exited with {process.returncode}: {shlex.join(command)}")
