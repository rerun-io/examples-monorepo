"""The Hexels 34-scene benchmark queue (~/hexels-work/contracts/run-layout.md).

An item is one (dataset, scene, method) run directory. Each method runs through its tool in its
own Pixi environment: the WGSL `gsplat render` in `gsplat-rust-renderer`; the Brush protocol runner,
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

from hexels_bench.cameras import CamerasFile
from hexels_bench.layout import GSPLAT_RELEASE, PROTOCOL_SCENES, WORKSPACE, Dataset, Roots, read_json
from hexels_bench.methods import METHODS, Method, MethodSpec
from hexels_bench.metrics import MetricsFile, TrainRecord

Env: TypeAlias = Literal["gsplat-rust-renderer", "hexels"]


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
class Step:
    """One command of an item, run as `pixi run -e <env> --frozen <args>` from the workspace root."""

    name: str
    """Short step name for logs."""
    env: Env
    """Pixi environment."""
    args: tuple[str, ...]
    """Command and arguments (a Pixi task name or an executable)."""


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


def build_queue(datasets: tuple[Dataset, ...], scenes: tuple[str, ...] = (), methods: tuple[Method, ...] = tuple(METHODS)) -> list[Item]:
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


def incomplete_reason(run_dir: Path, method: Method) -> str | None:
    """None when the run directory is a finished, scored run of `method`; otherwise what is missing."""
    spec: MethodSpec = METHODS[method]
    if not (run_dir / "metrics.json").is_file():
        return "no metrics.json"
    if not (run_dir / "cameras.json").is_file():
        return "no cameras.json"
    try:
        metrics: MetricsFile = read_json(MetricsFile, run_dir / "metrics.json")
        names: list[str] = [camera.name for camera in read_json(CamerasFile, run_dir / "cameras.json").cameras]
        if spec.trains:
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
    if spec.renders_from is not None and metrics.fps is None:
        return "no FPS"
    if spec.published and spec.renders_from is None and not (run_dir / "upstream_float_metrics.json").is_file():
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


def to_attic(roots: Roots, item: Item, partials: tuple[Path, ...]) -> None:
    """Move partial outputs of an item out of the run layout into one `runs/_attic/` directory, kept for inspection."""
    existing: list[Path] = [path for path in partials if path.exists()]
    if not existing:
        return
    stamp: str = datetime.now().strftime("%Y%m%dT%H%M%S-%f")
    attic: Path = roots.runs / "_attic" / f"{item.dataset}-{item.scene}-{item.method}-{stamp}"
    for path in existing:
        target: Path = attic if path == roots.run_dir(item.dataset, item.scene, item.method) else attic / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)
        print(f"moved partial {path} -> {target}")


def score_step(item: Item, fps: float | None) -> Step:
    fps_args: tuple[str, ...] = () if fps is None else ("--fps", repr(fps))
    return Step(name="score", env="hexels", args=("hexels-score", "--dataset", item.dataset, "--scene", item.scene, "--method", item.method, *fps_args))


def brush_train_step(roots: Roots, item: Item) -> Step:
    """The Brush protocol runner for one scene, without --gpu-lock (the conductor already holds it)."""
    args: tuple[str, ...] = (
        "python", str(WORKSPACE / "packages" / "hexels" / "tools" / "train_brush.py"), "--dataset", item.dataset, "--scenes", item.scene, "--method", item.method,
        "--steps", str(METHODS[item.method].steps), "--roots.data", str(roots.data), "--roots.runs", str(roots.runs),
        "--roots.brush-prepared", str(roots.brush_prepared),
    )  # fmt: skip
    return Step(name="brush-train", env="hexels", args=args)


def wgsl_render_step(run_dir: Path) -> Step:
    args: tuple[str, ...] = (
        str(GSPLAT_RELEASE / "gsplat"), "render", "--ply", str(run_dir / "model.ply"), "--camera", str(run_dir / "cameras.json"),
        "--output-dir", str(run_dir / "renders" / "test"), "--background", "0,0,0", "--png16", "--fps", str(run_dir / "fps.json"),
    )  # fmt: skip
    return Step(name="wgsl-render", env="gsplat-rust-renderer", args=args)


def stage_wgsl(roots: Roots, item: Item, source_method: Method) -> Path:
    """Link the source run's model.ply and copy its cameras.json into the WGSL run directory."""
    source: Path = roots.run_dir(item.dataset, item.scene, source_method)
    if not (source / "model.ply").is_file() or not (source / "cameras.json").is_file():
        raise RuntimeError(f"source run {source} has no model.ply/cameras.json yet")
    if METHODS[source_method].trains and not (source / "train.json").is_file():
        raise RuntimeError(f"source run {source} has not finished training (no train.json)")
    run_dir: Path = roots.run_dir(item.dataset, item.scene, item.method)
    run_dir.mkdir(parents=True, exist_ok=True)
    link: Path = run_dir / "model.ply"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(Path("..") / source.name / "model.ply")
    shutil.copyfile(source / "cameras.json", run_dir / "cameras.json")
    return run_dir


def execute(roots: Roots, item: Item, run: Runner) -> None:
    """Run one incomplete item to completion; raises when a step fails or the result is still incomplete."""
    spec: MethodSpec = METHODS[item.method]
    run_dir: Path = roots.run_dir(item.dataset, item.scene, item.method)
    log: Path = run_dir / "log.txt"
    if spec.renders_from is not None:
        stage_wgsl(roots, item, spec.renders_from)
        run(wgsl_render_step(run_dir), log)
        run(score_step(item, read_json(FpsReport, run_dir / "fps.json").median_fps), log)
    elif spec.variant is None:
        if not (run_dir / "train.json").is_file():
            run(brush_train_step(roots, item), log)
        run(score_step(item, None), log)
    elif spec.trains:
        to_attic(roots, item, (run_dir,))
        args: tuple[str, ...] = ("hexels-train", "--dataset", item.dataset, "--scene", item.scene, "--variant", spec.variant, "--steps", str(spec.steps))
        run(Step(name="hexels-train", env="hexels", args=args), log)
    else:
        to_attic(roots, item, (run_dir / "upstream-eval", run_dir / "upstream-eval-job"))
        args = ("hexels-render-published", "--dataset", item.dataset, "--scenes", item.scene, "--variants", spec.variant)
        run(Step(name="hexels-render-published", env="hexels", args=args), log)
    reason: str | None = incomplete_reason(run_dir, item.method)
    if reason is not None:
        raise RuntimeError(f"steps finished but the run is incomplete: {reason}")


def record_failure(roots: Roots, item: Item, error: str) -> None:
    record = FailureRecord(
        time=datetime.now().isoformat(timespec="seconds"),
        dataset=item.dataset,
        scene=item.scene,
        method=item.method,
        error=error,
        log=str(roots.run_dir(item.dataset, item.scene, item.method) / "log.txt"),
    )
    roots.runs.mkdir(parents=True, exist_ok=True)
    with (roots.runs / "FAILURES.jsonl").open("a") as stream:
        stream.write(to_json(record) + "\n")


def run_queue(items: list[Item], roots: Roots, lock: Path, run: Runner) -> QueueSummary:
    """Run every incomplete item under the GPU lock; a failure is logged and the queue continues."""
    summary = QueueSummary()
    for index, item in enumerate(items):
        label: str = f"[{index + 1}/{len(items)}] {item.dataset}/{item.scene}/{item.method}"
        run_dir: Path = roots.run_dir(item.dataset, item.scene, item.method)
        if incomplete_reason(run_dir, item.method) is None:
            summary.skipped.append(item)
            print(f"{label}: complete, skipped")
            continue
        started: float = time.perf_counter()
        with gpu_lock(lock):
            reason: str | None = incomplete_reason(run_dir, item.method)
            if reason is None:
                summary.skipped.append(item)
                print(f"{label}: completed by another job while waiting, skipped")
                continue
            print(f"{label}: running ({reason})", flush=True)
            try:
                execute(roots, item, run)
            except (StepFailed, OSError, RuntimeError, ValueError) as error:
                summary.failed.append(item)
                record_failure(roots, item, f"{type(error).__name__}: {error}")
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
    with subprocess.Popen(command, cwd=WORKSPACE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
    seconds: float = time.perf_counter() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"== bench {step.name}: {shlex.join(command)} (cwd {WORKSPACE})\n")
        stream.writelines(lines)
        stream.write(f"== bench {step.name}: exit {process.returncode} after {seconds:.1f} s\n")
    if process.returncode != 0:
        raise StepFailed(f"{step.name} exited with {process.returncode}: {shlex.join(command)}")
