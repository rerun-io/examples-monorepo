"""Run the Hexels benchmark queue: every (dataset, scene, method) run directory of docs/run-layout.md."""

from dataclasses import dataclass, field
from pathlib import Path

from hexels_bench.bench import Item, QueueSummary, build_queue, pixi_runner, run_queue
from hexels_bench.layout import Dataset, Roots
from hexels_bench.methods import BENCHMARK_METHODS, Method


@dataclass(frozen=True, slots=True)
class Config:
    """Run incomplete benchmark items one by one under the GPU lock; complete items are skipped."""

    datasets: tuple[Dataset, ...] = ("dtu", "mipnerf360", "tnt")
    """Datasets in queue order."""
    scenes: tuple[str, ...] = ()
    """Scene filter (upstream names, e.g. garden scan24 Truck); empty runs every protocol scene."""
    methods: tuple[Method, ...] = BENCHMARK_METHODS
    """Methods per scene, in this order."""
    tail_methods: tuple[Method, ...] = ()
    """Methods queued afterwards for every protocol scene of the datasets (e.g. brush-7k to score finished Brush runs)."""
    roots: Roots = field(default_factory=Roots)
    """Data, run and Brush-prepared roots."""
    gpu_lock: Path | None = None
    """flock file shared by every GPU-heavy job on the host; None takes `gpu.lock` beside the runs root."""


def queue_for(config: Config) -> list[Item]:
    """The selected scenes with every method, then the tail methods on the remaining protocol scenes."""
    queue: list[Item] = build_queue(config.datasets, config.scenes, config.methods)
    return queue + [item for item in build_queue(config.datasets, methods=config.tail_methods) if item not in queue]


def main(config: Config) -> None:
    queue: list[Item] = queue_for(config)
    lock: Path = config.gpu_lock if config.gpu_lock is not None else config.roots.runs.parent / "gpu.lock"
    print(f"{len(queue)} items; runs {config.roots.runs}; lock {lock}", flush=True)
    summary: QueueSummary = run_queue(queue, config.roots, lock, pixi_runner)
    print(f"done {len(summary.done)}, skipped {len(summary.skipped)}, failed {len(summary.failed)}")
    for item in summary.failed:
        print(f"FAILED {item.dataset}/{item.scene}/{item.method} (see {config.roots.runs / 'FAILURES.jsonl'})")
    if summary.failed:
        raise SystemExit(1)
