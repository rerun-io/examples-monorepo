"""Concurrent multi-sequence ARKitScenes ingestion."""

import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from arkitscenes_download.ingest.layers import LAYER_NAMES

# Package root is parents[2] from arkitscenes_download/ingest/batch.py. Fail at
# import time rather than as thousands of per-sequence subprocess failures.
INGEST_TOOL: Path = Path(__file__).resolve().parents[2] / "tools" / "apps" / "ingest_sequence.py"
assert INGEST_TOOL.is_file(), f"ingest tool shim missing: {INGEST_TOOL}"


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """Completed subprocess result for one sequence."""

    video_id: str
    """ARKitScenes sequence identifier."""
    wall_seconds: float
    """Elapsed wall time for this sequence."""
    ok: bool
    """Whether the subprocess exited successfully."""
    output: str
    """Combined standard output and error from the subprocess."""


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Aggregate timing and failure information for a batch."""

    total_wall_seconds: float
    """Elapsed wall time for the entire batch."""
    sum_sequence_wall_seconds: float
    """Sum of each sequence's elapsed wall time."""
    effective_speedup: float
    """Ratio of summed sequence walls to total batch wall time."""
    failed_video_ids: list[str]
    """Identifiers of sequences whose subprocess failed."""


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for concurrent ARKitScenes ingestion."""

    video_ids: list[str] | None = None
    """Sequence identifiers; autodiscover both raw splits when omitted."""
    workers: int = 4
    """Maximum number of concurrent sequence subprocesses."""
    data_dir: Path = Path("data")
    """Dataset root containing raw/Training and raw/Validation."""
    output: Path = Path("data/rrd")
    """Directory in which to write RRD files."""
    keep_transcode_cache: bool = False
    """Keep prepared MP4 tracks after each ingest."""
    skip_existing: bool = False
    """Skip sequence identifiers only when all eight required layer RRDs exist."""


def discover_video_ids(data_dir: Path) -> list[str]:
    """Return sorted unique sequence directory names from both raw splits."""
    video_ids: set[str] = set()
    for split in ("Training", "Validation"):
        split_dir: Path = data_dir / "raw" / split
        if split_dir.is_dir():
            video_ids.update(path.name for path in split_dir.iterdir() if path.is_dir())
    return sorted(video_ids)


def has_all_layers(output: Path, video_id: str) -> bool:
    """Whether all required layer files have been published for a sequence."""
    return all((output / video_id / f"{layer_name}.rrd").is_file() for layer_name in LAYER_NAMES)


def summarize(results: list[SequenceResult], total_wall_seconds: float) -> BatchSummary:
    """Compute aggregate batch timing and failure information."""
    sum_sequence_wall_seconds: float = sum(result.wall_seconds for result in results)
    effective_speedup: float = sum_sequence_wall_seconds / total_wall_seconds if total_wall_seconds > 0.0 else 0.0
    failed_video_ids: list[str] = [result.video_id for result in results if not result.ok]
    return BatchSummary(total_wall_seconds, sum_sequence_wall_seconds, effective_speedup, failed_video_ids)


def _run_sequence(video_id: str, config: Config) -> SequenceResult:
    """Run one single-sequence CLI subprocess and capture its combined output."""
    command: list[str] = [
        sys.executable,
        str(INGEST_TOOL),
        "--video-id",
        video_id,
        "--data-dir",
        str(config.data_dir),
        "--output",
        str(config.output),
    ]
    if config.keep_transcode_cache:
        command.append("--keep-transcode-cache")
    started: float = time.perf_counter()
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        wall_seconds: float = time.perf_counter() - started
        return SequenceResult(video_id, wall_seconds, False, str(error))
    wall_seconds = time.perf_counter() - started
    return SequenceResult(video_id, wall_seconds, completed.returncode == 0, completed.stdout)


def _print_prefixed_output(result: SequenceResult, progress: Progress) -> None:
    """Print captured subprocess output with its sequence identifier."""
    for line in result.output.splitlines():
        progress.console.print(f"[{result.video_id}] {line}", markup=False)


def _print_summary(results: list[SequenceResult], summary: BatchSummary, console: Console) -> None:
    """Print the per-sequence and aggregate batch summary."""
    table: Table = Table(title="Ingest summary")
    table.add_column("video_id")
    table.add_column("wall seconds", justify="right")
    table.add_column("status")
    for result in sorted(results, key=lambda item: item.video_id):
        status: str = "ok" if result.ok else "FAILED"
        table.add_row(result.video_id, f"{result.wall_seconds:.2f}", status)
    console.print(table)
    console.print(f"Total wall time: {summary.total_wall_seconds:.2f}s")
    console.print(f"Sum of sequence walls: {summary.sum_sequence_wall_seconds:.2f}s")
    console.print(f"Effective speedup: {summary.effective_speedup:.2f}x")


def run_batch(config: Config, progress: Progress) -> BatchSummary:
    """Run selected sequences concurrently, reporting every completed job.

    The caller owns the live progress region (rich Live regions cannot nest).
    """
    if config.workers < 1:
        raise ValueError("workers must be at least 1")
    selected_video_ids: list[str] = discover_video_ids(config.data_dir) if config.video_ids is None else list(dict.fromkeys(config.video_ids))
    video_ids: list[str] = []
    for video_id in selected_video_ids:
        if config.skip_existing and has_all_layers(config.output, video_id):
            progress.console.print(f"[{video_id}] skipped: output already exists", markup=False)
        else:
            video_ids.append(video_id)

    # Capacity policy: 4 workers × 8 depth threads can momentarily request all
    # 32 cores on this box; that is accepted, and workers remains the throttle.
    started: float = time.perf_counter()
    results: list[SequenceResult] = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures: dict[Future[SequenceResult], str] = {pool.submit(_run_sequence, video_id, config): video_id for video_id in video_ids}
        ok_count: int = 0
        failed_count: int = 0
        task_id: TaskID = progress.add_task("ingest (ok=0 failed=0)", total=len(futures))
        for future in as_completed(futures):
            result: SequenceResult = future.result()
            results.append(result)
            ok_count += int(result.ok)
            failed_count += int(not result.ok)
            _print_prefixed_output(result, progress)
            progress.advance(task_id)
            progress.update(task_id, description=f"ingest (ok={ok_count} failed={failed_count})")
        progress.remove_task(task_id)
    total_wall_seconds: float = time.perf_counter() - started
    summary: BatchSummary = summarize(results, total_wall_seconds)
    _print_summary(results, summary, progress.console)
    return summary


def main(config: Config) -> None:
    """Run a batch and exit unsuccessfully if any job failed."""
    with Progress(console=Console(markup=False)) as progress:
        summary: BatchSummary = run_batch(config, progress)
    raise SystemExit(1 if summary.failed_video_ids else 0)
