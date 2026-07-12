"""Resumable chunked ARKitScenes download, ingest, and shipping pipeline."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from _thread import LockType
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeAlias

import tyro
from tqdm import tqdm

from arkitscenes_download.download_dataset import human_bytes
from arkitscenes_download.fs import directory_size
from arkitscenes_download.ingest.batch import BatchSummary, run_batch
from arkitscenes_download.ingest.batch import Config as BatchConfig
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL, register_sequences
from arkitscenes_download.ingest.catalog import Config as CatalogConfig

ASSETS: tuple[str, ...] = (
    "mov",
    "annotation",
    "mesh",
    "lowres_wide.traj",
    "confidence",
    "lowres_depth",
    "lowres_wide_intrinsics",
    "ultrawide_intrinsics",
    "highres_depth",
)
FAILED_RAW_LIMIT_BYTES: int = 50 * 1024**3
SSH_DESTINATION_PATTERN: re.Pattern[str] = re.compile(r"^(?P<host>[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*):(?P<path>/.*)$")
VIDEO_ID_PATTERN: re.Pattern[str] = re.compile(r"^[0-9]+$")


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for the overnight pipeline."""

    data_dir: Path = Path("data")
    """Dataset root containing metadata and local staging."""
    destination: str = "data/published"
    """Local directory or ``user@host:/absolute/path`` publication target."""
    read_mount: Path | None = None
    """Optional locally readable view of an SSH destination."""
    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog URL used for registration."""
    chunk_size: int = 75
    """Number of sequences per pipeline chunk."""
    workers: int = 4
    """Concurrent ingestion subprocess count."""
    min_free_gb: int = 100
    """Minimum free staging space before starting a download."""
    max_chunks: int | None = None
    """Maximum chunks to process; zero performs no pipeline work."""
    video_ids: list[str] | None = None
    """Optional manifest subset, reordered by metadata order."""
    register: bool = True
    """Register verified NAS RRDs into the running catalog."""
    state_dir: Path = Path("data/pipeline-state")
    """Persistent resume and progress state directory."""
    retry_failed: bool = False
    """Re-queue previously failed sequences (clears their failed.txt entries so transient failures get another attempt)."""
    status: bool = False
    """Print current pipeline status and exit."""


@dataclass(frozen=True, slots=True)
class StatusSummary:
    """Computed persistent pipeline status."""

    done: int
    """Number of shipped and checksum-verified sequences."""
    failed: int
    """Number of terminally failed sequences."""
    remaining: int
    """Number of sequences with no terminal state."""
    failed_reasons: dict[str, str]
    """Failure reason keyed by sequence identifier."""
    destination_bytes: int | None
    """Bytes stored at the readable destination, or None without a read mount."""
    sequences_per_hour: float
    """Rolling throughput reported by chunk progress records."""
    eta_seconds: float | None
    """Estimated time to finish remaining sequences."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Result of sequentially downloading one chunk."""

    successful_ids: list[str]
    """Sequence IDs whose downloader subprocess succeeded."""
    failures: dict[str, str]
    """Downloader failure reasons keyed by sequence ID."""
    safety_stop: bool
    """Whether cumulative failed raw staging exceeded the limit."""


Runner: TypeAlias = Callable[..., subprocess.CompletedProcess[str]]


class DiskUsage(Protocol):
    """Free-space portion of a disk usage result."""

    @property
    def free(self) -> int:
        """Available bytes."""
        ...


@dataclass(frozen=True, slots=True)
class Destination:
    """Publication transport paired with its verification strategy."""

    value: str
    """Original local or SSH destination specification."""
    local_root: Path | None
    """Local destination root, or None for SSH transport."""
    ssh_host: str | None
    """SSH host specification, or None for local transport."""
    remote_root: str | None
    """Absolute remote root, or None for local transport."""
    read_mount: Path | None
    """Locally readable view used for verification and registration."""

    @classmethod
    def parse(cls, value: str, read_mount: Path | None = None) -> Destination:
        """Parse a local path or ``user@host:/absolute/path`` destination."""
        match: re.Match[str] | None = SSH_DESTINATION_PATTERN.fullmatch(value)
        if match is not None:
            return cls(value, None, match.group("host"), match.group("path"), read_mount)
        local_root: Path = Path(value)
        return cls(value, local_root, None, None, local_root if read_mount is None else read_mount)

    @property
    def registration_root(self) -> Path | None:
        """Return the readable root from which catalog layers can register."""
        return self.read_mount

    def ship_and_verify(self, source: Path, runner: Runner = subprocess.run) -> bool:
        """Publish one sequence and verify its complete file tree."""
        if self.local_root is not None:
            landed: Path = self.local_root / source.name
            shutil.copytree(source, landed, dirs_exist_ok=True)
            return bool(_tree_sha256(source)) and _tree_sha256(source) == _tree_sha256(landed)
        assert self.ssh_host is not None and self.remote_root is not None
        remote_root_quoted: str = shlex.quote(self.remote_root)
        with subprocess.Popen(["tar", "-C", str(source.parent), "-cf", "-", source.name], stdout=subprocess.PIPE) as archive:
            ship: subprocess.CompletedProcess[str] = runner(
                ["ssh", self.ssh_host, f"mkdir -p {remote_root_quoted} && tar -C {remote_root_quoted} -xf -"],
                stdin=archive.stdout,
                capture_output=True,
                text=True,
                check=False,
            )
            if archive.stdout is not None:
                archive.stdout.close()
            archive_status: int = archive.wait()
        if ship.returncode != 0 or archive_status != 0:
            tqdm.write(f"tar-ssh ship failed for {source.name}: {ship.stderr.strip()[-200:]}")
            return False
        if self.read_mount is not None:
            landed = self.read_mount / source.name
            return bool(_tree_sha256(source)) and _tree_sha256(source) == _tree_sha256(landed)
        verify: subprocess.CompletedProcess[str] = runner(
            ["ssh", self.ssh_host, f"cd {shlex.quote(self.remote_root + '/' + source.name)} && sha256sum -- *"],
            capture_output=True,
            text=True,
            check=False,
        )
        if verify.returncode != 0:
            return False
        remote_digests: dict[str, str] = {}
        for line in verify.stdout.splitlines():
            digest, separator, name = line.partition("  ")
            if separator:
                remote_digests[name] = digest
        return bool(remote_digests) and remote_digests == _tree_sha256(source)


def _read_id_lines(path: Path) -> set[str]:
    """Read the first tab-delimited field from a state file."""
    if not path.is_file():
        return set()
    return {line.split("\t", maxsplit=1)[0] for line in path.read_text().splitlines() if line.strip()}


def requeue_failed(state_dir: Path) -> set[str]:
    """Clear failed.txt and return the ids so a later manifest re-attempts them.

    Failures are recorded pessimistically (a CDN blip or one bad ingest marks a
    sequence terminal), so re-queuing is the recovery path after transient
    outages; genuinely broken sequences simply fail again and re-record.
    """
    failed_path: Path = state_dir / "failed.txt"
    requeued: set[str] = _read_id_lines(failed_path)
    if requeued:
        failed_path.unlink()
    return requeued


def _read_failed(path: Path) -> dict[str, str]:
    """Read failure reasons from the failure state file."""
    if not path.is_file():
        return {}
    failures: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields: list[str] = line.split("\t", maxsplit=1)
        if fields[0]:
            failures[fields[0]] = fields[1] if len(fields) == 2 else "unknown failure"
    return failures


def _append_line(path: Path, line: str, lock: LockType | None = None) -> None:
    """Durably append one flushed state record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if lock is None:
        with path.open("a") as output:
            output.write(f"{line}\n")
            output.flush()
            os.fsync(output.fileno())
        return
    with lock:
        _append_line(path, line)


def build_manifest(
    metadata_path: Path,
    done_ids: set[str],
    failed_ids: set[str],
    chunk_size: int,
    max_chunks: int | None,
    video_ids: list[str] | None,
) -> list[list[str]]:
    """Build terminal-state-filtered chunks in stable split order."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    with metadata_path.open(newline="") as metadata_file:
        rows: list[dict[str, str]] = list(csv.DictReader(metadata_file))
    ordered_rows: list[dict[str, str]] = sorted(rows, key=lambda row: 0 if row["fold"] == "Training" else 1)
    known_ids: set[str] = {row["video_id"] for row in ordered_rows}
    if video_ids is None:
        selected_ids: list[str] = [row["video_id"] for row in ordered_rows]
    else:
        unknown_ids: list[str] = [video_id for video_id in video_ids if video_id not in known_ids]
        if unknown_ids:
            raise ValueError(f"video ids absent from metadata: {unknown_ids}")
        selected_set: set[str] = set(video_ids)
        selected_ids = [row["video_id"] for row in ordered_rows if row["video_id"] in selected_set]
    malformed_ids: list[str] = [video_id for video_id in selected_ids if not VIDEO_ID_PATTERN.fullmatch(video_id)]
    if malformed_ids:
        raise ValueError(f"video ids must be numeric (they become filesystem and remote-shell path segments): {malformed_ids[:5]}")
    pending_ids: list[str] = [video_id for video_id in selected_ids if video_id not in done_ids and video_id not in failed_ids]
    chunks: list[list[str]] = [pending_ids[index : index + chunk_size] for index in range(0, len(pending_ids), chunk_size)]
    return chunks if max_chunks is None else chunks[:max_chunks]


def wait_for_disk_space(
    data_dir: Path,
    min_free_bytes: int,
    disk_usage: Callable[[Path], DiskUsage] = shutil.disk_usage,
    sleep: Callable[[float], None] = time.sleep,
    write: Callable[[str], None] = tqdm.write,
) -> None:
    """Wait until staging has at least the configured free byte count."""
    notified: bool = False
    while disk_usage(data_dir).free < min_free_bytes:
        if not notified:
            write(f"disk guard: less than {min_free_bytes / 1024**3:.1f} GiB free under {data_dir}; waiting")
            notified = True
        sleep(30.0)


def _sha256_of(path: Path) -> str:
    """Stream one file's sha256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> dict[str, str]:
    """Map relative file names to sha256 digests."""
    return {str(item.relative_to(root)): _sha256_of(item) for item in sorted(root.rglob("*")) if item.is_file()}


def ship_verify_and_cleanup(
    local_rrd: Path,
    local_raw: Path,
    destination: Destination,
    runner: Runner = subprocess.run,
    before_cleanup: Callable[[], None] | None = None,
) -> bool:
    """Publish, verify, persist completion, and remove local staging."""
    if not destination.ship_and_verify(local_rrd, runner):
        tqdm.write(f"sha256 verification failed for {local_rrd.name}")
        return False
    if before_cleanup is not None:
        before_cleanup()
    shutil.rmtree(local_rrd)
    shutil.rmtree(local_raw)
    return True


def compute_state_counts(metadata_path: Path, state_dir: Path) -> tuple[int, int, int, dict[str, str]]:
    """Compute terminal counts using metadata and state files only."""
    with metadata_path.open(newline="") as metadata_file:
        total: int = sum(1 for _ in csv.DictReader(metadata_file))
    done_ids: set[str] = _read_id_lines(state_dir / "done.txt")
    failed_reasons: dict[str, str] = _read_failed(state_dir / "failed.txt")
    terminal_ids: set[str] = done_ids | set(failed_reasons)
    remaining: int = max(0, total - len(terminal_ids))
    return len(done_ids), len(failed_reasons), remaining, failed_reasons


def compute_status(metadata_path: Path, state_dir: Path, readable_destination: Path | None) -> StatusSummary:
    """Compute expensive destination size, rolling throughput, and ETA for status."""
    done, failed, remaining, failed_reasons = compute_state_counts(metadata_path, state_dir)
    destination_bytes: int | None = directory_size(readable_destination) if readable_destination is not None else None
    rates: list[float] = []
    progress_path: Path = state_dir / "progress.log"
    if progress_path.is_file():
        for line in progress_path.read_text().splitlines()[-10:]:
            fields: dict[str, str] = dict(field.split("=", maxsplit=1) for field in line.split("\t") if "=" in field)
            wall_seconds: float = float(fields.get("wall_seconds", "0"))
            finished: int = int(fields.get("ok", "0")) + int(fields.get("failed", "0"))
            if wall_seconds > 0.0:
                rates.append(finished / wall_seconds)
    rate: float = sum(rates) / len(rates) if rates else 0.0
    eta_seconds: float | None = remaining / rate if rate > 0.0 else None
    return StatusSummary(done, failed, remaining, failed_reasons, destination_bytes, rate * 3600.0, eta_seconds)


def _metadata_splits(metadata_path: Path) -> dict[str, str]:
    """Return split names keyed by sequence ID."""
    with metadata_path.open(newline="") as metadata_file:
        return {row["video_id"]: row["fold"] for row in csv.DictReader(metadata_file)}


def _failed_raw_size(data_dir: Path, splits: dict[str, str], failed_ids: set[str]) -> int:
    """Return cumulative bytes retained for terminally failed sequences."""
    return sum(directory_size(data_dir / "raw" / splits[video_id] / video_id) for video_id in failed_ids if video_id in splits)


def _download_chunk(config: Config, chunk: list[str], splits: dict[str, str], existing_failed: set[str]) -> DownloadResult:
    """Download a chunk sequentially with retry, disk guard, and safety stop."""
    successful_ids: list[str] = []
    failures: dict[str, str] = {}
    failed_raw_bytes: int = _failed_raw_size(config.data_dir, splits, existing_failed)
    with tqdm(total=len(chunk), position=1, desc="download", unit="sequence", dynamic_ncols=True, leave=False) as progress:
        for video_id in chunk:
            if failed_raw_bytes > FAILED_RAW_LIMIT_BYTES:
                return DownloadResult(successful_ids, failures, True)
            wait_for_disk_space(config.data_dir, config.min_free_gb * 1024**3)
            command: list[str] = [
                sys.executable,
                "-m",
                "arkitscenes_download",
                "--download-dir",
                str(config.data_dir),
                "--video-ids",
                video_id,
                "--assets",
                *ASSETS,
                "--no-include-point-clouds",
            ]
            error: str = "download failed"
            for attempt in range(2):
                completed: subprocess.CompletedProcess[str] = subprocess.run(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
                )
                if completed.returncode == 0:
                    successful_ids.append(video_id)
                    break
                error = f"download exit {completed.returncode}: {completed.stdout.strip()[-1000:]}"
                if attempt == 0:
                    tqdm.write(f"[{video_id}] download failed; retrying once")
            else:
                failures[video_id] = error
                if video_id in splits:
                    failed_raw_bytes += directory_size(config.data_dir / "raw" / splits[video_id] / video_id)
            progress.update()
    return DownloadResult(successful_ids, failures, False)


def _catalog_reachable(catalog_url: str) -> bool:
    """Whether the local Rerun server accepts TCP connections."""
    try:
        address: str = catalog_url.removeprefix("rerun+").removeprefix("http://")
        host, _, port_text = address.partition(":")
        with socket.create_connection((host, int(port_text)), timeout=2.0):
            return True
    except OSError:
        return False


def _format_duration(seconds: float | None) -> str:
    """Format a duration for compact status output."""
    if seconds is None:
        return "unknown"
    return f"{seconds / 3600.0:.1f}h"


def _print_status(status: StatusSummary) -> None:
    """Print the human-readable status report."""
    print(f"done: {status.done}")
    print(f"failed: {status.failed}")
    print(f"remaining: {status.remaining}")
    for video_id, reason in sorted(status.failed_reasons.items()):
        print(f"  {video_id}: {reason}")
    size: str = "unknown (no read mount)" if status.destination_bytes is None else human_bytes(status.destination_bytes)
    print(f"destination size: {size}")
    print(f"rolling throughput: {status.sequences_per_hour:.2f} sequences/hour")
    print(f"rolling ETA: {_format_duration(status.eta_seconds)}")


def _ship_and_register_chunk(
    config: Config,
    destination: Destination,
    successful_ingests: list[str],
    splits: dict[str, str],
    state_lock: LockType,
    overall: tqdm,
    done_ids: set[str],
    failed_reasons: dict[str, str],
    prior_failures: int,
    chunk_index: int,
    chunk_count: int,
    total_population: int,
    chunk_started: float,
) -> None:
    """Publish one ingested chunk, register it, and record progress."""
    shipped_ids: list[str] = []
    published_bytes: int = 0
    with tqdm(total=len(successful_ingests), position=3, desc="ship+verify", unit="sequence", dynamic_ncols=True, leave=False) as stage:
        for video_id in successful_ingests:
            local_rrd: Path = config.data_dir / "rrd" / video_id
            local_raw: Path = config.data_dir / "raw" / splits[video_id] / video_id
            sequence_bytes: int = directory_size(local_rrd)

            def mark_done(completed_video_id: str = video_id) -> None:
                """Persist completion before local staging is removed."""
                _append_line(config.state_dir / "done.txt", completed_video_id, state_lock)

            if ship_verify_and_cleanup(local_rrd, local_raw, destination, before_cleanup=mark_done):
                shipped_ids.append(video_id)
                published_bytes += sequence_bytes
                done_ids.add(video_id)
                overall.update()
            stage.update()
    if shipped_ids and config.register:
        registration_root: Path | None = destination.registration_root
        if registration_root is None:
            for video_id in shipped_ids:
                _append_line(config.state_dir / "unregistered.txt", video_id, state_lock)
        elif _catalog_reachable(config.catalog_url):
            try:
                register_sequences(CatalogConfig(rrd_dir=registration_root, catalog_url=config.catalog_url, video_ids=shipped_ids))
            except (OSError, RuntimeError) as error:
                tqdm.write(f"catalog registration failed; recording unregistered IDs: {error}")
                for video_id in shipped_ids:
                    _append_line(config.state_dir / "unregistered.txt", video_id, state_lock)
        else:
            tqdm.write(f"Rerun catalog at {config.catalog_url} is unreachable; shipped IDs recorded as unregistered")
            for video_id in shipped_ids:
                _append_line(config.state_dir / "unregistered.txt", video_id, state_lock)
    chunk_wall: float = time.perf_counter() - chunk_started
    finished_count: int = len(shipped_ids) + prior_failures
    rate: float = finished_count / chunk_wall if chunk_wall > 0.0 else 0.0
    remaining: int = max(0, total_population - len(done_ids) - len(failed_reasons))
    eta: float | None = remaining / rate if rate > 0.0 else None
    record: str = (
        f"{datetime.now(UTC).isoformat()}\tchunk={chunk_index + 1}/{chunk_count}\tok={len(shipped_ids)}"
        f"\tfailed={prior_failures}\twall_seconds={chunk_wall:.3f}"
        f"\tpublished_bytes={published_bytes}\teta_seconds={eta if eta is not None else 'unknown'}"
    )
    _append_line(config.state_dir / "progress.log", record, state_lock)
    overall.set_postfix(ok=len(done_ids), failed=len(failed_reasons), chunk=f"{chunk_index + 1}/{chunk_count}")
    tqdm.write(record)


def run_pipeline(config: Config) -> None:
    """Run the resumable overlapped pipeline until its manifest is exhausted."""
    metadata_path: Path = config.data_dir / "raw" / "metadata.csv"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    destination: Destination = Destination.parse(config.destination, config.read_mount)
    if config.status:
        status: StatusSummary = compute_status(metadata_path, config.state_dir, destination.read_mount)
        _print_status(status)
        return
    done_count, failed_count, remaining_count, _ = compute_state_counts(metadata_path, config.state_dir)
    if config.register and destination.registration_root is None:
        tqdm.write("destination has no read mount; shipped IDs will be recorded as unregistered")
    done_ids: set[str] = _read_id_lines(config.state_dir / "done.txt")
    failed_reasons: dict[str, str] = _read_failed(config.state_dir / "failed.txt")
    if config.retry_failed and failed_reasons:
        requeued: set[str] = requeue_failed(config.state_dir)
        tqdm.write(f"retry-failed: re-queued {len(requeued)} previously failed sequences")
        failed_reasons = {}
    chunks: list[list[str]] = build_manifest(metadata_path, done_ids, set(failed_reasons), config.chunk_size, config.max_chunks, config.video_ids)
    splits: dict[str, str] = _metadata_splits(metadata_path)
    total_population: int = done_count + failed_count + remaining_count
    state_lock: LockType = threading.Lock()
    stop_requested: bool = False
    with tqdm(
        total=total_population,
        initial=done_count + failed_count,
        position=0,
        desc="sequences",
        unit="sequence",
        dynamic_ncols=True,
    ) as overall:
        overall.set_postfix(ok=done_count, failed=failed_count, chunk=f"0/{len(chunks)}")
        if not chunks:
            return
        with (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="download") as download_pool,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="ship") as ship_pool,
        ):
            ship_future: Future[None] | None = None
            download_future: Future[DownloadResult] = download_pool.submit(_download_chunk, config, chunks[0], splits, set(failed_reasons))
            for chunk_index, _chunk in enumerate(chunks):
                chunk_started: float = time.perf_counter()
                try:
                    download_result: DownloadResult = download_future.result()
                except KeyboardInterrupt:
                    tqdm.write("interrupt requested; waiting for the in-flight download sequence")
                    stop_requested = True
                    download_result = download_future.result()
                for video_id, reason in download_result.failures.items():
                    _append_line(config.state_dir / "failed.txt", f"{video_id}\t{reason}", state_lock)
                    failed_reasons[video_id] = reason
                    overall.update()
                if download_result.safety_stop:
                    tqdm.write(f"failed raw staging exceeds {human_bytes(FAILED_RAW_LIMIT_BYTES)}; stopping before new downloads")
                    stop_requested = True
                next_future: Future[DownloadResult] | None = None
                if not stop_requested and chunk_index + 1 < len(chunks):
                    next_future = download_pool.submit(_download_chunk, config, chunks[chunk_index + 1], splits, set(failed_reasons))
                ingest_ids: list[str] = download_result.successful_ids
                summary: BatchSummary = run_batch(
                    BatchConfig(
                        video_ids=ingest_ids,
                        workers=config.workers,
                        data_dir=config.data_dir,
                        output=config.data_dir / "rrd",
                        skip_existing=True,
                        progress_position=2,
                    )
                )
                ingest_failures: set[str] = set(summary.failed_video_ids)
                for video_id in ingest_failures:
                    reason: str = "ingest failed"
                    _append_line(config.state_dir / "failed.txt", f"{video_id}\t{reason}", state_lock)
                    failed_reasons[video_id] = reason
                    overall.update()
                successful_ingests: list[str] = [video_id for video_id in ingest_ids if video_id not in ingest_failures]
                if ship_future is not None:
                    ship_future.result()
                ship_future = ship_pool.submit(
                    _ship_and_register_chunk,
                    config,
                    destination,
                    successful_ingests,
                    splits,
                    state_lock,
                    overall,
                    done_ids,
                    failed_reasons,
                    len(download_result.failures) + len(ingest_failures),
                    chunk_index,
                    len(chunks),
                    total_population,
                    chunk_started,
                )
                if stop_requested:
                    break
                if next_future is not None:
                    download_future = next_future
            if ship_future is not None:
                ship_future.result()


def main() -> None:
    """Parse CLI configuration and run the pipeline."""
    try:
        run_pipeline(tyro.cli(Config))
    except KeyboardInterrupt:
        tqdm.write("pipeline interrupted cleanly; persistent state is ready to resume")


if __name__ == "__main__":
    main()
