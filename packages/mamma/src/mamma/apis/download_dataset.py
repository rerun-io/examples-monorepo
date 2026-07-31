"""Download the public MAMMA dataset collections."""

import os
import re
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias
from urllib.parse import quote, urlencode

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from simplecv.print_utils import format_bytes

from mamma.apis.download_manifest import (
    DANCE_CAMERAS,
    DANCE_SEQUENCES,
    EVAL_CAMERAS,
    EVAL_DANCE_SEQUENCES,
    EVAL_EXTRA_SEQUENCES,
    EVAL_SINGLES_SEQUENCES,
    IPHONE_CAMERAS,
    IPHONE_INDOOR_SEQUENCES,
    IPHONE_OUTDOOR_SEQUENCES,
    MULTI_PEOPLE_CAMERAS,
    MULTI_PEOPLE_SEQUENCES,
)

_ERROR_MARKERS: tuple[bytes, ...] = (b"error: file not found.", b"<!doctype html", b"<html")
_MPI_DOWNLOAD_URL: str = "https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"
_DOWNLOAD_WORKERS: int = 4
_REQUEST_TIMEOUT_SECONDS: float = 120.0

CollectionName: TypeAlias = Literal["eval", "dance", "multi_people", "iphone"]
DownloadStatus: TypeAlias = Literal["downloaded", "skipped", "failed"]


@dataclass
class Config:
    """MAMMA dataset download settings."""

    output_root: Path = Path("/mnt/nas/datasets/mamma")
    """Directory under which collection-relative paths are stored."""
    eval: bool = True
    """Download eval ground truth, masks, markers, and CRF-16 videos."""
    dance: bool = True
    """Download metadata, predictions, and CRF-16 videos for all dance styles."""
    multi_people: bool = True
    """Download multi-person metadata, predictions, and CRF-16 videos."""
    iphone: bool = True
    """Download iPhone metadata, predictions, and H.265 CRF-16 videos."""
    synthetic: bool = True
    """Download the MammaSyn Hugging Face dataset snapshot."""


@dataclass(frozen=True, slots=True)
class DownloadPhase:
    """One independently tracked group of MPI files."""

    name: str
    """Human-readable phase name."""
    files: tuple[Path, ...]
    """Dataset-relative files in the phase."""


@dataclass(frozen=True, slots=True)
class MpiCredentials:
    """Credentials for the MAMMA MPI download server."""

    username: str
    """Registered MAMMA username."""
    password: str
    """Registered MAMMA password."""


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of one MPI file transfer."""

    relative_path: Path
    """Dataset-relative file path."""
    status: DownloadStatus
    """Whether the file was downloaded, skipped, or failed."""
    size_bytes: int
    """Valid local file size, or zero for a failed transfer."""
    error: str | None = None
    """Failure detail when status is failed."""


def credentials_from_environment() -> MpiCredentials:
    """Load required MPI credentials from the process environment."""
    username: str | None = os.environ.get("MAMMA_USERNAME")
    password: str | None = os.environ.get("MAMMA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "MAMMA_USERNAME and MAMMA_PASSWORD must be set. Register at https://mamma.is.tue.mpg.de/ and export both variables."
        )
    return MpiCredentials(username=username, password=password)


def _camera_paths(sequences_and_cameras: tuple[tuple[str, tuple[str, ...]], ...], directory: str, suffix: str) -> tuple[Path, ...]:
    """Build per-camera paths for sequences with possibly different rigs."""
    return tuple(Path(sequence) / directory / f"{camera}{suffix}" for sequence, cameras in sequences_and_cameras for camera in cameras)


def _calibration_paths(sequences_and_cameras: tuple[tuple[str, tuple[str, ...]], ...], directory: str) -> tuple[Path, ...]:
    """Build global plus per-camera npz paths under a calibration-style directory (``meta`` or ``gt``)."""
    return tuple(
        path
        for sequence, cameras in sequences_and_cameras
        for path in (Path(sequence) / directory / "global.npz", *[Path(sequence) / directory / f"{camera}.npz" for camera in cameras])
    )


def _prediction_paths(sequences_and_counts: tuple[tuple[str, int], ...]) -> tuple[Path, ...]:
    """Build zero-padded per-person prediction paths."""
    return tuple(Path(sequence) / "pred" / f"params_{index:02d}.npz" for sequence, count in sequences_and_counts for index in range(count))


def _dance_person_count(sequence: str) -> int:
    """Infer the dance subject count from the sequence naming convention.

    Sequence names embed one 5-digit subject id per dancer (solo captures like
    ``..._Improv_1_03685_1`` have one, couple captures like
    ``..._SugarPush_03688_03689_1`` have two). Verified against the full
    upstream corpus: the two-id pattern matches exactly the sequences that
    ship a ``params_01.npz``.
    """
    if re.search(r"_[0-9]{5}_[0-9]{5}(?:_[0-9]+){1,3}$", Path(sequence).name):
        return 2
    return 1


def _markerless_phases(
    collection: str, sequences_and_counts: tuple[tuple[str, int], ...], cameras: tuple[str, ...], video_directory: str
) -> tuple[DownloadPhase, ...]:
    """Build the meta/pred/videos phase triple shared by the markerless collections."""
    sequences_and_cameras: tuple[tuple[str, tuple[str, ...]], ...] = tuple((sequence, cameras) for sequence, _ in sequences_and_counts)
    return (
        DownloadPhase(f"{collection}:meta", _calibration_paths(sequences_and_cameras, "meta")),
        DownloadPhase(f"{collection}:pred", _prediction_paths(sequences_and_counts)),
        DownloadPhase(f"{collection}:{video_directory}", _camera_paths(sequences_and_cameras, video_directory, ".mp4")),
    )


def build_collection_phases(collection: CollectionName) -> tuple[DownloadPhase, ...]:
    """Expand a collection into the asset phases defined by its reference script."""
    if collection == "eval":
        sequences_and_cameras: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
            (sequence, EVAL_CAMERAS[sequence.split("/", maxsplit=1)[0]])
            for sequence in (*EVAL_SINGLES_SEQUENCES, *EVAL_EXTRA_SEQUENCES, *EVAL_DANCE_SEQUENCES)
        )
        marker_files: tuple[Path, ...] = tuple(
            Path(sequence) / "markers" / filename
            for sequence in EVAL_EXTRA_SEQUENCES
            for filename in ("vicon_m37.npy", "baseline_m37.npy", "labels_m37.npy")
        )
        return (
            DownloadPhase("eval:gt", _calibration_paths(sequences_and_cameras, "gt")),
            DownloadPhase("eval:masks", _camera_paths(sequences_and_cameras, "masks", "_masks.tar")),
            DownloadPhase("eval:markers", marker_files),
            DownloadPhase("eval:videos_crf16", _camera_paths(sequences_and_cameras, "videos_crf16", ".mp4")),
        )

    if collection == "dance":
        dance_counts: tuple[tuple[str, int], ...] = tuple(
            (sequence, _dance_person_count(sequence)) for style_sequences in DANCE_SEQUENCES.values() for sequence in style_sequences
        )
        return _markerless_phases("dance", dance_counts, DANCE_CAMERAS, "videos_crf16")

    if collection == "multi_people":
        multi_counts: tuple[tuple[str, int], ...] = tuple(
            (sequence, person_count) for person_count, sequences in MULTI_PEOPLE_SEQUENCES.items() for sequence in sequences
        )
        return _markerless_phases("multi_people", multi_counts, MULTI_PEOPLE_CAMERAS, "videos_crf16")

    if collection == "iphone":
        iphone_counts: tuple[tuple[str, int], ...] = tuple(
            (sequence, person_count) for person_count, sequence in (*IPHONE_INDOOR_SEQUENCES, *IPHONE_OUTDOOR_SEQUENCES)
        )
        return _markerless_phases("iphone", iphone_counts, IPHONE_CAMERAS, "videos")

    raise ValueError(f"unsupported MAMMA collection: {collection}")


def download_url(relative_path: Path) -> str:
    """Build the MPI download URL for a dataset-relative path."""
    remote_path: str = quote(f"datasets/{relative_path.as_posix()}", safe="/")
    return f"{_MPI_DOWNLOAD_URL}&sfile={remote_path}"


def valid_download_size(path: Path) -> int | None:
    """Return the size of a non-empty, non-HTML dataset file, or None when invalid.

    One open + fstat covers existence, size, and the prefix check without
    stat/open races. NFS mounts can throw transient EACCES/ESTALE even for
    world-readable files; treat those as "not valid" (re-download) instead
    of letting one glitch kill the whole run.
    """
    try:
        with path.open("rb") as file:
            size: int = os.fstat(file.fileno()).st_size
            prefix: bytes = file.read(256).lower()
    except OSError:
        return None
    if size == 0 or any(marker in prefix for marker in _ERROR_MARKERS):
        return None
    return size


def _download_file(output_root: Path, relative_path: Path, wget: str, post_data: str) -> DownloadResult:
    """Download one MPI file or return its valid existing copy."""
    destination: Path = output_root / relative_path
    existing_size: int | None = valid_download_size(destination)
    if existing_size is not None:
        return DownloadResult(relative_path=relative_path, status="skipped", size_bytes=existing_size)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Transfer into a .part file so an interrupted run never leaves a truncated
    # file at the final path (the prefix-only validator would skip it forever);
    # a kept .part also lets wget --continue resume it on the next run.
    partial: Path = destination.with_name(destination.name + ".part")
    command: list[str] = [
        wget,
        "--post-data",
        post_data,
        download_url(relative_path),
        "-O",
        str(partial),
        "--no-check-certificate",
        "--continue",
        "--quiet",
        f"--timeout={_REQUEST_TIMEOUT_SECONDS}",
        "--tries=2",
    ]
    result: subprocess.CompletedProcess[bytes] = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        return DownloadResult(
            relative_path=relative_path, status="failed", size_bytes=0, error=f"wget exit {result.returncode}: {result.stderr.decode()[:120]}"
        )

    size: int | None = valid_download_size(partial)
    if size is None:
        partial.unlink(missing_ok=True)
        return DownloadResult(relative_path=relative_path, status="failed", size_bytes=0, error="server returned an empty or HTML response")
    partial.replace(destination)
    return DownloadResult(relative_path=relative_path, status="downloaded", size_bytes=size)


def _progress(console: Console) -> Progress:
    """Create the shared files-and-bytes phase progress display."""
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("files"),
        TextColumn("• {task.fields[bytes_text]}"),
        TimeElapsedColumn(),
        console=console,
    )


def _download_mpi_phases(output_root: Path, phases: tuple[DownloadPhase, ...], credentials: MpiCredentials, console: Console) -> int:
    """Download all MPI phases and return the number of failed files."""
    # download.is.tue.mpg.de serves the HTML landing page to python TLS clients
    # (urllib gets HTML with the identical POST body and headers that wget sends,
    # including a spoofed wget User-Agent), so the fetch itself must go through
    # wget like the upstream scripts.
    wget: str | None = shutil.which("wget")
    if wget is None:
        raise RuntimeError("wget not found on PATH; it is required for MPI transfers")
    post_data: str = urlencode({"username": credentials.username, "password": credentials.password})
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    with _progress(console) as progress:
        for phase in phases:
            task_id: TaskID = progress.add_task(phase.name, total=len(phase.files), bytes_text="0 B")
            phase_bytes: int = 0
            with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as executor:
                futures: list[Future[DownloadResult]] = [
                    executor.submit(_download_file, output_root, relative_path, wget, post_data) for relative_path in phase.files
                ]
                for future in as_completed(futures):
                    result: DownloadResult = future.result()
                    phase_bytes += result.size_bytes
                    progress.update(task_id, advance=1, bytes_text=format_bytes(phase_bytes))
                    if result.status == "downloaded":
                        downloaded += 1
                    elif result.status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                        console.print(f"[red]FAIL[/red] {result.relative_path}: {result.error}")
    console.print(f"MPI files: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    return failed


def _download_synthetic(output_root: Path, console: Console) -> None:
    """Download MammaSyn with NFS-safe Hugging Face locking."""
    import filelock

    # This dataset lives on NFSv3 with local_lock=none, where flock can hang.
    # Patch FileLock before importing huggingface_hub so its cache uses soft locks.
    filelock.FileLock = filelock.SoftFileLock
    from huggingface_hub import snapshot_download

    local_dir: Path = output_root / "MammaSyn"
    local_dir.mkdir(parents=True, exist_ok=True)
    # Only sweep huggingface_hub's own lock dir — a dataset-wide rglob walks the
    # whole multi-TB tree and could remove locks a concurrent snapshot still holds.
    for lock_path in (local_dir / ".cache" / "huggingface").rglob("*.lock"):
        lock_path.unlink(missing_ok=True)

    with _progress(console) as progress:
        task_id: TaskID = progress.add_task("synthetic:snapshot", total=1, bytes_text="0 B")
        snapshot_download(
            "Intelligent-Systems/MammaSyn",
            repo_type="dataset",
            local_dir=local_dir,
            max_workers=8,
        )
        progress.update(task_id, completed=1, bytes_text="complete")


def main(config: Config) -> int:
    """Download the selected MAMMA collections."""
    console: Console = Console()
    selected_collections: tuple[CollectionName, ...] = tuple(
        collection
        for collection, selected in (
            ("eval", config.eval),
            ("dance", config.dance),
            ("multi_people", config.multi_people),
            ("iphone", config.iphone),
        )
        if selected
    )
    phases: tuple[DownloadPhase, ...] = tuple(
        phase for collection in selected_collections for phase in build_collection_phases(collection)
    )
    failed: int = 0
    if phases:
        try:
            credentials: MpiCredentials = credentials_from_environment()
            failed = _download_mpi_phases(config.output_root, phases, credentials, console)
        except RuntimeError as error:
            console.print(f"[red]Error:[/red] {error}")
            return 2

    if config.synthetic:
        _download_synthetic(config.output_root, console)
    if not phases and not config.synthetic:
        console.print("No collections selected.")
    return 1 if failed else 0
