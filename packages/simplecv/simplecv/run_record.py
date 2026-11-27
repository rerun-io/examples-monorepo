"""A layer's run record: the command, config, code, environment, host and cost of the run that wrote the layer.

The stage that writes a layer writes its record into the same recording, so a layer run by hand carries one as well
as a layer the flow runner started, and the runner never keeps a second record of what ran. The record is a recording
property group named ``run_<layer>``: the catalog merges a segment's property columns across its layers and a column
two layers share keeps only the last registered value (:mod:`simplecv.catalog_layers`), so each layer needs its own
group. Re-running ``command`` from ``cwd`` in ``pixi_environment`` at ``git_commit`` (with no ``git_dirty_sha256``)
and the same ``pixi_lock_sha256`` repeats the run.
"""

import dataclasses
import hashlib
import os
import resource
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import rerun as rr
from rerun.catalog import DatasetEntry
from serde import from_dict, serde

from simplecv.catalog_layers import segment_properties

RUN_RECORD_PREFIX: str = "run_"
"""A layer's record is the property group ``run_<layer>``."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunRecord:
    """What wrote one layer of one segment."""

    layer: str
    """The catalog layer the run wrote."""
    segment: str
    """The segment the layer belongs to."""
    command: str
    """The process's command line, as a shell would take it (``python <argv...>``)."""
    cwd: str
    """The working directory, relative to the repository root when it is inside it."""
    pixi_environment: str
    """The pixi environment the process ran in; empty outside pixi."""
    config: str
    """The stage's resolved configuration (its ``repr``), defaults included: the command alone does not state them."""
    git_commit: str
    """The commit checked out; empty outside a git checkout."""
    git_dirty_sha256: str
    """SHA-256 of the uncommitted changes (``git diff HEAD`` and every untracked file); empty for a clean checkout."""
    pixi_lock_sha256: str
    """SHA-256 of the repository's ``pixi.lock``; empty when there is none."""
    host: str
    """The machine's host name."""
    gpu: str
    """The GPUs ``nvidia-smi`` reports, with their driver, ``; ``-separated; empty without one."""
    started_at: str
    """When the run started, ISO 8601 UTC."""
    wall_s: float
    """Seconds from ``started_at`` to when the record was taken."""
    peak_rss_mb: float
    """The process's peak resident memory."""
    peak_gpu_mb: float | None = None
    """Peak memory torch reserved on the GPU; None when the process used no torch CUDA."""


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _dirty_sha256(root: Path) -> str:
    status: bytes = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not status:
        return ""
    digest = hashlib.sha256(_git(root, "diff", "HEAD", "--binary"))
    for entry in status.split(b"\0"):
        if entry.startswith(b"?? "):
            path: bytes = entry[3:]
            digest.update(path + b"\0" + (root / path.decode()).read_bytes())
    return digest.hexdigest()


def _gpus() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return "; ".join(line.strip() for line in result.stdout.splitlines() if line.strip()) if result.returncode == 0 else ""


def _peak_gpu_mb() -> float | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return None
    return float(torch.cuda.max_memory_reserved()) / 2**20


def capture_run_record(layer: str, segment: str, *, config: object, started: datetime, cwd: Path | None = None) -> RunRecord:
    """Take this process's record for ``layer``, now: call it when the layer's content is complete.

    Args:
        layer: The catalog layer being written.
        segment: Its segment.
        config: The stage's parsed configuration; its ``repr`` is recorded.
        started: When the run started (timezone-aware).
        cwd: The working directory; the process's own by default.

    Returns:
        The record.
    """
    directory: Path = (Path.cwd() if cwd is None else cwd).resolve()
    root: Path | None = None
    if shutil.which("git") is not None:
        found = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
        root = Path(found.stdout.strip()).resolve() if found.returncode == 0 else None
    lock: Path | None = None if root is None else root / "pixi.lock"
    rss_kb: float = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return RunRecord(
        layer=layer,
        segment=segment,
        command=shlex.join(["python", *sys.argv]),
        cwd=str(directory.relative_to(root)) if root is not None and directory.is_relative_to(root) else str(directory),
        pixi_environment=os.environ.get("PIXI_ENVIRONMENT_NAME", ""),
        config=repr(config),
        git_commit="" if root is None else _git(root, "rev-parse", "HEAD").decode().strip(),
        git_dirty_sha256="" if root is None else _dirty_sha256(root),
        pixi_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock is not None and lock.is_file() else "",
        host=socket.gethostname(),
        gpu=_gpus(),
        started_at=started.isoformat(),
        wall_s=(datetime.now(UTC) - started).total_seconds(),
        # ru_maxrss is KiB on Linux, bytes on macOS.
        peak_rss_mb=rss_kb / 2**20 if sys.platform == "darwin" else rss_kb / 2**10,
        peak_gpu_mb=_peak_gpu_mb(),
    )


def write_run_record(recording: rr.RecordingStream, record: RunRecord) -> None:
    """Write ``record`` into the layer's recording as the property group ``run_<layer>``."""
    values: rr.AnyValues = rr.AnyValues()
    for field in dataclasses.fields(record):
        values.with_component_from_data(field.name, getattr(record, field.name))  # a None field is dropped: it reads back as None
    recording.send_property(f"{RUN_RECORD_PREFIX}{record.layer}", values)


def read_run_record(entry: DatasetEntry, segment: str, layer: str) -> RunRecord:
    """The record of the run that wrote ``layer`` of ``segment``, from the catalog's segment table.

    Raises:
        ValueError: If the layer carries no record, or the record does not decode.
    """
    cells: dict[str, list[object]] = segment_properties(entry, segment, f"{RUN_RECORD_PREFIX}{layer}")
    if not cells:
        raise ValueError(f"no run record for layer {layer!r} of segment {segment!r} in dataset {entry.name!r}")
    return from_dict(RunRecord, {name: cell[0] for name, cell in cells.items()})
