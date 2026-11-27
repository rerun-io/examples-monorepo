"""A layer's run record: the command, config, code, environment, host and cost of the run that wrote the layer.

The stage that writes a layer writes its record into the same recording, so a layer run by hand carries one as well
as a layer the flow runner started, and the runner never keeps a second record of what ran. The record is a recording
property group named ``run_<layer>``: the catalog merges a segment's property columns across its layers and a column
two layers share keeps only the last registered value (:mod:`simplecv.catalog_layers`), so each layer needs its own
group. At ``git_commit`` (with no ``git_dirty_sha256``) and the same ``pixi_lock_sha256``, re-running
``cd <repository root>/<cwd> && env <environment> pixi run -e <pixi_environment> --frozen <command>`` repeats the run.
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
from collections.abc import Sequence
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
    """What wrote one layer of one segment. A field that may be empty defaults to it: the catalog returns no cell for an
    empty string."""

    layer: str
    """The catalog layer the run wrote."""
    segment: str
    """The segment the layer belongs to."""
    command: str
    """The process's command line, as a shell would take it (``python <argv...>``)."""
    cwd: str
    """The working directory, relative to the root of the stage's repository when it is inside it, else absolute."""
    config: str
    """The stage's resolved configuration (its ``repr``), defaults included: the command alone does not state them."""
    host: str
    """The machine's host name."""
    started_at: str
    """When the run started, ISO 8601 UTC."""
    wall_s: float
    """Seconds from ``started_at`` to when the layer's record was taken."""
    process_peak_rss_mb: float
    """The process's peak resident memory since it started: a batch process's earlier work counts, so it bounds this
    layer's need from above."""
    pixi_environment: str = ""
    """The pixi environment the process ran in; empty outside pixi."""
    git_commit: str = ""
    """The commit checked out; empty outside a git checkout."""
    git_dirty_sha256: str = ""
    """SHA-256 of the uncommitted changes (``git diff HEAD`` and every untracked file); empty for a clean checkout."""
    pixi_lock_sha256: str = ""
    """SHA-256 of the repository's ``pixi.lock``; empty when there is none."""
    gpu: str = ""
    """The GPUs ``nvidia-smi`` reports, with their driver, ``; ``-separated; empty without one, ``unavailable: <why>``
    when the probe failed."""
    environment: str = ""
    """The environment variables the stage reads as inputs, set ones only, as ``NAME=value`` words (``shlex``): what a
    replay needs beside ``command`` (an output root, a catalog URL taken from the environment)."""
    peak_torch_gpu_mb: float | None = None
    """Peak GPU memory torch's allocator reserved in the process; None when it used no torch CUDA. Native GPU backends
    (wgpu, TensorRT) allocate outside it, so this is not the process's GPU memory."""


GPU_PROBE_TIMEOUT_S: float = 10.0
"""Longest wait for ``nvidia-smi``: a stalled driver must not hold a finished layer back from being published."""
HASH_BLOCK_BYTES: int = 1 << 20
"""Read size when hashing uncommitted files: a large untracked file is hashed without being held in memory."""


@dataclass(frozen=True, slots=True)
class RunStart:
    """What a run is, taken when it starts: the code it loaded, not the checkout as it is when a layer is finished."""

    command: str
    """:attr:`RunRecord.command`."""
    cwd: str
    """:attr:`RunRecord.cwd`."""
    pixi_environment: str
    """:attr:`RunRecord.pixi_environment`."""
    config: str
    """:attr:`RunRecord.config`: replace it (``dataclasses.replace``) once the stage has resolved what it was given."""
    git_commit: str
    """:attr:`RunRecord.git_commit`."""
    git_dirty_sha256: str
    """:attr:`RunRecord.git_dirty_sha256`."""
    pixi_lock_sha256: str
    """:attr:`RunRecord.pixi_lock_sha256`."""
    host: str
    """:attr:`RunRecord.host`."""
    gpu: str
    """:attr:`RunRecord.gpu`."""
    environment: str
    """:attr:`RunRecord.environment`."""
    started: datetime
    """When the run started, timezone-aware; a batch process replaces it per item it converts."""

    def finish(self, layer: str, segment: str) -> RunRecord:
        """The record of ``layer``, now: call it when the layer's content is complete."""
        rss_kb: float = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        torch = sys.modules.get("torch")
        return RunRecord(
            layer=layer,
            segment=segment,
            command=self.command,
            cwd=self.cwd,
            pixi_environment=self.pixi_environment,
            config=self.config,
            git_commit=self.git_commit,
            git_dirty_sha256=self.git_dirty_sha256,
            pixi_lock_sha256=self.pixi_lock_sha256,
            host=self.host,
            gpu=self.gpu,
            environment=self.environment,
            started_at=self.started.isoformat(),
            wall_s=(datetime.now(UTC) - self.started).total_seconds(),
            # ru_maxrss is KiB on Linux, bytes on macOS.
            process_peak_rss_mb=rss_kb / 2**20 if sys.platform == "darwin" else rss_kb / 2**10,
            peak_torch_gpu_mb=float(torch.cuda.max_memory_reserved()) / 2**20 if torch is not None and torch.cuda.is_initialized() else None,
        )


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _dirty_sha256(root: Path) -> str:
    status: bytes = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not status:
        return ""
    digest = hashlib.sha256()
    with subprocess.Popen(["git", "-C", str(root), "diff", "HEAD", "--binary"], stdout=subprocess.PIPE) as diff:
        assert diff.stdout is not None
        while block := diff.stdout.read(HASH_BLOCK_BYTES):
            digest.update(block)
    if diff.returncode != 0:
        raise subprocess.CalledProcessError(diff.returncode, diff.args)
    for entry in status.split(b"\0"):
        if entry.startswith(b"?? "):
            path: bytes = entry[3:]
            digest.update(path + b"\0")
            untracked: Path = root / path.decode()
            # Git records a link as its target text; following it could open a directory or a missing file.
            if untracked.is_symlink():
                digest.update(os.readlink(untracked).encode())
                continue
            with untracked.open("rb") as stream:
                while block := stream.read(HASH_BLOCK_BYTES):
                    digest.update(block)
    return digest.hexdigest()


def _gpus() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False, timeout=GPU_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"unavailable: nvidia-smi gave no answer in {GPU_PROBE_TIMEOUT_S:.0f} s"
    if result.returncode != 0:
        return f"unavailable: nvidia-smi exited {result.returncode}: {result.stderr.strip()[:200]}"
    return "; ".join(line.strip() for line in result.stdout.splitlines() if line.strip())


def start_run(config: object, *, source: Path, environment: Sequence[str] = ()) -> RunStart:
    """Take this process's run identity, now: call it before the stage starts its work.

    Args:
        config: The stage's configuration; its ``repr`` is recorded. Pass it resolved where the stage resolves it
            (defaults from the environment, a catalog URL from a settings file), or replace it later.
        source: A file of the stage's own code (its ``__file__``): the repository holding it names the code and the
            lock, whatever directory the process runs in.
        environment: The environment variables the stage reads as inputs; the set ones are recorded.

    Returns:
        The identity, started now.
    """
    directory: Path = Path.cwd().resolve()
    root: Path | None = None
    if shutil.which("git") is not None:
        found = subprocess.run(["git", "-C", str(source.resolve().parent), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
        root = Path(found.stdout.strip()).resolve() if found.returncode == 0 else None
    lock: Path | None = None if root is None else root / "pixi.lock"
    lock_digest = hashlib.sha256()
    if lock is not None and lock.is_file():
        with lock.open("rb") as stream:
            while block := stream.read(HASH_BLOCK_BYTES):
                lock_digest.update(block)
    return RunStart(
        command=shlex.join(["python", *sys.argv]),
        cwd=str(directory.relative_to(root)) if root is not None and directory.is_relative_to(root) else str(directory),
        pixi_environment=os.environ.get("PIXI_ENVIRONMENT_NAME", ""),
        config=repr(config),
        git_commit="" if root is None else _git(root, "rev-parse", "HEAD").decode().strip(),
        git_dirty_sha256="" if root is None else _dirty_sha256(root),
        pixi_lock_sha256=lock_digest.hexdigest() if lock is not None and lock.is_file() else "",
        host=socket.gethostname(),
        gpu=_gpus(),
        environment=shlex.join(f"{name}={os.environ[name]}" for name in environment if name in os.environ),
        started=datetime.now(UTC),
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
