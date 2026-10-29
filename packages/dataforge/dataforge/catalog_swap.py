"""Idle-gated, bounded catalog layer swaps, without scanning segment tables."""

from __future__ import annotations

import math
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic as time_monotonic
from time import sleep as time_sleep
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

import rerun.chunk as rrc
from rerun.catalog import OnDuplicateSegmentLayer, RegistrationResult


@runtime_checkable
class CatalogTask(Protocol):
    """Catalog mutation completion handle."""

    def wait(self) -> RegistrationResult | None:
        """Wait until the mutation finishes."""
        ...


@runtime_checkable
class LayerEntry(Protocol):
    """The only dataset operations a swap needs; no full-table reads."""

    def unregister(self, *, segments_to_drop: list[str], layers_to_drop: list[str]) -> CatalogTask:
        """Drop exactly these segment layers."""
        ...

    def register(self, uris: list[str], /, *, layer_name: str, on_duplicate: OnDuplicateSegmentLayer) -> CatalogTask:
        """Register a batch of files."""
        ...


def listen_backlog_idle(port: int) -> bool:
    """Require a local listener with zero queued connections."""
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            ["ss", "-H", "-ltn", f"( sport = :{port} )"], capture_output=True, text=True, check=True, timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    rows: list[list[str]] = [line.split() for line in result.stdout.splitlines() if line.strip()]
    return bool(rows) and all(len(row) > 1 and row[1] == "0" for row in rows)


def listing_idle(catalog_url: str) -> bool:
    """Bound the cheap dataset listing to ten seconds in a disposable process."""
    try:
        subprocess.run(
            [sys.executable, "-c", "import sys; from rerun.catalog import CatalogClient; CatalogClient(sys.argv[1]).datasets()", catalog_url],
            capture_output=True, check=True, timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def nfs_read_stats(mount: Path) -> tuple[int, float]:
    """Return cumulative READ operations and execute milliseconds for one mount."""
    active: bool = False
    for line in Path("/proc/self/mountstats").read_text().splitlines():
        if line.startswith("device "):
            active = f" mounted on {mount} with fstype nfs" in line
        elif active and line.strip().startswith("READ:"):
            fields: list[str] = line.split()
            return int(fields[1]), float(fields[8])
    raise ValueError(f"NFS mount has no READ stats: {mount}")


def catalog_idle(
    catalog_url: str,
    nfs_mount: Path | None = None,
    *,
    local: bool = True,
    backlog_probe: Callable[[int], bool] = listen_backlog_idle,
    listing_probe: Callable[[str], bool] = listing_idle,
    read_probe: Callable[[Path], tuple[int, float]] = nfs_read_stats,
    sleep: Callable[[float], None] = time_sleep,
) -> bool:
    """Check backlog, optional 20-second NFS latency (<100 ms), and listing."""
    if local and not backlog_probe(urlparse(catalog_url.removeprefix("rerun+")).port or 80):
        return False
    if nfs_mount is not None:
        before: tuple[int, float] = read_probe(nfs_mount)
        sleep(20.0)
        after: tuple[int, float] = read_probe(nfs_mount)
        operations: int = after[0] - before[0]
        elapsed_ms: float = after[1] - before[1]
        if operations < 0 or elapsed_ms < 0.0 or (operations > 0 and elapsed_ms / operations >= 100.0):
            return False
    return listing_probe(catalog_url)


def swap_layer(
    entry: LayerEntry,
    layer: str,
    files: Sequence[Path],
    *,
    idle_probe: Callable[[], bool],
    batch_size: int = 500,
    gate_timeout_s: float = 600.0,
    monotonic: Callable[[], float] = time_monotonic,
    sleep: Callable[[float], None] = time_sleep,
) -> None:
    """Wait before each batch, unregister exact file stems, then register with SKIP.

    Every file must contain exactly one recording matching its stem. Validate
    the entire request before the first mutation. An empty request is refused.
    Each batch waits at most gate_timeout_s, logs progress every 30 seconds,
    and prints its IDs before unregistering. Registration must return exactly
    those IDs; a mismatch raises and stops subsequent batches.
    """
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", layer) is None:
        raise ValueError("layer must match [a-zA-Z0-9][a-zA-Z0-9_-]*")
    if batch_size <= 0:
        raise ValueError("swap requires a positive batch size")
    if not math.isfinite(gate_timeout_s) or gate_timeout_s <= 0.0:
        raise ValueError("idle gate timeout must be finite and positive")
    if not files:
        raise ValueError("swap-layer requires files")
    ids: list[str] = [path.stem for path in files]
    if len(ids) != len(set(ids)):
        raise ValueError("swap file stems must be unique segment IDs")
    for path in files:
        if path.suffix != ".rrd" or not path.stem or not path.is_file():
            raise ValueError(f"swap requires an existing segment .rrd: {path}")
        recordings: list[rrc.StoreEntry] = rrc.RrdReader(path).recordings()
        if len(recordings) != 1 or recordings[0].recording_id != path.stem:
            found: list[str] = [recording.recording_id for recording in recordings]
            raise ValueError(f"{path}: expected exactly one recording with ID {path.stem!r}, found {found}")
    for start in range(0, len(files), batch_size):
        batch: Sequence[Path] = files[start : start + batch_size]
        segments: list[str] = [path.stem for path in batch]
        if not segments:
            raise ValueError("refusing an empty unregister: it would drop all segments")
        deadline: float = monotonic() + gate_timeout_s
        while True:
            idle: bool = idle_probe()
            remaining: float = deadline - monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"idle gate timed out after {gate_timeout_s:g}s for layer {layer}")
            if idle:
                break
            print(f"waiting for catalog idle gate: layer={layer}, {remaining:.0f}s remaining", flush=True)
            sleep(min(30.0, remaining))
        print(f"swap layer={layer}: unregister segment ids={segments}", flush=True)
        entry.unregister(segments_to_drop=segments, layers_to_drop=[layer]).wait()
        result: RegistrationResult | None = entry.register(
            [path.resolve().as_uri() for path in batch], layer_name=layer, on_duplicate=OnDuplicateSegmentLayer.SKIP,
        ).wait()
        if result is None or sorted(result.segment_ids) != sorted(segments):
            actual: list[str] | None = None if result is None else result.segment_ids
            raise RuntimeError(f"registration IDs differ: expected {segments}, got {actual}; layer={layer}")
