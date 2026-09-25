"""Per-sequence wall times and append-only benchmark records.

Stages may overlap (base writing includes transcode); never sum them for totals.
The context-local timer lets converters add stages without changing their API.
"""

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import pyarrow as pa
import pyarrow.compute as pc
import rerun.chunk as rrc
from serde import SerdeError, serde
from serde.json import from_json, to_json


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ConvertRecord:
    """One sequence conversion, including a no-op conversion."""

    dataset: str
    """Source dataset name."""
    recording_id: str
    """Stable sequence identity."""
    converter_version: str
    """Conversion schema version and source commit when available."""
    started_at: str
    """ISO UTC start."""
    stage_s: dict[str, float]
    """Seconds per named stage."""
    total_s: float
    """Elapsed conversion seconds."""
    capture_s: float | None
    """Base video_time span; null when absent."""
    layer_bytes: dict[str, int]
    """Bytes in each file written by this invocation."""
    host: str
    """Host name."""
    skipped: bool
    """No layer files changed in a successful conversion."""
    error: str | None = None
    """Failure description, or null after success."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RegisterRecord:
    """One catalog registration run."""

    dataset: str
    """Catalog dataset name."""
    started_at: str
    """ISO UTC start."""
    layer_s: dict[str, float]
    """Registration seconds per layer, including wait."""
    blueprint_s: float
    """Seconds publishing and retiring blueprints."""
    segment_count: int
    """Number of selected base recordings."""
    total_s: float
    """Elapsed run seconds."""
    host: str
    """Host name."""


class SequenceTimer:
    """Accumulate repeated named stages using a monotonic clock."""

    def __init__(self) -> None:
        self.started_at: str = datetime.now(UTC).isoformat()
        self.start: float = perf_counter()
        self.stage_s: dict[str, float] = {}

    @property
    def total_s(self) -> float:
        """Elapsed wall seconds."""
        return perf_counter() - self.start

    def add(self, name: str, seconds: float) -> None:
        """Accumulate seconds onto a named stage."""
        self.stage_s[name] = self.stage_s.get(name, 0.0) + seconds

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure a stage even when it raises."""
        start: float = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)


_ACTIVE: ContextVar[SequenceTimer | None] = ContextVar("dataforge_timer", default=None)


@contextmanager
def sequence_timer() -> Iterator[SequenceTimer]:
    """Start an isolated timer for one convert call."""
    timer: SequenceTimer = SequenceTimer()
    token: Token[SequenceTimer | None] = _ACTIVE.set(timer)
    try:
        yield timer
    finally:
        _ACTIVE.reset(token)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Instrument an optional stage; direct library callers need no timer."""
    timer: SequenceTimer | None = _ACTIVE.get()
    if timer is None:
        yield
    else:
        with timer.stage(name):
            yield


def record(name: str, seconds: float) -> None:
    """Add measured wall seconds to the active timer; no-op without a timer."""
    timer: SequenceTimer | None = _ACTIVE.get()
    if timer is not None:
        timer.add(name, seconds)


def append_record(path: Path, record: ConvertRecord | RegisterRecord) -> None:
    """Append a complete JSON line under a process lock, creating directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        fcntl.flock(output, fcntl.LOCK_EX)  # held until close() has flushed the line
        output.write(to_json(record) + "\n")


def load_convert_records(path: Path) -> list[ConvertRecord]:
    """Read typed records; report the file and line of a malformed record."""
    records: list[ConvertRecord] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(from_json(ConvertRecord, line))
        except (SerdeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    return records


def capture_span(path: Path) -> float | None:
    """Read the base recording's video_time bounds without decoding video."""
    bounds: list[int] = []
    reader: rrc.RrdReader = rrc.RrdReader(path)
    for chunk in reader.stream():
        if "video_time" in chunk.timeline_names:
            values: pa.Array = chunk.to_record_batch().column("video_time").cast("int64")
            limits = pc.call_function("min_max", [values]).as_py()
            if limits["min"] is not None:
                bounds.extend((limits["min"], limits["max"]))
    return (max(bounds) - min(bounds)) / 1e9 if bounds else None
