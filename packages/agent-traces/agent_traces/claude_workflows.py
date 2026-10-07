"""Read inventoried Claude workflow sidecars without executing scripts."""

from dataclasses import dataclass
from pathlib import Path

from serde import SerdeError, serde
from serde.json import from_json

from agent_traces.events import Lifecycle, TimedRecord
from agent_traces.timestamps import parse_timestamp_ns


@serde
@dataclass(frozen=True, slots=True)
class Workflow:
    """Partial third-party run envelope; the full document remains visible."""

    runId: str = ""
    """Workflow run identifier."""
    timestamp: str | None = None
    """Run wall timestamp when available."""
    status: str = ""
    """Reported run state."""
    error: str | None = None
    """Run failure description."""
    totalTokens: int | None = None
    """Reported aggregate token count."""


@dataclass(frozen=True, slots=True)
class WorkflowSource:
    """One sidecar's content, optional run envelope, and reported time."""

    path: Path
    """Inventoried source file."""
    text: str
    """Verbatim decoded content."""
    run: Workflow | None
    """Run metadata, absent for scripts."""
    timestamp_ns: int | None
    """Reported wall time, absent for untimed sidecars."""


def workflow_records(paths: tuple[Path, ...], fallback_timestamp_ns: int | None) -> list[TimedRecord]:
    """Retain full runs and script bytes as lifecycle rows with source provenance."""
    if not paths:
        return []
    loaded: list[WorkflowSource] = []
    for path in paths:
        text: str = path.read_bytes().decode("utf-8", "replace")
        try:
            run: Workflow | None = from_json(Workflow, text) if path.suffix != ".js" else None
            timestamp: int | None = parse_timestamp_ns(run.timestamp) if run is not None and run.timestamp else None
        except (SerdeError, ValueError) as error:
            raise ValueError(f"{path}: invalid workflow ({type(error).__name__})") from None
        loaded.append(WorkflowSource(path, text, run, timestamp))
    fallback: int | None = min((stamp for stamp in [fallback_timestamp_ns, *(source.timestamp_ns for source in loaded)] if stamp is not None), default=None)
    if fallback is None:
        raise ValueError(f"{paths[0]}: workflow sidecars have no timed event in the session")
    records: list[TimedRecord] = []
    for index, source in enumerate(loaded):
        run = source.run
        if run is None:
            records.append(TimedRecord(Lifecycle("workflow_scripts", source.text), fallback, index, {"source_path": str(source.path)}))
        else:
            records.append(TimedRecord(Lifecycle("workflows", source.text, "ERROR" if run.error or run.status == "failed" else "INFO"),
                source.timestamp_ns if source.timestamp_ns is not None else fallback, index,
                {"source_path": str(source.path), "run_id": run.runId, "status": run.status,
                 **({"total_tokens": run.totalTokens} if run.totalTokens is not None else {})}))
    return records
