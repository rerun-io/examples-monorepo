"""Save collected agent rows as deterministic Rerun columns."""

import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import rerun as rr

from agent_traces.blueprint import session_blueprint
from agent_traces.events import Scalar, Session
from agent_traces.rows import FACT_TYPES, ImageRow, Row, Rows, ScalarRow, TextRow, collect_rows
from agent_traces.writing import atomic_write

PROPERTY_TYPES: dict[str, pa.DataType] = {
    **dict.fromkeys(("session_id", "profile", "agent", "source_path", "source_sha256", "host", "cwd", "git_branch",
                     "title", "cli_versions", "models", "provider", "originator", "thread_source", "forked_from", "parent_thread"), pa.string()),
    **dict.fromkeys(("n_turns", "n_subagents", "n_tool_calls", "n_images", "n_inlined_outputs", "total_input_tokens", "total_output_tokens", "total_cache_read_tokens"), pa.int64()),
    "total_cost_usd": pa.float64(),
}
"""Stable catalog types, including properties whose current value is null."""


@dataclass(frozen=True, slots=True)
class WrittenRecording:
    """Published recording and temporal row counts."""

    path: Path
    """Published path."""
    entity_rows: dict[str, int]
    """Temporal rows per entity."""


def metadata_columns(batch: Sequence[TextRow | ScalarRow | ImageRow]) -> rr.ComponentColumnList:
    """Write declared facts with one null policy; keep other provenance in JSON."""
    keys: set[str] = {key for row in batch for key in row.values}
    typed: set[str] = (keys & FACT_TYPES.keys()) | {"agent_id"}
    return rr.AnyValues.columns(
        **{key: pa.array([[row.values[key]] if key in row.values else [] for row in batch], type=pa.list_(FACT_TYPES[key]))
           for key in sorted(typed)},
        metadata_json=pa.array([[orjson.dumps({key: value for key, value in row.values.items() if key not in typed},
                                             option=orjson.OPT_SORT_KEYS).decode()] for row in batch], type=pa.list_(pa.string())),
    )


def write_session_rrd(session: Session, out: Path, *, host: str | None = None) -> WrittenRecording:
    """Collect and publish a session with a capability-derived blueprint atomically."""
    rows: Rows = collect_rows(session)
    recording: rr.RecordingStream = rr.RecordingStream("agent_traces", recording_id=rows.session_id, send_properties=False)
    counts: dict[str, int] = {}
    def send(entity: str, batch: Sequence[Row], columns: Callable[[Sequence[Row]], rr.ComponentColumnList]) -> None:  # noqa: UP047 - Runtime TypeVar annotations.
        """Send a preordered batch with its assigned event indices."""
        indexes: list[rr.TimeColumn] = [
            rr.TimeColumn("wall", timestamp=np.array([row.timestamp_ns for row in batch], dtype="datetime64[ns]")),
            rr.TimeColumn("event", sequence=[row.event for row in batch]),
        ]
        recording.send_columns(entity, indexes=indexes, columns=[*columns(batch), *metadata_columns(batch)], strict=True)
        counts[entity] = len(batch)

    with atomic_write(out) as temporary:
        try:
            entities: set[str] = {entity for batches in (rows.texts, rows.documents, rows.scalars, rows.images)
                                  for entity, batch in batches.items() if batch}
            recording.save(temporary, default_blueprint=session_blueprint(entities))
            for entity, documents in rows.documents.items():
                send(entity, documents, lambda batch: rr.TextDocument.columns(text=[row.text for row in batch], media_type=["text/markdown"] * len(batch)))
            for entity, texts in rows.texts.items():
                send(entity, texts, lambda batch: rr.TextLog.columns(text=[row.text for row in batch], level=[row.level for row in batch],
                                                       color=np.array([row.color for row in batch], dtype=np.uint32)))
            for entity, scalars in rows.scalars.items():
                recording.log(entity, rr.SeriesLines(names=rows.series_names[entity]), static=True)
                send(entity, scalars, lambda batch: rr.Scalars.columns(scalars=[row.value for row in batch]))
            for entity, images in rows.images.items():
                send(entity, images, lambda batch: rr.EncodedImage.columns(blob=[row.blob for row in batch], media_type=[row.media_type for row in batch]))
            properties: dict[str, Scalar | None] = {**rows.properties, "host": host if host is not None else socket.gethostname()}
            recording.send_property("session", rr.AnyValues(drop_untyped_nones=True, **{
                key: pa.array([value], type=PROPERTY_TYPES[key])
                for key, value in properties.items()
            }))
            if rows.agents:
                recording.send_property("agents", rr.AnyValues(
                    n=pa.array([row.n for row in rows.agents], type=pa.int64()),
                    agent_id=pa.array([row.agent_id for row in rows.agents], type=pa.string()),
                    agent_type=pa.array([row.metadata.agent_type for row in rows.agents], type=pa.string()),
                    description=pa.array([row.metadata.description for row in rows.agents], type=pa.string()),
                ))
            if rows.skipped:
                recording.send_property("skipped", rr.AnyValues(drop_untyped_nones=True, **rows.skipped))
            first_event: int | None = min((record.timestamp_ns for records in (session.main, *session.subagents.values()) for record in records), default=None)
            if first_event is not None:
                recording.send_recording_start_time_nanos(first_event)
            recording.send_recording_name(rows.name)
            recording.flush()
        finally:
            recording.disconnect()
    return WrittenRecording(out, counts)
