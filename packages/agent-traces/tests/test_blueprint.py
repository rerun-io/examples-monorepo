"""Saved layouts show only the available data, including child-only families."""

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest
from rerun.chunk import RrdReader

from agent_traces.events import (
    AssistantText,
    Image,
    Lifecycle,
    Payload,
    Session,
    Thinking,
    TimedRecord,
    ToolCall,
    ToolResult,
    Usage,
    UsageSample,
)
from agent_traces.rerun_log import write_session_rrd


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize(("payload", "expected"), [
    (AssistantText("reply"), {"Conversation"}),
    (Thinking("reasoning"), {"Thinking"}),
    (ToolCall("Read", "call", "{}", "file_read"), {"Tools"}),
    (ToolResult("Read", "call", "result", "{}", "file_read", 250.0), {"Tools", "Tool elapsed (ms)"}),
    (ToolResult("Read", "call", "result", "{}", "file_read", None), {"Tools"}),
    (Lifecycle("system", "started"), {"Lifecycle"}),
    (UsageSample(Usage(output_tokens=3)), {"Tokens per request"}),
    (UsageSample(Usage(cache_read_tokens=12)), {"Cache tokens"}),
    (Image(b"image", "image/png"), {"Images"}),
    (None, set()),
])
def test_saved_views_require_matching_data(tmp_path: Path, payload: Payload | None, expected: set[str], child: bool) -> None:
    """Unrelated views and empty panes are absent; child entities enable their own views."""
    records: list[TimedRecord] = [TimedRecord(payload, 1_000_000_000, 0)] if payload is not None else []
    session: Session = Session("synthetic", "test", tmp_path / "source.jsonl", [] if child else records,
                               {"child": records} if child else {}, Counter())
    recording: Path = write_session_rrd(session, tmp_path / "views.rrd").path
    reader: RrdReader = RrdReader(recording)
    names: set[str] = set()
    for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "ViewBlueprint:display_name" in batch.schema.names:
            names.update(name for row in batch.column("ViewBlueprint:display_name").to_pylist() for name in row)
        if "ContainerBlueprint:contents" in batch.schema.names:
            assert all(batch.column("ContainerBlueprint:contents").to_pylist()), "No empty layout containers"
    if not child and isinstance(payload, AssistantText):
        expected = expected | {"Current message"}
    assert names == expected
