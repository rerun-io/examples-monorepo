"""Publish fresh recording and segment-table layouts without overwriting live files."""

from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from agent_traces.writing import atomic_write


def save_table_blueprint(table_path: Path, columns: list[str]) -> None:
    """Write the session table order and hide provider skip counters."""
    shown: dict[str, str] = {
        "property:RecordingInfo:name": "Recording",
        "wall:start": "Start",
        "property:session:n_turns": "Turns",
        "property:session:host": "Host",
        "property:session:agent": "Agent",
        "property:session:profile": "Profile",
    }
    # Rerun 0.38 table layouts use blueprint entities until a high-level API exists.
    with atomic_write(table_path) as temporary, rr.RecordingStream._from_native(
        rr.bindings.new_blueprint(application_id="agent_traces", make_default=False, make_thread_default=False, default_enabled=True)
    ) as stream:
        stream.save(str(temporary))
        stream.set_time("blueprint", sequence=0)
        for prefix in ("/table/layouts/table/columns", "/table/layouts/cards/fields"):
            for column, label in shown.items():
                stream.log(f"{prefix}/{rr.escape_entity_path_part(column)}", rrb.experimental.TableColumn(visible=True, name=label))
            for column in columns:
                if column.startswith("property:skipped:"):
                    stream.log(f"{prefix}/{rr.escape_entity_path_part(column)}", rrb.experimental.TableColumn(visible=False))
        stream.log("/table/layouts/table", rrb.experimental.TableLayout(column_order=list(shown)))
        stream.log("/table/layouts/cards", rrb.experimental.CardLayout(field_order=list(shown), title="property:RecordingInfo:name"))
