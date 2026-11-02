from pathlib import Path

import pyarrow as pa
import pytest
import rerun as rr
import rerun.chunk as rrc

from dataforge.writing import atomic_recording, atomic_write, recording_to, should_skip


def property_columns(target: Path) -> set[str]:
    """Every ``property:*`` column of a saved rrd, read through the public reader."""
    store: rrc.ChunkStore = rrc.ChunkStore.from_chunks(list(rrc.RrdReader(target).stream()))
    table: pa.Table = store.reader(index=None, contents="/__properties/**").to_arrow_table()
    return {name for name in table.column_names if name.startswith("property:")}


def test_atomic_write_replaces_the_target_only_on_success(tmp_path: Path) -> None:
    target: Path = tmp_path / "blueprints" / "robocap.rbl"
    with atomic_write(target) as temp_path:
        assert temp_path.parent == target.parent  # same filesystem, so os.replace is atomic
        temp_path.write_bytes(b"blueprint")
    assert target.read_bytes() == b"blueprint"
    assert target.stat().st_mode & 0o777 == 0o644  # mkstemp's 0600 would hide it from the catalog server
    assert not list(target.parent.glob("*.tmp"))


def test_atomic_write_leaves_a_previous_target_intact_on_failure(tmp_path: Path) -> None:
    target: Path = tmp_path / "robocap.rbl"
    target.write_bytes(b"old")
    with pytest.raises(RuntimeError, match="boom"), atomic_write(target) as temp_path:
        temp_path.write_bytes(b"half")
        raise RuntimeError("boom")
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_exclusive_write_refuses_an_existing_target_and_keeps_its_content(tmp_path: Path) -> None:
    target: Path = tmp_path / "segment.rrd"
    target.write_bytes(b"registered")
    with pytest.raises(FileExistsError, match="segment.rrd"), atomic_write(target, exclusive=True) as temp_path:
        temp_path.write_bytes(b"new")
    assert target.read_bytes() == b"registered"
    assert list(tmp_path.iterdir()) == [target]


def test_concurrent_exclusive_writes_stage_apart_and_the_first_to_finish_wins(tmp_path: Path) -> None:
    target: Path = tmp_path / "segment.rrd"
    with pytest.raises(FileExistsError), atomic_write(target, exclusive=True) as slower:
        slower.write_bytes(b"slower")
        with atomic_write(target, exclusive=True) as faster:
            assert faster != slower
            faster.write_bytes(b"faster")
    assert target.read_bytes() == b"faster"
    assert list(tmp_path.iterdir()) == [target]


def test_exclusive_write_publishes_a_world_readable_target_and_no_stage(tmp_path: Path) -> None:
    target: Path = tmp_path / "layer" / "segment.rrd"
    with atomic_write(target, exclusive=True) as temp_path:
        assert temp_path.parent == target.parent  # a hard link cannot cross filesystems
        temp_path.write_bytes(b"layer")
    assert target.read_bytes() == b"layer"
    assert target.stat().st_mode & 0o777 == 0o644
    assert target.stat().st_nlink == 1
    assert list(target.parent.iterdir()) == [target]


def test_failed_exclusive_write_leaves_nothing(tmp_path: Path) -> None:
    target: Path = tmp_path / "segment.rrd"
    with pytest.raises(RuntimeError, match="boom"), atomic_write(target, exclusive=True) as temp_path:
        temp_path.write_bytes(b"half")
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_should_skip_existing_unless_forced(tmp_path: Path) -> None:
    target: Path = tmp_path / "base" / "x.rrd"
    assert not should_skip(target, force=False)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"done")
    assert should_skip(target, force=False)
    assert not should_skip(target, force=True)


def test_atomic_recording_publishes_only_on_success(tmp_path: Path) -> None:
    target: Path = tmp_path / "robocap__a__b.rrd"
    with atomic_recording(target, application_id="dataforge", recording_id="robocap__a__b") as recording:
        recording.log("/world", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    assert target.exists()
    assert target.stat().st_size > 0
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_recording_leaves_no_target_or_tmp_on_failure(tmp_path: Path) -> None:
    target: Path = tmp_path / "robocap__a__c.rrd"
    with pytest.raises(RuntimeError, match="boom"), atomic_recording(target, application_id="dataforge", recording_id="robocap__a__c"):
        raise RuntimeError("boom")
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_recording_creates_parent_dirs(tmp_path: Path) -> None:
    target: Path = tmp_path / "rrd" / "base" / "robocap__a__d.rrd"
    with atomic_recording(target, application_id="dataforge", recording_id="robocap__a__d") as recording:
        recording.log("/world", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    assert target.exists()


def test_a_derived_layer_keeps_its_own_properties_but_states_no_recording_info(tmp_path: Path) -> None:
    """``send_properties=False`` drops Rerun's own RecordingInfo, not the caller's property groups.

    Both halves matter to the layer rule and neither is obvious, so they are
    pinned together: a derived layer must be able to carry ``property:gt:*``
    (that is its whole payload beside the data) while carrying no ``start_time``
    of its own, since that would be whenever it was last rebuilt rather than when
    the capture happened. Verified against the recording, not the docs.
    """
    derived: Path = tmp_path / "derived.rrd"
    with atomic_recording(derived, recording_id="msd-index__a__b", send_properties=False) as recording:
        recording.send_property("gt", rr.AnyValues(num_poses=41, source="mocap"))

    columns: set[str] = property_columns(derived)
    assert {"property:gt:num_poses", "property:gt:source"} <= columns
    assert not [name for name in columns if name.startswith("property:RecordingInfo:")], sorted(columns)


def test_a_base_layer_states_its_name_and_its_wall_clock(tmp_path: Path) -> None:
    """The default: a base layer is the recording, so Rerun's RecordingInfo belongs on it."""
    base: Path = tmp_path / "base.rrd"
    with atomic_recording(base, recording_id="msd-index__a__b") as recording:
        recording.send_recording_name("msd-index__a__b")
        recording.log("/world", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)

    columns: set[str] = property_columns(base)
    assert "property:RecordingInfo:start_time" in columns
    assert "property:RecordingInfo:name" in columns


def test_recording_to_writes_the_path_it_is_given_and_publishes_nothing(tmp_path: Path) -> None:
    """The staging half: a converter publishing several layers together owns the replace.

    A file written this way is complete and readable on exit, which is what lets
    a derived layer read the staged base rrd it was handed before either is
    published.
    """
    staged: Path = tmp_path / "staged.rrd"
    with recording_to(staged, recording_id="msd-index__a__b") as recording:
        recording.log("/world", rr.Points3D([[1.0, 2.0, 3.0]]), static=True)

    assert staged.is_file() and staged.stat().st_size > 0
    chunks: list[rrc.Chunk] = list(rrc.RrdReader(staged).stream())
    assert any(chunk.entity_path == "/world" for chunk in chunks), "the closed recording is readable"


def test_save_table_blueprint_writes_a_preview_card_for_the_recording_link(tmp_path: Path) -> None:
    """Rerun 0.38 table blueprints are the views plus /table entities naming the preview column and its views.

    With no declared fields the layouts keep the viewer default: no column is hidden."""
    import rerun.blueprint as rrb

    from dataforge.writing import TableFields, save_table_blueprint

    pane: rrb.Spatial2DView = rrb.Spatial2DView(origin="/world/rig_01/cam_00/pinhole", contents=["+ /world/rig_01/cam_00/pinhole/video"])
    follow: rrb.Spatial3DView = rrb.Spatial3DView(origin="/world/rig_00")
    target: Path = tmp_path / "blueprints" / "show3d-table.rbl"
    save_table_blueprint(
        rrb.Blueprint(rrb.Horizontal(follow, pane), collapse_panels=True),
        target,
        timeline="video_time",
        fields=TableFields(),
        columns=["rerun_segment_id", "property:capture:schema"],
    )
    reader: rrc.RrdReader = rrc.RrdReader(target)
    stores = reader.blueprints()
    assert len(stores) == 1
    chunks = list(reader.stream(store=stores[0]).to_chunks())
    paths: set[str] = {str(chunk.entity_path) for chunk in chunks}
    assert {"/table", "/table/layouts/table", "/table/layouts/cards"} <= paths
    column: str = "/table/layouts/table/columns/recording\\ link"
    assert column in paths and "/table/layouts/cards/fields/recording\\ link" in paths
    preview_views: list[str] = [
        view
        for chunk in chunks
        if str(chunk.entity_path) == column
        for column_name in chunk.to_record_batch().schema.names
        if column_name.endswith("TableColumnPreview:views")
        for view in chunk.to_record_batch().column(column_name).to_pylist()[0]
    ]
    assert sorted(preview_views) == sorted([follow.blueprint_path(), pane.blueprint_path()])
    assert all(path in paths for path in (f"/{follow.blueprint_path()}", f"/{pane.blueprint_path()}"))
    assert not any("capture:schema" in path for path in paths), "no declared fields: nothing is hidden"


def test_save_table_blueprint_shows_the_declared_fields_and_hides_every_other_column(tmp_path: Path) -> None:
    """A layout with declared fields shows them under their headers, first, and hides every other non-system column."""
    import rerun as rr
    import rerun.blueprint as rrb

    from dataforge.writing import SEGMENT_LINK_COLUMN, TableField, TableFields, save_table_blueprint

    action: TableField = TableField("property:episode:action", "action")
    coverage: TableField = TableField("property:object_pose:coverage", "object coverage")
    target: Path = tmp_path / "table.rbl"
    save_table_blueprint(
        rrb.Blueprint(rrb.Spatial3DView(origin="/world")),
        target,
        timeline="video_time",
        fields=TableFields(cards=(action,), table=(action, coverage)),
        columns=["rerun_segment_id", action.column, coverage.column, "property:capture:schema", "video_time:start"],
    )
    reader: rrc.RrdReader = rrc.RrdReader(target)
    logged: dict[tuple[str, str], object] = {
        (str(chunk.entity_path), name.rsplit(":", 1)[-1]): chunk.to_record_batch().column(name).to_pylist()[0]
        for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks()
        for name in chunk.to_record_batch().schema.names
        if name.startswith(("TableColumn:", "CardLayout:", "TableLayout:"))
    }

    def column(layout: str, name: str) -> tuple[object, object]:
        path: str = f"/table/layouts/{layout}/{rr.escape_entity_path_part(name)}"
        return logged.get((path, "visible")), logged.get((path, "name"))

    assert column("cards/fields", action.column) == ([True], ["action"])
    assert column("table/columns", action.column) == ([True], ["action"])
    assert column("table/columns", coverage.column) == ([True], ["object coverage"])
    assert column("cards/fields", coverage.column) == ([False], None), "declared for the table only"
    for layout in ("cards/fields", "table/columns"):
        assert column(layout, "property:capture:schema") == ([False], None)
        assert column(layout, "video_time:start") == ([False], None)
        assert column(layout, "rerun_segment_id") == ([False], None)
    assert logged[("/table/layouts/cards", "field_order")] == [SEGMENT_LINK_COLUMN, action.column]
    assert logged[("/table/layouts/table", "column_order")] == [SEGMENT_LINK_COLUMN, action.column, coverage.column]


def test_published_columns_applies_the_fields_to_the_columns_at_publish_time() -> None:
    from dataforge.writing import SEGMENT_LINK_COLUMN, PublishedColumns, TableField, TableFields, published_columns

    fields = TableFields(cards=(TableField("a", "A"),), table=(TableField("b", "B"),), hidden=("later", "a"), sort_by="b")
    assert published_columns(fields, ["a", "b", SEGMENT_LINK_COLUMN, "rerun_segment_id"]) == PublishedColumns(
        table_hidden=("a", "rerun_segment_id", "later"), cards_hidden=("b", "rerun_segment_id", "later"), sort_by="b"
    )
    assert published_columns(fields, ["a"]).sort_by is None, "the table has no such column yet"
    assert published_columns(TableFields(), ["a"]) == PublishedColumns((), (), None), "a layout with no fields keeps the viewer default"


def test_save_table_blueprint_sorts_the_rows_and_hides_columns_the_table_does_not_list_yet(tmp_path: Path) -> None:
    """``sort_by`` applies to a column the table has; ``hidden`` covers a column a later layer may add."""
    import rerun as rr
    import rerun.blueprint as rrb

    from dataforge.writing import TableField, TableFields, save_table_blueprint

    session: TableField = TableField("property:calib:session", "Capture / session")

    def saved(name: str, columns: list[str]) -> dict[tuple[str, str], object]:
        target: Path = tmp_path / name
        fields = TableFields(cards=(session,), table=(session,), hidden=("property:capture:collector",), sort_by=session.column)
        save_table_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/world")), target, timeline="video_time", fields=fields, columns=columns)
        reader: rrc.RrdReader = rrc.RrdReader(target)
        return {
            (str(chunk.entity_path), name.rsplit(":", 1)[-1]): chunk.to_record_batch().column(name).to_pylist()[0]
            for chunk in reader.stream(store=reader.blueprints()[0]).to_chunks()
            for name in chunk.to_record_batch().schema.names
            if name.startswith(("TableColumn:", "TableBlueprint:"))
        }

    logged = saved("with.rbl", ["rerun_segment_id", session.column])
    assert logged[("/table", "sort_by")] == [{"column": session.column, "direction": 1}]
    collector: str = rr.escape_entity_path_part("property:capture:collector")
    for layout in ("cards/fields", "table/columns"):
        assert logged[(f"/table/layouts/{layout}/{collector}", "visible")] == [False]
    assert ("/table", "sort_by") not in saved("without.rbl", ["rerun_segment_id"]), "a raw capture has no session column yet"


def test_optimized_recording_compacts_columns_and_preserves_rows(tmp_path: Path) -> None:
    target = tmp_path / "compact.rrd"
    with atomic_recording(target, recording_id="compact", send_properties=False, optimize=True) as recording:
        for frame in range(40):
            recording.send_columns("/pose", indexes=[rr.TimeColumn("frame", sequence=[frame])], columns=rr.Scalars.columns(scalars=[float(frame)]))
    chunks = [c for c in rrc.RrdReader(target).stream() if not c.is_static]
    assert len(chunks) == 1
    batch = chunks[0].to_record_batch()
    assert batch.column("frame").to_pylist() == list(range(40))
    assert batch.column("Scalars:scalars").to_pylist() == [[float(i)] for i in range(40)]
