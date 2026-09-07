from pathlib import Path

import pyarrow as pa
import pytest
import rerun as rr

from dataforge.writing import atomic_recording, atomic_write, recording_to, should_skip


def property_columns(target: Path) -> set[str]:
    """Every ``property:*`` column of a saved rrd, read through the public reader."""
    store: rr.experimental.ChunkStore = rr.experimental.ChunkStore.from_chunks(list(rr.experimental.RrdReader(target).stream()))
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
    chunks: list[rr.experimental.Chunk] = list(rr.experimental.RrdReader(staged).stream())
    assert any(chunk.entity_path == "/world" for chunk in chunks), "the closed recording is readable"
