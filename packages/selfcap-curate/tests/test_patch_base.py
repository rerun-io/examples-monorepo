"""A synthetic served base migrates without changing unrelated chunk bytes."""

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from dataforge import schema
from dataforge.datasets.selfcap import log_quest_hands, read_quest_hands
from dataforge.logging_toolkit import capture_column
from dataforge.schema import CAPTURE_TIMELINE
from dataforge.writing import recording_to
from rerun.chunk import Chunk, LazyChunkStream, RrdReader
from rrd_fixtures import synthetic_served
from serde.json import from_json

from selfcap_curate.patch_base import patch
from selfcap_curate.records import TimeSync, read_sync, write_sync

FIXTURES = Path(__file__).parent / "fixtures"
VIDEO_FIXTURES = Path(__file__).parents[2] / "dataforge/tests/fixtures"


@pytest.mark.parametrize("quest_delta_ms", [None, 12.345])
def test_patch_preserves_unrelated_batches_and_moves_both_video_clocks(tmp_path: Path, quest_delta_ms: float | None) -> None:
    served, raw, sync, starts = synthetic_served(tmp_path)
    original_bytes = served.read_bytes()
    target = tmp_path / "patched.rrd"
    assert patch(served, raw, sync, output_path=target, quest_video_delta_ms=quest_delta_ms, mode="migrate-keep-clock") == target
    assert served.read_bytes() == original_bytes
    source = RrdReader(served)
    result = RrdReader(target)
    before = {chunk.to_record_batch().schema.metadata[b"rerun:id"]: chunk for chunk in source.stream()}
    seen: set[str] = set()
    hands_found = False
    patch_record: TimeSync | None = None
    for chunk in result.stream():
        entity = str(chunk.entity_path)
        batch = chunk.to_record_batch()
        chunk_id = batch.schema.metadata[b"rerun:id"]
        if entity == "/__properties/time_sync":
            before.pop(chunk_id, None)
            if "record_json" in batch.schema.names:
                patch_record = from_json(TimeSync, batch.column("record_json")[0].as_py()[0])
                assert batch.column("method")[0].as_py() == ["keep-this-field"]
            continue
        if entity == schema.coco133_xyz_path() and not chunk.is_static:
            assert batch.column(CAPTURE_TIMELINE).cast(pa.int64()).to_pylist() == [17, 27]
            shift = round((sync.offsets_ms["quest_pose"] - 48.0) * 1e6)
            assert batch.column("video_time").cast(pa.int64()).to_pylist() == [17 + shift, 27 + shift]
            hands_found = True
            continue
        if chunk_id not in before:
            assert entity in ("/world/gt", schema.coco133_xyz_path()) or (chunk.is_static and "raw_calibration_json" in batch.schema.names)
            continue
        original = before.pop(chunk_id).to_record_batch()
        if entity in starts and not chunk.is_static:
            seen.add(entity)
            for index in (CAPTURE_TIMELINE, "video_time"):
                extra = round((quest_delta_ms or 0.0) * 1e6) if entity.startswith("/world/rig_05/") and index == "video_time" else 0
                expected = np.asarray(original.column(index).cast(pa.int64())) + starts[entity] + extra
                np.testing.assert_array_equal(np.asarray(batch.column(index).cast(pa.int64())), expected)
            for name in original.schema.names:
                if name not in (CAPTURE_TIMELINE, "video_time"):
                    assert batch.column(name).equals(original.column(name))
        else:
            assert batch.serialize().to_pybytes() == original.serialize().to_pybytes(), entity
    assert not before
    assert seen == set(starts)
    assert hands_found
    assert patch_record is not None and patch_record.video_start_ms is not None
    assert len(patch_record.video_start_ms) == 9
    assert patch_record.video_start_ms["oak_rgb"] == pytest.approx(40.039062)
    assert patch_record.video_start_ms["oak_left"] == 0.0
    assert patch_record.video_start_ms["quest_left"] == 0.0
    assert patch_record.video_start_ms["quest_right"] == pytest.approx(40.039062)
    assert patch_record.hand_latency_ms == 48.0
    assert patch_record.quest_video_delta_ms == quest_delta_ms
    assert patch_record.offsets_ms == sync.offsets_ms
    original_blueprints = {
        chunk.to_record_batch().schema.metadata[b"rerun:id"]: chunk.to_record_batch().serialize().to_pybytes()
        for store in source.blueprints()
        for chunk in source.stream(store=store)
    }
    result_blueprints = {
        chunk.to_record_batch().schema.metadata[b"rerun:id"]: chunk.to_record_batch().serialize().to_pybytes()
        for store in result.blueprints()
        for chunk in result.stream(store=store)
    }
    assert result_blueprints == original_blueprints
    with pytest.raises(ValueError, match="already patched"):
        patch(target, raw, sync, output_path=tmp_path / "twice.rrd", mode="migrate-keep-clock")
    assert not (tmp_path / "twice.rrd").exists()


def test_patch_refuses_outputs_in_the_raw_tree(tmp_path: Path) -> None:
    sync = read_sync(FIXTURES / "2025-11-24__027d8215__ep002.sync.json")
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(ValueError, match="protected input"):
        patch(tmp_path / "missing.rrd", raw, sync, output_path=raw / "output.rrd", mode="migrate-keep-clock")
    assert not (raw / "output.rrd").exists()


@pytest.mark.parametrize("missing_video", [False, True])
def test_new_fit_record_aligns_a_fresh_base_without_readding_starts_or_hands(tmp_path: Path, missing_video: bool) -> None:
    """Fresh PR 2 input has presented capture_time and raw hands, but no time_sync."""

    served, raw, sync, starts = synthetic_served(tmp_path)
    fresh = tmp_path / "fresh.rrd"
    hands_path = tmp_path / "raw-hands.rrd"
    with recording_to(hands_path, recording_id=f"selfcap__{sync.key}", send_properties=False) as recording:
        log_quest_hands(recording, read_quest_hands(raw / "quest"))
        for entity in (schema.gyro_path(4, 0), schema.rig_path(5)):
            recording.send_columns(
                entity,
                indexes=[
                    capture_column(np.array([17, 27], dtype=np.int64)),
                    rr.TimeColumn("video_time", duration=np.array([17, 27], dtype="timedelta64[ns]")),
                ],
                columns=rr.Scalars.columns(scalars=[1.0, 2.0]),
            )

    def as_cut(chunk):
        if str(chunk.entity_path) == "/world/untouched" or (missing_video and str(chunk.entity_path) == schema.video_path(5, 0)):
            return []
        if str(chunk.entity_path) == "/__properties/time_sync":
            return []
        if chunk.is_static or str(chunk.entity_path) not in starts:
            return [chunk]
        batch = chunk.to_record_batch()
        for name in (CAPTURE_TIMELINE, "video_time"):
            index = batch.schema.get_field_index(name)
            times = np.asarray(batch.column(index).cast(pa.int64())) + starts[str(chunk.entity_path)]
            batch = batch.set_column(index, batch.schema.field(index), pa.array(times, type=pa.duration("ns")))
        return Chunk.from_record_batch(batch)

    LazyChunkStream.merge(RrdReader(served).stream().flat_map(as_cut), RrdReader(hands_path).stream()).write_rrd(
        fresh, application_id="dataforge", recording_id=f"selfcap__{sync.key}"
    )
    assert sync.offsets_ms is not None
    offsets = {**sync.offsets_ms, "oak_imu": -200.0, "quest_pose": 80.0, "quest_video": 100.0, "exo/Dragon-413A7CC1": -30.0}
    fitted = replace(
        sync,
        offsets_ms=offsets,
        oak_rgb_model=replace(sync.oak_rgb_model, a_ms=5.0, b_ms_per_s=0.1, t_start_session_s=10.0),
        version="selfcap-curate-cpu-v1",
    )
    record_path = tmp_path / "new-fit.json"
    write_sync(record_path, fitted)
    shutil.rmtree(raw)
    target = patch(fresh, None, read_sync(record_path), output_path=tmp_path / "fresh-aligned.rrd", mode="fresh")
    hand_rows = 0
    firsts = {}
    for chunk in RrdReader(target).stream():
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        entity = str(chunk.entity_path)
        capture = batch.column(CAPTURE_TIMELINE).cast(pa.int64()).to_pylist()
        video = batch.column("video_time").cast(pa.int64()).to_pylist()
        firsts.setdefault(entity, (capture[0], video[0]))
        if entity == schema.coco133_xyz_path():
            hand_rows += len(capture)
            assert capture == [17, 27]
            assert video == [32_000_017, 32_000_027]
    assert hand_rows == 2
    assert firsts[schema.video_path(4, 0)] == (40_039_062, 46_043_066)
    assert firsts[schema.video_path(5, 1)] == (40_039_062, 140_039_062)
    assert firsts[schema.video_path(0, 0)] == (0, -30_000_000)
    assert firsts[schema.gyro_path(4, 0)] == (17, -199_999_983)
    assert firsts[schema.rig_path(5)] == (17, 80_000_017)


def test_legacy_patch_keeps_each_factory_document_verbatim(tmp_path: Path) -> None:
    served, raw, sync, _ = synthetic_served(tmp_path)
    folders = sorted((raw / "exo").iterdir()) + [raw / "ego", raw / "quest"]
    documents = {}
    for rig, folder in enumerate(folders):
        content = f' {{"vendor": "{folder.name}"}}\r\n'
        (folder / "calibration.json").write_bytes(content.encode())
        documents[schema.rig_path(rig)] = content
    target = patch(served, raw, sync, output_path=tmp_path / "metadata.rrd", mode="migrate-keep-clock")
    found = {}
    for chunk in RrdReader(target).stream():
        batch = chunk.to_record_batch()
        if "raw_calibration_json" in batch.schema.names:
            assert chunk.is_static
            found[str(chunk.entity_path)] = batch.column("raw_calibration_json")[0].as_py()[0]
    assert found == documents


def test_unknown_temporal_entity_refused_when_aligning(tmp_path: Path) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    with pytest.raises(ValueError, match="unknown temporal entity"):
        patch(served, raw, replace(sync, version="new-fit"), output_path=tmp_path / "unknown.rrd", mode="migrate-realign")
    assert not (tmp_path / "unknown.rrd").exists()


def test_rebased_base_without_time_sync_is_refused_with_raw_metadata(tmp_path: Path) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    rebased = tmp_path / "rebased.rrd"
    LazyChunkStream.from_iter(
        chunk for chunk in RrdReader(served).stream() if str(chunk.entity_path) not in ("/__properties/time_sync", "/world/untouched")
    ).write_rrd(rebased, application_id="dataforge", recording_id=f"selfcap__{sync.key}")
    with pytest.raises(ValueError, match="start mismatch"):
        patch(rebased, raw, sync, output_path=tmp_path / "invalid.rrd", mode="fresh")


def test_aligning_updates_all_time_sync_properties(tmp_path: Path) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    clean = tmp_path / "clean.rrd"
    LazyChunkStream.from_iter(chunk for chunk in RrdReader(served).stream() if str(chunk.entity_path) != "/world/untouched").write_rrd(
        clean, application_id="dataforge", recording_id=f"selfcap__{sync.key}"
    )
    updated = replace(sync, method="new-method", flags=["new-flag"], aligned_clock="new-clock", version="new-fit")
    target = patch(clean, raw, updated, output_path=tmp_path / "aligned.rrd", mode="migrate-realign")
    properties = {
        name: batch.column(name)[0].as_py()[0]
        for chunk in RrdReader(target).stream().filter(content="/__properties/time_sync")
        for batch in [chunk.to_record_batch()]
        for name in batch.schema.names
        if name in ("flags", "method", "aligned_clock")
    }
    assert properties == {"flags": "new-flag", "method": "new-method", "aligned_clock": "new-clock"}


def test_duplicate_hand_timestamp_refused(tmp_path: Path) -> None:
    served, raw, sync, _ = synthetic_served(tmp_path)
    csv_path = raw / "quest/left_hand_poses.csv"
    csv_path.write_text(csv_path.read_text() + csv_path.read_text().splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="duplicate.*timestamp"):
        patch(served, raw, sync, output_path=tmp_path / "duplicate.rrd", mode="migrate-keep-clock")


def test_legacy_missing_video_has_no_start_metadata(tmp_path: Path) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    missing = tmp_path / "missing.rrd"
    LazyChunkStream.from_iter(chunk for chunk in RrdReader(served).stream() if str(chunk.entity_path) != schema.video_path(5, 0)).write_rrd(
        missing, application_id="dataforge", recording_id=f"selfcap__{sync.key}"
    )
    target = patch(missing, raw, sync, output_path=tmp_path / "patched.rrd", mode="migrate-keep-clock")
    record = next(
        from_json(TimeSync, chunk.to_record_batch().column("record_json")[0].as_py()[0])
        for chunk in RrdReader(target).stream().filter(content="/__properties/time_sync")
        if "record_json" in chunk.to_record_batch().schema.names
    )
    assert "quest_left" not in record.video_start_ms


def test_blueprint_activation_flag_preserved(tmp_path: Path) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    target = patch(served, raw, sync, output_path=tmp_path / "blueprint.rrd", mode="migrate-keep-clock")
    commands = []
    for path in (served, target):
        printed = subprocess.run([sys.executable, "-m", "rerun", "rrd", "print", str(path)], check=True, capture_output=True, text=True).stdout
        commands.append([line for line in printed.splitlines() if "BlueprintActivationCommand" in line])
    assert commands[0] and "make_active: false" in commands[0][0]
    assert commands[1] == commands[0]


def test_explicit_realign_rebuilds_properties_even_when_offsets_match(tmp_path: Path) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    clean = tmp_path / "clean.rrd"
    LazyChunkStream.from_iter(chunk for chunk in RrdReader(served).stream() if str(chunk.entity_path) != "/world/untouched").write_rrd(
        clean, application_id="dataforge", recording_id=f"selfcap__{sync.key}"
    )
    supplied = replace(sync, method="explicit-realign")
    target = patch(clean, raw, supplied, mode="migrate-realign", output_path=tmp_path / "realigned.rrd")
    properties = [c.to_record_batch() for c in RrdReader(target).stream() if str(c.entity_path) == "/__properties/time_sync"]
    assert any("method" in b.schema.names and b.column("method")[0].as_py() == ["explicit-realign"] for b in properties)


def test_explicit_restore_ignores_record_version_and_scans_source_once(tmp_path: Path, monkeypatch) -> None:

    served, raw, sync, _ = synthetic_served(tmp_path)
    original = RrdReader.stream
    calls = []

    def counted(reader, *args, **kwargs):
        calls.append(reader)
        return original(reader, *args, **kwargs)

    monkeypatch.setattr(RrdReader, "stream", counted)
    target = patch(served, raw, replace(sync, version="new-fit"), mode="migrate-keep-clock", output_path=tmp_path / "restored.rrd")
    # One inventory scan and one lazy writing pass; the third reader streams only the newly logged hands.
    assert len(calls) == 3
    assert calls[0] is calls[1]
    properties = [c.to_record_batch() for c in RrdReader(target).stream() if str(c.entity_path) == "/__properties/time_sync"]
    assert any("method" in b.schema.names and b.column("method")[0].as_py() == ["keep-this-field"] for b in properties)


def test_migration_parses_each_hand_csv_once(tmp_path: Path, monkeypatch) -> None:
    served, raw, sync, _ = synthetic_served(tmp_path)
    import pyarrow.csv as pacsv

    original = pacsv.read_csv
    opened = []

    def counted(path, *args, **kwargs):
        if Path(path).name.endswith("_hand_poses.csv"):
            opened.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pacsv, "read_csv", counted)
    patch(served, raw, sync, mode="migrate-keep-clock", output_path=tmp_path / "once.rrd")
    assert opened == [raw / "quest/left_hand_poses.csv", raw / "quest/right_hand_poses.csv"]
