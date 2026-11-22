"""The shared catalog layer writer and registrar: atomic files, get-or-create datasets, layer registration and properties."""

from pathlib import Path

import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")
pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")
import rerun.chunk as rrc  # noqa: E402
from rerun.catalog import CatalogClient  # noqa: E402

from simplecv.catalog_layers import atomic_recording, atomic_write, ensure_dataset, register_layer  # noqa: E402


def _write_layer(path: Path, group: str, value: str, *, send_properties: bool) -> Path:
    with atomic_recording(path, application_id="simplecv-test", recording_id="segment", send_properties=send_properties) as recording:
        recording.send_property(group, rr.AnyValues(value=value))
        recording.send_property("shared", rr.AnyValues(value=value))
        recording.log(f"/{group}", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    return path


def test_a_clean_exit_publishes_a_readable_world_readable_file(tmp_path: Path) -> None:
    target: Path = _write_layer(tmp_path / "out" / "segment.rrd", "masks", "a", send_properties=False)
    assert target.stat().st_mode & 0o777 == 0o644
    assert rrc.RrdReader(target).store().schema() is not None
    assert sorted(p.name for p in target.parent.iterdir()) == ["segment.rrd"]


def test_a_failed_write_leaves_no_partial_target_and_keeps_an_old_one(tmp_path: Path) -> None:
    target: Path = tmp_path / "segment.rrd"
    with pytest.raises(RuntimeError, match="boom"), atomic_recording(target, application_id="simplecv-test", recording_id="segment") as recording:
        recording.log("/points", rr.Points3D([[0.0, 0.0, 0.0]]))
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []

    _write_layer(target, "masks", "old", send_properties=False)
    old: bytes = target.read_bytes()
    with pytest.raises(RuntimeError, match="boom"), atomic_recording(target, application_id="simplecv-test", recording_id="segment") as recording:
        recording.log("/points", rr.Points3D([[1.0, 0.0, 0.0]]))
        raise RuntimeError("boom")
    assert target.read_bytes() == old
    assert list(tmp_path.iterdir()) == [target]


def test_an_exclusive_write_refuses_an_existing_target(tmp_path: Path) -> None:
    target: Path = tmp_path / "file.txt"
    target.write_text("kept")
    with pytest.raises(FileExistsError), atomic_write(target, exclusive=True):
        pass
    assert target.read_text() == "kept"


@pytest.mark.integration
def test_layers_register_and_merge_their_property_groups(tmp_path: Path) -> None:
    """A derived layer written with ``send_properties=False`` keeps the base's ``RecordingInfo``; distinct groups merge;
    a column two layers share takes the value of the layer registered last."""
    base: Path = _write_layer(tmp_path / "base.rrd", "capture", "base", send_properties=True)
    masks: Path = _write_layer(tmp_path / "masks.rrd", "masks", "masks", send_properties=False)
    with rr.server.Server() as server:
        client = server.client()
        only_base = ensure_dataset(client, "only-base")
        register_layer(only_base, [base], None)
        base_start: object = only_base.segment_table().to_arrow_table().to_pylist()[0]["property:RecordingInfo:start_time"]
        entry = ensure_dataset(client, "layers")
        assert ensure_dataset(client, "layers").name == entry.name  # get, not a second create
        register_layer(entry, [base], None)
        register_layer(entry, [masks], "masks")
        register_layer(entry, [masks], "masks")  # replace=True: a re-run replaces, no duplicate error
        with pytest.raises(Exception, match=r"(?i)duplicate|exist"):
            register_layer(entry, [masks], "masks", replace=False)
        row: dict[str, object] = entry.segment_table().to_arrow_table().to_pylist()[0]
    assert sorted(row["rerun_layer_names"]) == ["base", "masks"]  # pyrefly: ignore[bad-argument-type]
    assert row["property:capture:value"] == ["base"] and row["property:masks:value"] == ["masks"]
    assert row["property:shared:value"] == ["masks"]  # registered last
    assert row["property:RecordingInfo:start_time"] == base_start


@pytest.mark.integration
def test_ensure_dataset_retries_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with rr.server.Server() as server:
        client = server.client()
        create = client.create_dataset
        calls: list[str] = []

        def racing_create(name: str, *, exist_ok: bool = False):  # noqa: ANN202 — stands in for CatalogClient.create_dataset
            calls.append(name)
            if len(calls) == 1:
                raise RuntimeError("another worker is creating it")
            return create(name, exist_ok=exist_ok)

        monkeypatch.setattr("simplecv.catalog_layers.DATASET_RETRY_DELAY_S", 0.0)
        monkeypatch.setattr(CatalogClient, "create_dataset", lambda self, name, *, exist_ok=False: racing_create(name, exist_ok=exist_ok))
        assert ensure_dataset(client, "raced").name == "raced"
        assert calls == ["raced", "raced"]
