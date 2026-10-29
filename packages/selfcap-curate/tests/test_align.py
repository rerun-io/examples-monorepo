"""CLI selection and write-boundary checks happen before model setup."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from rerun.catalog import DatasetEntry

import selfcap_curate.apis.align as align
from selfcap_curate import apis as api_inputs
from selfcap_curate.apis.align import Config, main


@pytest.mark.parametrize("segments", [(), ("dataforge-selfcap",), ("selfcap__2025-12-30__65d00748__ep001", "selfcap__2025-11-24__027d8215__ep002")])
def test_align_refuses_empty_invalid_or_mixed_session(segments: tuple[str, ...], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session"):
        main(Config(base_dir=tmp_path / "base", output_dir=tmp_path / "out", segment_ids=segments))
    assert not (tmp_path / "out").exists()


def test_align_key_file_cannot_write_into_source_tree(tmp_path: Path) -> None:
    keys = tmp_path / "keys.txt"
    keys.write_text("selfcap__2025-12-30__65d00748__ep001\n")
    with pytest.raises(ValueError):
        main(Config(base_dir=tmp_path, output_dir=tmp_path / "out", key_file=keys))


@pytest.mark.parametrize("failure", ["after_detection", "fit", "later_patch"])
def test_failed_session_is_unpublished_and_retryable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure, fixture_sync) -> None:

    record = fixture_sync()
    second = replace(record, key=record.key[:-3] + "003", episode="ep003")
    records = {r.key: r for r in (record, second)}
    base = tmp_path / "base"
    base.mkdir()
    for key in records:
        (base / f"selfcap__{key}.rrd").touch()
    monkeypatch.setattr(align, "pose_models", lambda: (None, None))
    monkeypatch.setattr(api_inputs, "CatalogClient", lambda _: SimpleNamespace(get_dataset=lambda _: Mock(spec=DatasetEntry)))
    monkeypatch.setattr(align, "open_segment", lambda *_args: SimpleNamespace(layout=None))
    monkeypatch.setattr(
        align,
        "read_episode",
        lambda *a, **_kw: SimpleNamespace(layout=SimpleNamespace(oak_rig=4), videos=[SimpleNamespace(t_ns=np.array([0, 60_000_000_000]))]),
    )
    monkeypatch.setattr(align, "read_session_clock", lambda *a: SimpleNamespace(session=record.session, start_s=0.0))
    monkeypatch.setattr(align, "measure_links", lambda *a, **_kw: record.links)
    monkeypatch.setattr(align, "read_catalog_hands", lambda *a, **_kw: None)
    monkeypatch.setattr(align, "load_quest_hands", lambda *a, **_kw: {})
    monkeypatch.setattr(align, "detect_episode", lambda *a, **_kw: {})
    failed = False

    def fit(*_args):
        nonlocal failed
        if failure == "fit" and not failed:
            failed = True
            raise ValueError("injected failure")
        return records

    def detections(path, record, layout, hands):
        nonlocal failed
        assert not output.exists()
        assert path.parent.parent.parent == output.parent
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"complete detections")
        if failure == "after_detection" and not failed:
            failed = True
            raise ValueError("injected failure")

    def patch(_base, _raw, record, *, output_path, mode):
        assert mode == "fresh"
        nonlocal failed
        assert not output.exists()
        output_path.write_bytes(b"complete layer")
        if failure == "later_patch" and record.key == second.key and not failed:
            failed = True
            raise ValueError("injected failure")

    monkeypatch.setattr(align, "fit_session", fit)
    monkeypatch.setattr(align, "write_detections_layer", detections)
    monkeypatch.setattr(align, "patch", patch)
    output = tmp_path / "out"
    config = Config(base, output, tuple(f"selfcap__{key}" for key in records))
    with pytest.raises(ValueError, match="injected failure"):
        main(config)
    assert not output.exists()
    assert not list(tmp_path.glob(".out.*"))
    main(config)
    assert {p.name for p in output.iterdir()} == {name for key in records for name in (f"selfcap__{key}.rrd", f"selfcap__{key}.json")} | {"sapiens2"}
    assert len(list((output / "sapiens2").glob("*.rrd"))) == 2
    assert not list(tmp_path.glob(".out.*"))


def test_align_refuses_an_existing_empty_output_before_models(tmp_path, fixture_sync):
    base, output = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    key = fixture_sync().key
    (base / f"selfcap__{key}.rrd").touch()
    output.mkdir()
    with pytest.raises(FileExistsError):
        main(Config(base, output, (key,)))
    assert list(output.iterdir()) == []
