"""Tests for the ``RerunTyroConfig.__post_init__`` sink selection logic.

The ``live`` + ``save`` combination must fan out to both a spawned viewer and a
``.rrd`` file via ``set_sinks`` (the realtime-logging use case), while every
pre-existing mode (save-only, plain spawn, connect, serve, headless) keeps its
prior behavior. All Rerun side effects are spied so no viewer is launched.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import rerun as rr
from rerun.chunk import RrdReader

from simplecv.rerun_log_utils import RerunTyroConfig


@pytest.fixture
def rr_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "spawn": [],
        "set_sinks": [],
        "save": [],
        "connect_grpc": [],
        "serve_grpc": [],
        "grpc_sink": [],
        "file_sink": [],
    }
    monkeypatch.setattr(rr, "init", lambda **kw: None)
    # __post_init__ stores this in a `rr.RecordingStream`-typed attr (beartype-checked).
    monkeypatch.setattr(rr, "get_global_data_recording", lambda: MagicMock(spec=rr.RecordingStream))
    monkeypatch.setattr(rr, "spawn", lambda **kw: calls["spawn"].append(kw))
    monkeypatch.setattr(rr, "set_sinks", lambda *sinks, **kw: calls["set_sinks"].append(list(sinks)))
    monkeypatch.setattr(rr, "save", lambda path, *a, **kw: calls["save"].append(path))
    monkeypatch.setattr(rr, "connect_grpc", lambda *a, **kw: calls["connect_grpc"].append(kw))
    monkeypatch.setattr(rr, "serve_grpc", lambda *a, **kw: calls["serve_grpc"].append(kw))
    monkeypatch.setattr(rr, "serve_web_viewer", lambda *a, **kw: None)
    monkeypatch.setattr(rr, "GrpcSink", lambda url=None: calls["grpc_sink"].append(url) or ("grpc", url))
    monkeypatch.setattr(rr, "FileSink", lambda path: calls["file_sink"].append(path) or ("file", path))
    return calls


def test_live_and_save_fans_out_to_both_sinks(rr_spy: dict[str, list[Any]], tmp_path: Path) -> None:
    save = tmp_path / "out.rrd"
    RerunTyroConfig(save=save, live=True, port=9999)

    assert len(rr_spy["spawn"]) == 1
    assert rr_spy["spawn"][0]["connect"] is False
    assert rr_spy["spawn"][0]["port"] == 9999
    assert len(rr_spy["set_sinks"]) == 1
    assert rr_spy["grpc_sink"] == ["rerun+http://127.0.0.1:9999/proxy"]
    assert rr_spy["file_sink"] == [str(save)]
    assert rr_spy["save"] == []  # not the file-only path


def test_save_only_stays_file_only(rr_spy: dict[str, list[Any]], tmp_path: Path) -> None:
    save = tmp_path / "out.rrd"
    RerunTyroConfig(save=save, live=False)

    assert rr_spy["save"] == [save]
    assert rr_spy["set_sinks"] == []
    assert rr_spy["spawn"] == []


def test_live_without_save_spawns_plain_viewer(rr_spy: dict[str, list[Any]]) -> None:
    RerunTyroConfig(live=True, port=4321)  # no save -> ordinary viewer, no sinks

    assert len(rr_spy["spawn"]) == 1
    assert rr_spy["spawn"][0]["port"] == 4321
    assert "connect" not in rr_spy["spawn"][0]
    assert rr_spy["set_sinks"] == []


def test_headless_save_live_is_file_only(rr_spy: dict[str, list[Any]], tmp_path: Path) -> None:
    save = tmp_path / "out.rrd"
    RerunTyroConfig(save=save, live=True, headless=True)

    assert rr_spy["save"] == [save]
    assert rr_spy["set_sinks"] == []
    assert rr_spy["spawn"] == []


def test_new_fields_have_backwards_compatible_defaults() -> None:
    field_defaults = {f.name: f.default for f in RerunTyroConfig.__dataclass_fields__.values()}
    assert field_defaults["live"] is False
    assert field_defaults["port"] == 9876


def test_save_only_writes_few_large_chunks(tmp_path: Path) -> None:
    """A file holds a steady per-frame log in a few chunks, not one per 200 ms batcher tick: the viewer walks every
    chunk of a file it opens by path on every frame (re_viewer 0.42 prefetch), so tiny chunks cost frame time."""
    save = tmp_path / "out.rrd"
    config = RerunTyroConfig(application_id="tyro-file-batcher", save=save)
    for i in range(120):  # ~1.2 s of a 100 Hz log: six default ticks
        rr.set_time("frame", sequence=i)
        rr.log("/stats/value", rr.Scalars(float(i)))
        time.sleep(0.01)
    config.rec_stream.disconnect()

    assert sum(str(chunk.entity_path) == "/stats/value" for chunk in RrdReader(save).stream().to_chunks()) == 1
