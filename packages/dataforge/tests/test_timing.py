"""Timing persistence at the public JSONL boundary."""

import subprocess
from pathlib import Path

import pytest
import rerun as rr

from dataforge import paths, timing, writing
from dataforge.apis import convert
from dataforge.datasets.robocap import RobocapConfig, RobocapDataset
from dataforge.identity import SequenceIdentity
from dataforge.timing import ConvertRecord, SequenceTimer, append_record, load_convert_records, stage


def test_round_trip_and_accumulated_stages(tmp_path: Path) -> None:
    timer = SequenceTimer()
    with timer.stage("fetch"):
        pass
    with timer.stage("fetch"):
        pass
    record = ConvertRecord("show3d", "show3d__a", "1", timer.started_at, timer.stage_s, timer.total_s, 2.0, {"base": 12}, "host", False)
    path = tmp_path / "timing/convert.jsonl"
    append_record(path, record)
    append_record(path, record)
    assert load_convert_records(path) == [record, record]
    assert set(record.stage_s) == {"fetch"}
    assert record.total_s >= record.stage_s["fetch"] >= 0.0


def test_convert_records_written_and_skipped_sequences(tmp_path: Path, monkeypatch) -> None:
    identity = SequenceIdentity("robocap", ("a",))
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(RobocapDataset, "discover", lambda self: [(identity, tmp_path)])

    def fake_convert(self, identity, source, *, force):
        target = paths.rrd_path(tmp_path, layer="base", identity=identity)
        if not writing.should_skip(target, force=force):
            with stage("write:base"), writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
                rr.send_columns(
                    "/value",
                    indexes=[rr.TimeColumn("video_time", duration=[2.0, 5.0])],
                    columns=rr.Scalars.columns(scalars=[0.0, 1.0]),
                    recording=recording,
                )
        return target

    monkeypatch.setattr(RobocapDataset, "convert", fake_convert)
    convert.main(convert.Config(dataset=RobocapConfig()))

    def unexpected_read(path):
        raise AssertionError("skipped conversion must not read base")

    monkeypatch.setattr(convert, "capture_span", unexpected_read)
    convert.main(convert.Config(dataset=RobocapConfig()))
    first, second = load_convert_records(tmp_path / "timing/convert.jsonl")
    assert first.capture_s == 3.0
    assert second.capture_s is None
    assert first.layer_bytes["base"] > 0
    assert set(first.stage_s) == {"write:base"}
    assert not first.skipped
    assert second.skipped and second.layer_bytes == {} and second.stage_s == {}


def test_failed_conversion_has_a_timing_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(RobocapDataset, "discover", lambda self: [(SequenceIdentity("robocap", ("bad",)), tmp_path)])

    def fail(self, identity, source, *, force):
        raise ValueError("broken source")

    monkeypatch.setattr(RobocapDataset, "convert", fail)
    with pytest.raises(SystemExit, match="1 of 1"):
        convert.main(convert.Config(dataset=RobocapConfig()))
    record = load_convert_records(tmp_path / "timing/convert.jsonl")[0]
    assert record.error == "ValueError: broken source"
    assert not record.skipped
    assert record.capture_s is None


def test_converter_version_fallback(monkeypatch) -> None:
    for error in (FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")):
        def fail(*_args, error=error, **_kwargs):
            raise error

        monkeypatch.setattr(subprocess, "check_output", fail)
        assert convert.converter_version() == "1"


def test_record_adds_to_active_timer() -> None:
    timing.record("transcode", 9.0)
    with timing.sequence_timer() as timer:
        timing.record("transcode", 2.0)
        timing.record("transcode", 3.0)
    assert timer.stage_s == {"transcode": 5.0}
