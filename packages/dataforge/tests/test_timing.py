"""Timing persistence at the public JSONL boundary."""

import subprocess
from pathlib import Path

import pytest
import rerun as rr

from dataforge import paths, writing
from dataforge.apis import convert
from dataforge.datasets.robocap import RobocapConfig, RobocapDataset
from dataforge.identity import SequenceIdentity
from dataforge.timing import ConvertRecord, RegisterRecord, SequenceTimer, append_record, load_records


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
    assert load_records(path, ConvertRecord) == [record, record]
    assert set(record.stage_s) == {"fetch"}
    assert record.total_s >= record.stage_s["fetch"] >= 0.0


def test_convert_records_written_and_skipped_sequences(tmp_path: Path, monkeypatch) -> None:
    identity = SequenceIdentity("robocap", ("a",))
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(RobocapDataset, "discover", lambda self: [(identity, tmp_path)])

    def fake_convert(self, identity, source, *, force):
        target = paths.rrd_path(tmp_path, layer="base", identity=identity)
        if not writing.should_skip(target, force=force):
            with self.timer.stage("write:base"), writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
                rr.send_columns(
                    "/value",
                    indexes=[rr.TimeColumn("video_time", duration=[2.0, 5.0])],
                    columns=rr.Scalars.columns(scalars=[0.0, 1.0]),
                    recording=recording,
                )
            self.timer.capture_s = 3.0
        return target

    monkeypatch.setattr(RobocapDataset, "convert", fake_convert)
    convert.main(convert.Config(dataset=RobocapConfig()))

    convert.main(convert.Config(dataset=RobocapConfig()))
    first, second = load_records(tmp_path / "timing/convert.jsonl", ConvertRecord)
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
    record = load_records(tmp_path / "timing/convert.jsonl", ConvertRecord)[0]
    assert record.error == "ValueError: broken source"
    assert not record.skipped
    assert record.capture_s is None


def test_converter_version_fallback(monkeypatch) -> None:
    for error in (FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")):
        def fail(*_args, error=error, **_kwargs):
            raise error

        monkeypatch.setattr(subprocess, "check_output", fail)
        assert convert.converter_version() == "2"



def test_register_record_round_trip(tmp_path: Path) -> None:
    record = RegisterRecord("sample", "2026-09-25T00:00:00+00:00", {"base": 1.0}, 2.0, 3, 4.0, "host")
    path = tmp_path / "register.jsonl"
    append_record(path, record)
    assert load_records(path, RegisterRecord) == [record]


def test_converter_without_capture_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    identity = SequenceIdentity("robocap", ("unreported",))
    monkeypatch.setattr(RobocapDataset, "discover", lambda self: [(identity, tmp_path)])

    def fake_convert(self, identity, source, *, force):
        target = self.targets(identity)["base"]
        with writing.atomic_recording(target, recording_id=identity.recording_id):
            pass
        return target

    monkeypatch.setattr(RobocapDataset, "convert", fake_convert)
    convert.main(convert.Config(dataset=RobocapConfig()))
    record = load_records(tmp_path / "timing/convert.jsonl", ConvertRecord)[0]
    assert not record.skipped
    assert record.capture_s is None


def test_one_scene_prefetch_overlaps_and_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from threading import Event

    started = Event()
    first_done = Event()
    prefetched = Event()
    identities = [SequenceIdentity("robocap", (name,)) for name in ("a", "b", "c")]
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(RobocapDataset, "discover", lambda self: [(i, tmp_path) for i in identities])

    def prefetch(self: RobocapDataset, identity: SequenceIdentity, source: Path, *, force: bool) -> None:
        assert force
        if identity == identities[1]:
            started.set()
            assert first_done.wait(3)
            raise OSError("prefetch failed")
        assert identity == identities[2]
        prefetched.set()

    def write(self: RobocapDataset, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        if identity == identities[0]:
            assert started.wait(3), "next fetch did not overlap conversion"
            first_done.set()
        else:
            assert identity == identities[2]
            assert prefetched.is_set(), "conversion did not await fetch"
        target = self.targets(identity)["base"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"converted")
        return target

    monkeypatch.setattr(RobocapDataset, "prefetch", prefetch, raising=False)
    monkeypatch.setattr(RobocapDataset, "convert", write)
    with pytest.raises(SystemExit, match="1 of 3"):
        convert.main(convert.Config(dataset=RobocapConfig(), force=True))
    records = load_records(tmp_path / "timing/convert.jsonl", ConvertRecord)
    assert [r.error for r in records] == [None, "OSError: prefetch failed", None]


@pytest.mark.parametrize("failed", [False, True])
def test_conversion_waits_for_prefetch_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    from concurrent.futures import Future, ThreadPoolExecutor
    from threading import Event

    entered = Event()
    called = Event()
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    dataset = RobocapConfig().setup()
    identity = SequenceIdentity("robocap", ("a",))
    pending = Future()

    def write(self: RobocapDataset, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        called.set()
        target = self.targets(identity)["base"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"done")
        return target

    monkeypatch.setattr(RobocapDataset, "convert", write)

    def run() -> ConvertRecord:
        entered.set()
        return convert.convert_one(dataset, identity, tmp_path, force=False, version="test", prefetched=pending)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(run)
        try:
            assert entered.wait(2)
            assert not called.wait(0.1)
        finally:
            if failed:
                pending.set_exception(OSError("fetch failed"))
            else:
                pending.set_result(None)
        record = result.result(timeout=3)
    assert called.is_set() is not failed
    assert record.error == ("OSError: fetch failed" if failed else None)
    assert "fetch" in record.stage_s


@pytest.mark.parametrize("failed", [False, True])
def test_finished_prefetch_has_no_fetch_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool) -> None:
    from concurrent.futures import Future

    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path))
    dataset: RobocapDataset = RobocapDataset(RobocapConfig())
    identity: SequenceIdentity = SequenceIdentity("robocap", ("ready",))
    pending: Future[None] = Future()
    target: Path = dataset.targets(identity)["base"]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    monkeypatch.setattr(RobocapDataset, "convert", lambda self, identity, source, *, force: target)
    if failed:
        pending.set_exception(OSError("fetch failed"))
    else:
        pending.set_result(None)
    record: ConvertRecord = convert.convert_one(dataset, identity, tmp_path, force=False, version="test", prefetched=pending)
    assert "fetch" not in record.stage_s
    assert record.error == ("OSError: fetch failed" if failed else None)
