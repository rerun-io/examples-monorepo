import fcntl
import json
import subprocess
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pytest
from dataforge.quest_joint_poses import QuestJointPoses
from datafusion import SessionContext
from rerun.catalog import DatasetSelection
from rerun.chunk import RrdReader
from test_quality import small_fit
from test_xlo import REST

from selfcap_soma import batch, overlay, xlo
from selfcap_soma.batch import Config, run_segment, selected_segments
from selfcap_soma.catalog import Observations, RigPaths, SegmentCatalog
from selfcap_soma.fitting import Calibration
from selfcap_soma.overlay import read_layer_crop
from selfcap_soma.quality import read_quality
from selfcap_soma.shape_frames import ShapeFrames

SEGMENT = "selfcap__2025-12-03__d31bf1da__ep001"


def test_segment_list_is_ordered_deduplicated_and_rejects_path_injection(tmp_path: Path) -> None:
    path = tmp_path / "list.txt"
    path.write_text("selfcap__2025-12-03__d31bf1da__ep001\n# comment\n")
    assert selected_segments(["selfcap__2025-12-03__d31bf1da__ep001"], path) == ["selfcap__2025-12-03__d31bf1da__ep001"]
    with pytest.raises(ValueError, match="segment"):
        selected_segments(["../../other"], None)


def test_missing_body_exits_with_documented_input_limitation(monkeypatch, capsys):
    from selfcap_soma import batch
    from selfcap_soma.shape_frames import InsufficientBodyObservation

    def unavailable(config, segment):
        raise InsufficientBodyObservation("no usable torso frames")

    monkeypatch.setattr(batch, "run_segment", unavailable)
    config = batch.Config(catalog_url="test", dataset="test", segment=["selfcap__2025-12-04__5cb1e14c__ep001"])
    with pytest.raises(SystemExit) as raised:
        batch.main(config)
    assert raised.value.code == 20
    assert "SOMA_INPUT_LIMITATION insufficient_body_observation" in capsys.readouterr().err


def test_protected_dataset_is_rejected_in_local_mode_too(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="protected dataset"):
        run_segment(Config(catalog_url="test", dataset="dataforge-selfcap", output=tmp_path), SEGMENT)
    assert not list(tmp_path.iterdir())


def test_main_fails_nonzero_when_a_segment_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=f"SOMA batch failed for bad/name/{SEGMENT}.*simple name"):
        batch.main(Config(catalog_url="test", dataset="bad/name", output=tmp_path, segment=[SEGMENT]))


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> list[list[str] | str]:
    """Replace catalog inputs and fitting; any transport call is a test failure."""
    calls: list[list[str] | str] = []
    table = pa.table({"video_time:end": pa.array([timedelta(seconds=1)], pa.duration("ns")), "property:calib:session": ["a"]})
    view = Mock(spec=DatasetSelection)
    view.segment_table.return_value.to_arrow_table.return_value = table
    dataset = SimpleNamespace(
        filter_segments=lambda _segment: view, register=lambda uris, **_kwargs: calls.append(uris[0]) or SimpleNamespace(wait=lambda: None)
    )
    monkeypatch.setattr(batch, "CatalogClient", lambda _url: SimpleNamespace(get_dataset=lambda _name: dataset))
    data = Observations(
        np.array([0.0, 0.04]),
        np.zeros((2, 133, 3)),
        np.ones((2, 133)),
        np.zeros((2, 133, 3)),
        np.zeros((2, 3)),
        np.tile(np.eye(3), (2, 1, 1)),
        0.0,
    )
    monkeypatch.setattr(batch, "open_segment", lambda *_args: SegmentCatalog(dataset.filter_segments(SEGMENT), [], RigPaths(("/world/rig_00",), "/world/rig_04", "/world/rig_05")))
    monkeypatch.setattr(batch, "read_window", lambda *_args: data)
    data.hand_poses.update({side: QuestJointPoses(np.empty(0, dtype=np.int64), np.empty((0, 26, 3), dtype=np.float32), None) for side in ("left", "right")})
    monkeypatch.setattr(batch, "select_shape_frames", lambda *_args, **_kwargs: ShapeFrames(np.array([0, 1], dtype=np.int64), False))
    calibration = Calibration(
        np.zeros((1, 128), dtype=np.float32),
        1.0,
        np.zeros(3, dtype=np.float32),
        np.ones((1, 60), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        "ears-rotation",
    )
    fitted = SimpleNamespace(
        identity=calibration.identity,
        scale=1.0,
        head_offset=calibration.head_offset,
        bone_scales=calibration.bone_scales,
        head_attachment=calibration.head_attachment,
        head_anchor="ears-rotation",
        peak_bytes=0,
        joints=None,
        names=(),
    )
    monkeypatch.setattr(batch, "calibrate_shape", lambda *_args, **_kwargs: fitted)
    monkeypatch.setattr(batch, "fit_pose", lambda *_args, **_kwargs: fitted)
    monkeypatch.setattr(batch, "read_orientations", lambda *_args: SimpleNamespace(sources=("none", "none")))
    monkeypatch.setattr(batch, "write_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(batch, "layer_fit", lambda *_args: None)
    monkeypatch.setattr(batch, "group_jitter", lambda *_args: [])
    monkeypatch.setattr(batch, "write_layer", lambda _segment, _dataset, _url, _source, output, _fit, **_kwargs: output.write_bytes(b"layer"))

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_all_output_stays_local(tmp_path: Path, transport: list[list[str] | str]) -> None:
    run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), SEGMENT)
    directory = tmp_path / "test" / SEGMENT
    assert transport == []
    assert (directory / f"{SEGMENT}.rrd").read_bytes() == b"layer"
    assert json.loads((directory / "checkpoint.json").read_text())["phase"] == "export"
    assert not (directory / "fit.rrd").exists()


def test_each_segment_prints_one_time_line_with_its_stage_seconds(tmp_path: Path, transport: list[list[str] | str], capsys: pytest.CaptureFixture[str]) -> None:
    run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), SEGMENT)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("SOMA_TIME")]
    assert len(lines) == 1
    words = lines[0].split()
    assert words[1] == SEGMENT
    assert [word.split("=")[0] for word in words[2:]] == [field.name for field in fields(batch.Timings)]
    assert {"catalog_s", "shape_s", "fit_s", "layer_s", "total_s"} <= {word.split("=")[0] for word in words[2:]}


def test_old_format_checkpoint_starts_over(tmp_path: Path, transport: list[list[str] | str]) -> None:
    directory = tmp_path / "test" / SEGMENT
    directory.mkdir(parents=True)
    (directory / "checkpoint.json").write_text(json.dumps({"signature": "old", "phase": "calibrate_shape", "timings": {"write_s": 1.0, "mesh_overlay_s": 1.0}}))
    run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), SEGMENT)
    assert json.loads((directory / "checkpoint.json").read_text())["phase"] == "export"


def test_resume_reuses_the_layer_after_export(tmp_path: Path, transport: list[list[str] | str], monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config(catalog_url="test", dataset="test", output=tmp_path)
    run_segment(config, SEGMENT)
    monkeypatch.setattr(batch, "calibrate_shape", lambda *_args, **_kwargs: pytest.fail("a finished segment must not refit"))
    run_segment(config, SEGMENT)
    (tmp_path / "test" / SEGMENT / f"{SEGMENT}.rrd").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing/corrupt"):
        run_segment(config, SEGMENT)


def test_stale_stage_files_go_when_a_layer_is_written(tmp_path: Path, transport: list[list[str] | str]) -> None:
    directory = tmp_path / "test" / SEGMENT
    directory.mkdir(parents=True)
    (directory / "tmpabc.rrd.tmp").write_bytes(b"killed writer")
    run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), SEGMENT)
    assert not list(directory.glob("*.rrd.tmp"))


def test_a_locked_segment_fails_fast_and_another_segment_runs_beside_it(tmp_path: Path, transport: list[list[str] | str]) -> None:
    other = "selfcap__2025-12-04__5cb1e14c__ep001"
    locked = tmp_path / "test" / SEGMENT
    locked.mkdir(parents=True)
    with (locked / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), SEGMENT)
        run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), other)
    assert [path.name for path in locked.iterdir()] == [".lock"]
    assert (tmp_path / "test" / other / f"{other}.rrd").is_file()
    assert sorted(path.name for path in (tmp_path / "test").iterdir()) == sorted([SEGMENT, other])


def test_real_layer_from_memory_feeds_xlo_with_no_fit_recording(tmp_path: Path, transport: list[list[str] | str], monkeypatch: pytest.MonkeyPatch) -> None:
    data, result = small_fit()
    monkeypatch.undo()
    monkeypatch.setattr(batch, "CatalogClient", lambda _url: SimpleNamespace(get_dataset=lambda _name: _dataset()))
    monkeypatch.setattr(batch, "read_window", lambda *_args: data)
    data.hand_poses.update({side: QuestJointPoses(np.empty(0, dtype=np.int64), np.empty((0, 26, 3), dtype=np.float32), None) for side in ("left", "right")})
    monkeypatch.setattr(batch, "open_segment", lambda *_args: SegmentCatalog(_dataset().filter_segments(SEGMENT), [], RigPaths(("/world/rig_00",), "/world/rig_04", "/world/rig_05")))
    monkeypatch.setattr(batch, "select_shape_frames", lambda *_args, **_kwargs: ShapeFrames(np.array([0, 1], dtype=np.int64), False))
    monkeypatch.setattr(batch, "calibrate_shape", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(batch, "fit_pose", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(batch, "read_orientations", lambda *_args: SimpleNamespace(sources=("none", "none")))
    monkeypatch.setattr(batch, "group_jitter", lambda *_args: [])
    monkeypatch.setattr(overlay, "CatalogClient", lambda _url: SimpleNamespace(get_dataset=lambda _name: None))
    monkeypatch.setattr(overlay, "read_views", lambda *_args: ([], None))
    run_segment(Config(catalog_url="test", dataset="test", output=tmp_path), SEGMENT)
    directory = tmp_path / "test" / SEGMENT
    layer = directory / f"{SEGMENT}.rrd"
    assert not (directory / "fit.rrd").exists()
    assert layer.stat().st_mode & 0o777 == 0o644
    assert list(read_layer_crop(RrdReader(layer))) == [0, 1, 2]
    assert read_quality(layer).body_median_mm == pytest.approx(10.0)
    target = xlo.write_segment(xlo.Config(soma_dir=tmp_path, dataset="test", assets=tmp_path, output_dir=tmp_path / "xlo", segment=[SEGMENT]), REST, SEGMENT)
    assert target.is_file()


def _dataset() -> SimpleNamespace:
    table = pa.table({"video_time:end": pa.array([timedelta(seconds=1)], pa.duration("ns")), "property:calib:session": ["a"]})
    view = Mock(spec=DatasetSelection)
    view.segment_table.return_value.to_arrow_table.return_value = table
    return SimpleNamespace(filter_segments=lambda _segment: view)


def test_changed_camera_calibration_invalidates_a_completed_checkpoint(tmp_path: Path, transport, monkeypatch) -> None:
    config = Config(catalog_url="test", dataset="test", output=tmp_path)
    catalog = batch.open_segment(None, SEGMENT)
    column = "/world/rig_00/cam_00/pinhole:Pinhole:image_from_camera"
    catalog = SegmentCatalog(catalog.view, [column], catalog.rigs)
    monkeypatch.setattr(batch, "open_segment", lambda *_args: catalog)
    table = pa.table({"video_time": pa.array([0], pa.duration("ns")), column: [[[1.0] * 9]]})
    catalog.view.filter_contents.return_value.reader.side_effect = lambda **_kwargs: SessionContext().from_arrow(table)
    run_segment(config, SEGMENT)
    checkpoint = tmp_path / "test" / SEGMENT / "checkpoint.json"
    before = json.loads(checkpoint.read_text())["signature"]
    table = pa.table({"video_time": pa.array([0], pa.duration("ns")), column: [[[2.0] * 9]]})
    run_segment(config, SEGMENT)
    assert json.loads(checkpoint.read_text())["signature"] != before


def test_checkpoint_accepts_a_phone_without_camera_centers(tmp_path: Path, transport, monkeypatch) -> None:
    """A union catalog schema may name evidence absent from this segment or index."""
    config = Config(catalog_url="test", dataset="test", output=tmp_path)
    catalog = batch.open_segment(None, SEGMENT)
    lens = "/world/rig_00/cam_00/pinhole:Pinhole:image_from_camera"
    missing = "/world/rig_00/cam_00/pinhole/camera_centers:Points2D:positions"
    centers = "/world/rig_01/cam_00/pinhole/camera_centers:Points2D:positions"
    catalog = SegmentCatalog(catalog.view, [lens, missing, centers], catalog.rigs)
    monkeypatch.setattr(batch, "open_segment", lambda *_args: catalog)
    static = pa.table({lens: [[[1.0] * 9]]})
    temporal = pa.table({"video_time": pa.array([0], pa.duration("ns")), centers: [[[10.0, 20.0]]]})
    catalog.view.filter_contents.return_value.reader.side_effect = lambda *, index: SessionContext().from_arrow(static if index is None else temporal)

    run_segment(config, SEGMENT)

    checkpoint = tmp_path / "test" / SEGMENT / "checkpoint.json"
    before = json.loads(checkpoint.read_text())
    assert before["phase"] == "export"
    temporal = pa.table({"video_time": pa.array([0], pa.duration("ns")), centers: [[[30.0, 20.0]]]})
    run_segment(config, SEGMENT)
    assert json.loads(checkpoint.read_text())["signature"] != before["signature"]
