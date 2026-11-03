"""Quality flags screen fit residuals, including absent observations."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rerun as rr
from rerun.chunk import RrdReader
from serde.json import from_json

from selfcap_soma import overlay
from selfcap_soma.catalog import Observations
from selfcap_soma.fitting import FitResult
from selfcap_soma.observations import coco_joint_names
from selfcap_soma.overlay import layer_fit, write_layer
from selfcap_soma.quality import SomaQuality, assess_quality, log_quality, read_quality


def test_quality_pass_and_explicit_thresholds() -> None:
    points = np.zeros((3, 133, 3), dtype=np.float64)
    points[:, 4, 0] = 0.15
    quality = assess_quality(points, points, points)
    assert quality.status == "pass"
    assert quality.body_median_mm == 0.0
    assert quality.shoulders_threshold_mm == 50.0
    assert quality.ears_threshold_mm == 30.0
    assert quality.body_threshold_mm == 40.0
    assert quality.hands_threshold_mm == 15.0


def test_quality_flags_bad_ears_and_missing_hands() -> None:
    points = np.zeros((3, 133, 3), dtype=np.float64)
    points[:, 4, 0] = 0.15
    source = points.copy()
    source[:, 3:5, 0] += 0.031
    hands = np.full_like(points, np.nan)
    quality = assess_quality(points, source, hands)
    assert quality.status == "flag"
    assert quality.reasons == ("ears_above_threshold", "hands_missing")
    assert quality.hands_median_mm is None


def test_quality_uses_upper_body_and_jointly_finite_points() -> None:
    points = np.zeros((3, 133, 3), dtype=np.float64)
    points[:, 4, 0] = 0.15
    source = points.copy()
    source[:, 13:23] = 100.0
    source[0, 5] = np.nan
    source[1, 5, 0] = 0.060
    source[:, 6, 0] = 0.060
    quality = assess_quality(points, source, points)
    assert quality.body_median_mm == 0.0
    assert quality.shoulders_median_mm == 60.0
    assert quality.reasons == ("shoulders_above_threshold",)


def test_quality_property_roundtrip(tmp_path: Path) -> None:
    points = np.zeros((3, 133, 3), dtype=np.float64)
    points[:, 4, 0] = 0.15
    quality = assess_quality(points, points, points)
    path = tmp_path / "quality.rrd"
    recording = rr.RecordingStream("quality-test")
    recording.save(path)
    log_quality(recording, quality)
    recording.flush()
    recording.disconnect()
    chunks = RrdReader(path).stream().filter(content="/__properties/soma_quality").to_chunks()
    assert chunks
    batch = chunks[0].to_record_batch()
    payload = batch.column("record_json")[0].as_py()[0]
    assert from_json(SomaQuality, payload) == quality
    assert batch.column("status")[0].as_py() == ["pass"]


def test_quality_reports_active_scale_bound_and_ear_fallback() -> None:
    points = np.zeros((3, 133, 3), dtype=np.float64)
    quality = assess_quality(points, points, points, scale=1.25)
    assert quality.scale_bound_active
    assert quality.head_torso_rotation_fallback
    assert quality.ear_span_rejected_frames == 3
    assert quality.status == "flag"


def test_read_quality_round_trips_the_written_layer_with_missing_metrics(tmp_path: Path) -> None:
    written = SomaQuality(12.5, 20.0, None, None, "flag", ("ears_missing", "hands_missing"), ear_span_surviving_frames=7, scale_bound_active=True)
    path = tmp_path / "layer.rrd"
    recording = rr.RecordingStream("quality-test", recording_id="segment")
    recording.save(path)
    log_quality(recording, written)
    recording.log("/world/other", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    recording.flush()
    recording.disconnect()
    read = read_quality(path)
    assert read == written
    assert read.ears_median_mm is None
    assert read.reasons == ("ears_missing", "hands_missing")
    assert read.ear_span_surviving_frames == 7


def test_read_quality_names_the_file_when_the_property_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "no-quality.rrd"
    recording = rr.RecordingStream("quality-test", recording_id="segment")
    recording.save(path)
    recording.log("/world/other", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    recording.flush()
    recording.disconnect()
    with pytest.raises(ValueError, match="no-quality.rrd.*soma_quality"):
        read_quality(path)


def small_fit(frames: int = 2) -> tuple[Observations, FitResult]:
    """Two fitted frames, 10 mm from the triangulated points, with no Quest hands."""
    named = list(dict.fromkeys(["LeftEye", "Head", *coco_joint_names().values()]))
    names = tuple(named + [f"Spare{index}" for index in range(77 - len(named))])
    times = np.array([0.0, 0.04], dtype=np.float64)[:frames]
    joints = np.zeros((frames, 77, 3), dtype=np.float32)
    vertices = np.zeros((frames, 4, 3), dtype=np.float32)
    data = Observations(
        times,
        np.full((frames, 133, 3), 0.01, dtype=np.float64) * np.array([1.0, 0.0, 0.0]),
        np.ones((frames, 133), dtype=np.float64),
        np.full((frames, 133, 3), np.nan, dtype=np.float64),
        np.zeros((frames, 3), dtype=np.float64),
        np.tile(np.eye(3, dtype=np.float64), (frames, 1, 1)),
        0.0,
    )
    result = FitResult(
        np.zeros((frames, 77, 3), dtype=np.float32),
        np.zeros((frames, 3), dtype=np.float32),
        np.zeros((1, 128), dtype=np.float32),
        1.0,
        joints,
        vertices,
        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        names,
        np.zeros(78, dtype=np.int64),
        0.1,
        0,
        np.zeros(frames, dtype=np.float64),
        np.eye(3, dtype=np.float32),
        np.zeros((frames, 2, 3), dtype=np.float32),
        "ears-rotation",
        np.zeros(3, dtype=np.float32),
        np.ones((1, 60), dtype=np.float32),
        upper_body_vertices=np.array([0, 1, 2], dtype=np.int64),
        visible_joints=np.arange(77, dtype=np.int64),
    )
    return data, result


def test_layer_export_keeps_quality_from_the_fit_and_writes_no_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data, result = small_fit()
    monkeypatch.setattr(overlay, "open_segment", lambda *_args: None)
    monkeypatch.setattr(overlay, "CatalogClient", lambda _url: SimpleNamespace(get_dataset=lambda _name: None))
    monkeypatch.setattr(overlay, "read_views", lambda *_args: ([], None))
    layer = tmp_path / "layer" / "segment.pending.rrd"
    write_layer("segment", "test", "test", "rtmw", layer, layer_fit(data, result))
    assert sorted(path.name for path in layer.parent.iterdir()) == ["segment.pending.rrd"]
    assert read_quality(layer).body_median_mm == pytest.approx(10.0)
