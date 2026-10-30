"""Quality flags screen fit residuals, including absent observations."""

from pathlib import Path

import numpy as np
import rerun as rr
from rerun.chunk import RrdReader
from serde.json import from_json

from selfcap_soma.quality import SomaQuality, assess_quality, log_quality


def test_quality_pass_and_explicit_thresholds() -> None:
    points = np.zeros((2, 133, 3), dtype=np.float64)
    quality = assess_quality(points, points, points)
    assert quality.status == "pass"
    assert quality.body_median_mm == 0.0
    assert quality.shoulders_threshold_mm == 50.0
    assert quality.ears_threshold_mm == 30.0
    assert quality.body_threshold_mm == 40.0
    assert quality.hands_threshold_mm == 15.0


def test_quality_flags_bad_ears_and_missing_hands() -> None:
    points = np.zeros((2, 133, 3), dtype=np.float64)
    source = points.copy()
    source[:, 3:5, 0] = 0.031
    hands = np.full_like(points, np.nan)
    quality = assess_quality(points, source, hands)
    assert quality.status == "flag"
    assert quality.reasons == ("ears_above_threshold", "hands_missing")
    assert quality.hands_median_mm is None


def test_quality_uses_upper_body_and_jointly_finite_points() -> None:
    points = np.zeros((2, 133, 3), dtype=np.float64)
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
    points = np.zeros((2, 133, 3), dtype=np.float64)
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
