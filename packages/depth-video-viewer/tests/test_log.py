from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc

from depth_video_viewer.log import ARCHETYPE, DepthVideoProperties, log_depth_video, static_components


def test_static_components_use_the_viewer_descriptors() -> None:
    props = DepthVideoProperties(meter=500.0, colormap=rr.components.Colormap.Turbo, depth_range=(0.0, 1000.0), point_fill_ratio=0.5)
    descriptors = [b.component_descriptor() for b in static_components(props)]
    assert [d.component for d in descriptors] == [
        "DepthVideoStream:codec",
        "DepthVideoStream:meter",
        "DepthVideoStream:colormap",
        "DepthVideoStream:depth_range",
        "DepthVideoStream:point_fill_ratio",
    ]
    assert {d.archetype for d in descriptors} == {ARCHETYPE}
    assert descriptors[0].component_type == "rerun.components.VideoCodec"
    defaults = static_components(DepthVideoProperties(meter=1000.0))
    assert [b.component_descriptor().component for b in defaults] == ["DepthVideoStream:codec", "DepthVideoStream:meter"]


def test_sample_count_must_match_timestamps() -> None:
    rec = rr.RecordingStream("depth_video_viewer_test")
    with pytest.raises(ValueError, match="2 samples for 1 timestamps"):
        log_depth_video("depth", [b"a", b"b"], np.array([0], np.int64), DepthVideoProperties(meter=1000.0), recording=rec)


def test_timestamps_survive_an_rrd_round_trip(tmp_path: Path) -> None:
    # Nanosecond timestamps that a float64 in seconds cannot hold exactly.
    times = np.array([1_700_000_000_123_456_789, 1_700_000_000_156_790_123], np.int64)
    path = tmp_path / "times.rrd"
    rec = rr.RecordingStream("depth_video_viewer_test")
    rec.save(path)
    log_depth_video("depth", [b"a", b"b"], times, DepthVideoProperties(meter=1000.0), recording=rec)
    rec.disconnect()  # closes the file, footer included
    written = [
        t
        for chunk in rrc.RrdReader(path).store().stream()
        if "DepthVideoStream:sample" in (batch := chunk.to_record_batch()).schema.names
        for t in batch.column("video_time").cast("int64").to_pylist()
    ]
    assert written == times.tolist()
