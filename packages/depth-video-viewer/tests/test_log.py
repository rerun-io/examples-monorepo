import numpy as np
import pytest
import rerun as rr

from depth_video_viewer.log import ARCHETYPE, DepthVideoProperties, log_depth_video, static_components


def test_static_components_use_the_viewer_descriptors() -> None:
    batches = static_components(DepthVideoProperties(meter=500.0, depth_range=(0.0, 1000.0), point_fill_ratio=0.5))
    descriptors = [b.component_descriptor() for b in batches]
    assert [d.component for d in descriptors] == [
        "DepthVideoStream:codec",
        "DepthVideoStream:meter",
        "DepthVideoStream:colormap",
        "DepthVideoStream:depth_range",
        "DepthVideoStream:point_fill_ratio",
    ]
    assert {d.archetype for d in descriptors} == {ARCHETYPE}
    assert descriptors[0].component_type == "rerun.components.VideoCodec"


def test_sample_count_must_match_timestamps() -> None:
    rec = rr.RecordingStream("depth_video_viewer_test")
    with pytest.raises(ValueError, match="2 samples for 1 timestamps"):
        log_depth_video("depth", [b"a", b"b"], np.array([0], np.int64), DepthVideoProperties(meter=1000.0), recording=rec)
