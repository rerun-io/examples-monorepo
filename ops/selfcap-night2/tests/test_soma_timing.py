from soma_timing import timing_window


def test_timing_window_excludes_inflight_jobs_and_stops_at_target():
    events = [(100.0, 20.0, "inflight"), (120.0, 15.0, "a"), (150.0, 30.0, "b"), (180.0, 10.0, "c")]
    result = timing_window(events, 100.0, 2)
    assert result["mean_seconds"] == 22.5
    assert result["samples"] == 2
    assert result["complete"]
    assert result["segments"] == ["a", "b"]
