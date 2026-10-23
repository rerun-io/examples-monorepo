"""``RerunTyroConfig.open_segment``: an App that reads a catalog segment writes into the segment's own recording."""

from pathlib import Path

import pytest
import rerun as rr
import rerun.blueprint as rrb
from dagkit.sources import SegmentRef
from rerun.chunk import RrdReader

from simplecv.rerun_log_utils import RerunTyroConfig


def test_a_saved_run_holds_only_the_segment_s_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The file opens together with the segment's URL: the viewer merges the two because the ids are the segment's."""
    monkeypatch.setattr(SegmentRef, "recording_ids", lambda self: ("dataset-id", self.segment))
    save: Path = tmp_path / "out.rrd"
    config = RerunTyroConfig(application_id="calibrate", save=save)

    config.open_segment(SegmentRef("ds", "seg0"))
    config.rec_stream.log("/world/p", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    config.rec_stream.flush()
    config.rec_stream.disconnect()

    assert [(s.application_id, s.recording_id) for s in RrdReader(save).recordings()] == [("dataset-id", "seg0")]


@pytest.mark.integration
def test_the_viewer_shows_the_segment_and_the_outputs_as_one_recording(tmp_path: Path) -> None:
    import time

    from rerun.experimental import ViewerClient

    base = rr.RecordingStream("base", recording_id="seg0", send_properties=False)
    base.save(tmp_path / "base.rrd")
    base.log("/world/base", rr.Points3D([[1.0, 0.0, 0.0]]), static=True)
    base.flush()
    base.disconnect()
    executable: Path = Path(rr.__file__).parents[1] / "rerun_cli" / "rerun"
    with (
        rr.server.Server(host="127.0.0.1", datasets={"ds": [str(tmp_path / "base.rrd")]}) as server,
        ViewerClient.spawn(headless=True, port=9897, hide_welcome_screen=True, executable_path=str(executable)) as viewer,
    ):
        segment = SegmentRef("ds", "seg0", catalog_url=server.url())
        config = RerunTyroConfig(application_id="calibrate", connect=True, port=9897)
        config.open_segment(segment)
        config.rec_stream.log("/world/outputs", rr.Points3D([[0.0, 1.0, 0.0]]), static=True)
        config.rec_stream.send_blueprint(rrb.Blueprint(rrb.Spatial3DView(origin="/world")))  # as every App does: makes it active
        config.rec_stream.flush()
        dataset_id, _ = segment.recording_ids()
        deadline: float = time.monotonic() + 30.0
        while (stores := [r.store_id for r in viewer.viewer_state().recordings]) != [f"Recording:{dataset_id}:seg0"] and time.monotonic() < deadline:
            time.sleep(0.5)
        owner: str | None = viewer.viewer_state().url  # the active recording's source (rerun 0.38: 0.40 has no such field)
        config.rec_stream.disconnect()  # before the viewer goes
        assert stores == [f"Recording:{dataset_id}:seg0"]  # the segment and the outputs, no empty placeholder beside them
        assert owner == segment.viewer_url()  # the catalog owns it, so it fetches the segment's chunks (our stream would read /proxy)
