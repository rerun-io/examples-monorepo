"""The App on a real catalog segment window through the ONNX nets (never registered), and the Types through an in-process catalog.

The App test needs a catalog server holding the robocap dataset with base and slam_rs layers, and the ONNX models: set
``ROBOCAP_LIVE_CATALOG_URL``, ``ROBOCAP_LIVE_MODELS_DIR`` (detnet_full.onnx, keynet.onnx) and ``ROBOCAP_LIVE_SEGMENT``.
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from dagkit.sources import Column, RecordingSource, SegmentRef
from dagkit.time import STATIC
from dagkit.types import rerun_roundtrip
from datafusion import col
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.types.rig import RigCameras

from robocap_live.apis.hands_layer import Config, main
from robocap_live.nodes.hands import HandsParams
from robocap_live.types import RIG_ROOT, CapRig, CatalogPoses, LayerRun

pytestmark = pytest.mark.integration


def test_a_catalog_segment_window_becomes_an_unregistered_hands_layer(tmp_path: Path) -> None:
    catalog: str | None = os.environ.get("ROBOCAP_LIVE_CATALOG_URL")
    models: str | None = os.environ.get("ROBOCAP_LIVE_MODELS_DIR")
    segment: str | None = os.environ.get("ROBOCAP_LIVE_SEGMENT")
    if not (catalog and models and segment):
        pytest.skip("set ROBOCAP_LIVE_CATALOG_URL, ROBOCAP_LIVE_MODELS_DIR and ROBOCAP_LIVE_SEGMENT (catalog server + ONNX models)")
    if not (Path(models) / "detnet_full.onnx").is_file():
        pytest.skip(f"ONNX models absent from {models}")
    try:
        times = SegmentRef("robocap", segment, catalog).entry().filter_segments(segment).filter_contents(["/world/rig_00/cam_00/pinhole/video"])
        first: int = times.reader(index="video_time").select(col("video_time").cast(pa.int64())).to_arrow_table().column(0)[0].as_py()
    except ConnectionError as error:
        pytest.skip(f"catalog {catalog} unreachable: {error}")
    window = SegmentRef("robocap", segment, catalog, window_ns=(first, first + 1_000_000_000))
    report = main(Config(RerunTyroConfig(headless=True), window, tmp_path, HandsParams(Path(models), device="cpu"), register="new"))
    assert report.status == "complete" and report.nodes["hands"].ticks >= 25
    (published,) = (tmp_path / segment).glob("hands.window.*.rrd")  # a window never registers, even with register="new"
    assert published.stat().st_size > 0


def test_cap_rig_and_run_counts_round_trip_through_a_catalog(tmp_path: Path) -> None:
    rig = RigCameras(
        tuple(f"cam_{i:02d}" for i in range(6)),
        np.tile(np.diag([600.0, 600.0, 1.0]), (6, 1, 1)),
        np.tile(np.eye(4), (6, 1, 1)),
        np.tile(np.array([1920, 1080]), (6, 1)),
    )
    cap = CapRig(rig, ("left_front", "left_side", "left_back", "right_front", "right_side", "right_back"), np.full((6, 8), 0.125))
    ((t_ns, back),) = rerun_roundtrip(cap, RIG_ROOT, "video_time", STATIC)
    assert t_ns == STATIC and back.names == cap.names and back.rig.names == rig.names
    np.testing.assert_array_equal(back.fisheye62, cap.fisheye62)
    run = LayerRun(150, 149, 1, 133, 145, 120, 141, 0.9922, True, *[0.5] * 7)
    assert rerun_roundtrip(run, "/verify/hands/run", "video_time", STATIC) == [(STATIC, run)]


def test_poses_read_from_a_recording_the_way_slam_rs_writes_them(tmp_path: Path) -> None:
    path: Path = tmp_path / "slam_rs.rrd"
    recording = rr.RecordingStream("slam-rs", recording_id="segment", send_properties=False)
    recording.save(path)
    for i, t_ns in enumerate([3_000_000, 1_000_000, 2_000_000]):
        recording.set_time("video_time", duration=np.timedelta64(t_ns, "ns"))
        recording.log(RIG_ROOT, rr.Transform3D(translation=[float(i), 0.0, 0.0], quaternion=rr.Quaternion(xyzw=[0.0, 0.0, 0.0, 1.0])))
    recording.flush()
    recording.disconnect()
    (event,) = list(RecordingSource(path, {"poses": Column(CatalogPoses, RIG_ROOT)}).events())  # one STATIC table, sorted by time
    assert isinstance(event.value, CatalogPoses) and event.value.t_ns.tolist() == [1_000_000, 2_000_000, 3_000_000]
    assert event.value.world_from_rig[:, 0, 3].tolist() == [1.0, 2.0, 0.0]
