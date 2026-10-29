"""Lag labels use the source-to-gyro correction sign."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from dataforge import schema
from dataforge.schema import CAPTURE_TIMELINE
from frozen_v2 import ROOT
from rerun.chunk import RrdReader
from simplecv.rerun_log_utils import RerunTyroConfig

from selfcap_curate.apis.sync_curves import Config, build_blueprint, gyro_zoom_center, main, measure_curves
from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_episode
from selfcap_curate.links import SpeedSeries
from selfcap_curate.records import read_frozen_sync


def test_known_shift_is_reported_against_gyro() -> None:
    times = np.arange(0.0, 16.0, 0.001)
    speed = 2.0 + np.sin(times**2) + 0.3 * np.cos(7.3 * times)
    curves = measure_curves((SpeedSeries(times - 0.04, speed), SpeedSeries(times, speed)), 1.0, 13.0, gyro_index=1)
    assert curves[0].lag_ms == pytest.approx(40.0, abs=0.1)
    assert curves[1].lag_ms == pytest.approx(0.0, abs=0.1)
    assert curves[0].normalized.mean() == pytest.approx(0.0, abs=1e-12)
    assert curves[0].normalized.std() == pytest.approx(1.0)


def test_synthetic_catalog_preserves_clock_shift(tmp_path: Path, private_server) -> None:
    """A catalog pose shifted 40 ms early reports +40 ms, corrected on video_time."""

    segment = "selfcap__2025-12-08__abcdef12__ep001"
    path = tmp_path / "synthetic.rrd"
    times = np.arange(0.0, 16.0, 0.01)
    angle = 0.4 * times + 0.025 * np.sin(times**2) + 0.01 * np.sin(7.3 * times)
    midpoint = (times[1:] + times[:-1]) / 2.0
    speed = np.abs(np.diff(angle) / np.diff(times))
    with rr.RecordingStream("synthetic", recording_id=segment) as recording:
        recording.save(path)
        for rig, kind in ((0, "exo"), (1, "exo"), (2, "exo"), (3, "ego"), (4, "quest")):
            recording.log(schema.rig_path(rig), rr.AnyValues(kind=kind, name=f"phone-{rig}"), static=True)
            entity = schema.video_path(rig, 0)
            recording.log(entity, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
            recording.send_columns(
                entity,
                indexes=[rr.TimeColumn(CAPTURE_TIMELINE, duration=[0.0, 16.0]), rr.TimeColumn("video_time", duration=[0.0, 16.0])],
                columns=rr.VideoStream.columns(sample=[[1], [2]], is_keyframe=[True, False]),
            )
        for rig, camera in ((3, 1), (4, 1)):
            entity = schema.video_path(rig, camera)
            recording.log(entity, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
            recording.send_columns(
                entity,
                indexes=[rr.TimeColumn(CAPTURE_TIMELINE, duration=[0.0, 16.0]), rr.TimeColumn("video_time", duration=[0.0, 16.0])],
                columns=rr.VideoStream.columns(sample=[[1], [2]], is_keyframe=[True, False]),
            )
        for rig, camera in ((3, 0), (3, 1), (4, 1)):
            recording.log(schema.video_path(rig, camera).removesuffix("/video"), rr.Pinhole(focal_length=100.0, resolution=[32, 32]), static=True)
        recording.send_columns(
            schema.gyro_path(3, 0),
            indexes=[rr.TimeColumn(CAPTURE_TIMELINE, duration=midpoint), rr.TimeColumn("video_time", duration=midpoint)],
            columns=rr.Scalars.columns(scalars=np.c_[speed, np.zeros((len(speed), 2))]).partition([3] * len(speed)),
        )
        quaternions = np.c_[np.zeros((len(times), 2)), np.sin(angle / 2.0), np.cos(angle / 2.0)]
        recording.send_columns(
            schema.rig_path(4),
            indexes=[rr.TimeColumn(CAPTURE_TIMELINE, duration=times - 0.04), rr.TimeColumn("video_time", duration=times)],
            columns=rr.Transform3D.columns(quaternion=quaternions),
        )
    with private_server({DATASET_NAME: [path]}) as server:
        handle = open_segment(server.client().get_dataset(DATASET_NAME), segment)
        for clock, expected in ((CAPTURE_TIMELINE, 40.0), ("video_time", 0.0)):
            inputs = read_episode(handle, timeline=clock)
            curves = measure_curves((inputs.pose, inputs.gyro), 1.0, 13.0, gyro_index=1)
            assert curves[0].lag_ms == pytest.approx(expected, abs=0.2)


@pytest.mark.integration
def test_real_segment_writes_five_curves_on_both_clocks(tmp_path: Path, private_catalog) -> None:
    key = "2025-12-08__cc808233__ep001"
    source = ROOT / "v2/sync" / f"{key}.json"
    if not source.is_file():
        pytest.skip(f"SelfCap sync reference absent: {source}")
    output = tmp_path / "curves.rrd"
    with private_catalog([read_frozen_sync(source)], tmp_path / "base", layers=False) as server:
        result = main(Config(f"selfcap__{key}", catalog_url=server.url(), rr_config=RerunTyroConfig(headless=True, save=output)))
    assert output.stat().st_size > 0
    assert set(result) == {"as_cut", "aligned"}
    assert all(len(curves) == 5 for curves in result.values())
    assert all(curve.lag_ms is not None for curve in result["aligned"])
    assert result["aligned"][3].lag_ms == pytest.approx(0.0, abs=0.1)


def test_zoom_uses_sharpest_gyro_peak_and_saves_fixed_range(tmp_path: Path) -> None:

    times = np.arange(0.0, 14.0, 0.001)
    # A taller broad peak must lose to the sharper, smaller peak.
    speed = 5.0 * np.exp(-(((times - 4.0) / 0.8) ** 2)) + 3.0 * np.exp(-(((times - 8.0) / 0.04) ** 2))
    gyro = SpeedSeries(times, speed)
    center = gyro_zoom_center(gyro, 1.0, 13.0)
    assert center == pytest.approx(8.0, abs=0.001)
    assert gyro_zoom_center(gyro, 1.04, 13.04) == center
    path = tmp_path / "zoom.rbl"
    build_blueprint("synthetic", center).save("test", path)
    reader = RrdReader(path)
    batches = [chunk.to_record_batch() for store in reader.blueprints() for chunk in reader.stream(store=store)]
    values = {
        name: batch.column(name).to_pylist()
        for batch in batches
        for name in batch.schema.names
        if name in ("VisibleTimeRanges:ranges", "ContainerBlueprint:row_shares")
    }
    np.testing.assert_allclose(values["ContainerBlueprint:row_shares"], [[0.3, 0.3, 0.4]])
    assert values["VisibleTimeRanges:ranges"] == [[{"timeline": "t", "range": {"start": 7500000000, "end": 8500000000}}]]
