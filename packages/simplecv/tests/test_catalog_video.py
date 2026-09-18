"""Bulk catalog video reads preserve each camera's timestamps and sparse keyframes."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

from simplecv.catalog_video import CatalogVideo, read_catalog_videos  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("timestamp", [False, True])
@pytest.mark.parametrize("dense_flags", [False, True])
def test_bulk_video_read_keeps_asynchronous_cameras_separate(tmp_path: Path, timestamp: bool, dense_flags: bool) -> None:
    path: Path = tmp_path / "cameras.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for camera, times in enumerate(((10, 20, 30), (11, 20, 31))):
            rec.log(f"/cam{camera}", rr.VideoStream.from_fields(codec=rr.VideoCodec.H264), static=True)
            for index, stamp in enumerate(times):
                if timestamp:
                    rec.set_time("video_time", timestamp=np.datetime64(stamp, "ns"))
                else:
                    rec.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
                rec.log(f"/cam{camera}", rr.VideoStream.from_fields(sample=bytes([camera, index]), is_keyframe=index == 0 if dense_flags else (True if index == 0 else None)))
    with rr.server.Server(datasets={"test": [path]}) as server:
        videos: tuple[CatalogVideo, ...] = read_catalog_videos(server.client().get_dataset("test"), "segment", ("/cam0", "/cam1"), "video_time")
    np.testing.assert_array_equal(videos[0].t_ns, [10, 20, 30])
    np.testing.assert_array_equal(videos[1].t_ns, [11, 20, 31])
    assert [bytes(sample) for sample in videos[0].samples] == [b"\x00\x00", b"\x00\x01", b"\x00\x02"]
    assert [bytes(sample) for sample in videos[1].samples] == [b"\x01\x00", b"\x01\x01", b"\x01\x02"]
    assert videos[0].keyframes == videos[1].keyframes == [True, False, False]
    assert videos[0].codec == videos[1].codec == "h264"
    assert videos[0].times.dtype == np.dtype("datetime64[ns]" if timestamp else "timedelta64[ns]")
