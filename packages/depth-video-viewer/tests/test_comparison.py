from pathlib import Path

import numpy as np
import rerun as rr
import rerun.chunk as rrc

from depth_video_viewer.apis.comparison import png_u16, read_depth_pngs


def test_depth_pngs_come_back_byte_exact_in_time_order(tmp_path: Path) -> None:
    pngs = [png_u16(np.full((4, 6), v, np.uint16)) for v in (7, 300, 1023)]
    times = np.array([20, 0, 10], np.int64)
    path = tmp_path / "depth.rrd"
    rec = rr.RecordingStream("depth_video_viewer_test")
    rec.save(path)
    rr.send_columns("cam/depth", indexes=[rr.TimeColumn("video_time", duration=times.astype("timedelta64[ns]"))],
                    columns=rr.EncodedDepthImage.columns(blob=pngs), recording=rec)  # fmt: skip
    rec.disconnect()
    read_times, read_pngs = read_depth_pngs(list(rrc.RrdReader(path).store().stream()), "/cam/depth", "video_time")
    assert read_times.tolist() == [0, 10, 20]
    assert read_pngs == [pngs[1], pngs[2], pngs[0]]
