"""The overview relays every camera's stored video chunk by chunk: the same samples, times and flags as a whole-column read."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

from simplecv.catalog_video import CatalogVideo  # noqa: E402
from simplecv.testing import whole_column  # noqa: E402

from colsfm.overview import relay_source_video  # noqa: E402

pytestmark = pytest.mark.integration

SOURCES: dict[str, tuple[int, int, int, bool]] = {
    # entity: (samples, GOP, time offset in ns, keyframe flags logged on keyframes only); 1 s apart, so several 30 s chunks
    "/world/rig_00/cam_00/pinhole/video": (95, 10, 0, True),
    "/world/rig_00/cam_01/pinhole/video": (88, 7, 4_000_000, False),
}


def _source(path: Path) -> None:
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        for camera, (entity, (count, gop, offset_ns, sparse)) in enumerate(SOURCES.items()):
            rec.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            times = (offset_ns + 1_000_000_000 * np.arange(count, dtype=np.int64)).astype("timedelta64[ns]")
            samples: list[bytes] = [bytes([camera, index % 256, index // 256]) * (1 + index % 5) for index in range(count)]
            rec.send_columns(entity, indexes=[rr.TimeColumn("video_time", duration=times)], columns=rr.VideoStream.columns(sample=samples))
            flags: np.ndarray = np.arange(count) % gop == 0
            rec.send_columns(
                entity,
                indexes=[rr.TimeColumn("video_time", duration=times[flags] if sparse else times)],
                columns=rr.VideoStream.columns(is_keyframe=(flags[flags] if sparse else flags).tolist()),
            )


def test_the_chunked_relay_logs_the_whole_column_relays_samples(tmp_path: Path) -> None:
    source_path: Path = tmp_path / "source.rrd"
    relayed_path: Path = tmp_path / "relayed.rrd"
    _source(source_path)
    targets: list[str] = [f"/world/colsfm/rig/cam_{camera:02d}/pinhole/video" for camera in range(len(SOURCES))]
    with rr.server.Server(datasets={"source": [source_path]}) as server:
        dataset = server.client().get_dataset("source")
        reference: tuple[CatalogVideo, ...] = whole_column(dataset, "segment", list(SOURCES), "video_time")
        with rr.RecordingStream("overview", recording_id="segment") as recording:
            recording.save(relayed_path)
            relay_source_video(dataset, "segment", list(SOURCES), targets, "video_time", recording)
    with rr.server.Server(datasets={"relayed": [relayed_path]}) as server:
        relayed: tuple[CatalogVideo, ...] = whole_column(server.client().get_dataset("relayed"), "segment", targets, "video_time")
    for got, want in zip(relayed, reference, strict=True):
        np.testing.assert_array_equal(got.times, want.times)
        assert [bytes(sample) for sample in got.samples] == [bytes(sample) for sample in want.samples]
        assert got.keyframes == want.keyframes
        assert got.codec == want.codec == "h264"
