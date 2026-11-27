"""The catalog segment's raw Y planes, decoded a chunk at a time, are the planes of one decoder over the whole stream."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

from robocap_live.catalog_segment import y_planes  # noqa: E402

pytestmark = pytest.mark.integration


def test_chunked_y_planes_equal_the_whole_stream_planes(tmp_path: Path) -> None:
    from simplecv.catalog_video_reader import CatalogVideoReader, StreamPackets, iter_chunks, read_video_index
    from simplecv.testing import encode_h264, whole_column

    samples: list[tuple[bytes, bool]] = encode_h264(50, 7)
    path: Path = tmp_path / "camera.rrd"
    entity: str = "/world/rig_00/cam_00/pinhole/video"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        rec.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
        rec.send_columns(
            entity,
            indexes=[rr.TimeColumn("video_time", duration=(33_333_333 * np.arange(len(samples), dtype=np.int64)).astype("timedelta64[ns]"))],
            columns=rr.VideoStream.columns(sample=[sample for sample, _ in samples], is_keyframe=[key for _, key in samples]),
        )
    with rr.server.Server(datasets={"test": [path]}) as server:
        dataset = server.client().get_dataset("test")
        reader = CatalogVideoReader(dataset, "segment", read_video_index(dataset, "segment", [entity], "video_time"), "video_time", chunk_s=0.5)
        assert reader.num_chunks >= 3
        chunked: list[np.ndarray] = [plane.copy() for plane in y_planes((chunk.streams[0] for chunk in iter_chunks(reader, range(reader.num_chunks))), "h264", 2)]
        video = whole_column(dataset, "segment", [entity], "video_time")[0]
    # The replaced path: the whole stream as one MP4.
    whole: list[np.ndarray] = [plane.copy() for plane in y_planes(iter([StreamPackets(first=0, samples=video.samples, keyframes=video.keyframes)]), "h264", 2)]
    assert len(chunked) == len(whole) == len(samples)
    for ours, reference in zip(chunked, whole, strict=True):
        np.testing.assert_array_equal(ours, reference)
