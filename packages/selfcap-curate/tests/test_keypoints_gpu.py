"""CPU contracts for GPU planning and the wanted-frame decode."""

import subprocess

import av
import numpy as np
import pytest
import rerun as rr
import torch
from simplecv.catalog_video_reader import open_decoders, select_decode_device
from torchcodec.decoders import VideoDecoder
from video_stubs import TimesOnly

from selfcap_curate.keypoints_gpu import decode_wanted, nearest_indices, plan_frames, pose_boxes, top_slots


def test_nearest_frame_ties_choose_left():
    np.testing.assert_array_equal(nearest_indices(np.array([0.0, 0.25, 0.5]), np.array([-1.0, 0.125, 0.38, 1.0])), [0, 0, 2, 2])


@pytest.mark.integration
def test_wanted_frames_decode_from_the_catalog_in_batches(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=24:duration=3",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "35",
            "-g",
            "48",
            "-threads",
            "1",
            str(source),
        ],
        check=True,
    )

    with av.open(str(source)) as container:
        packets = [(bytes(packet), bool(packet.is_keyframe)) for packet in container.demux(video=0) if packet.size]
    recording = tmp_path / "phone.rrd"
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(recording)
        rec.log("/phone/video", rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
        rec.send_columns(
            "/phone/video",
            indexes=[rr.TimeColumn("capture_time", duration=(41_666_667 * np.arange(len(packets), dtype=np.int64)).astype("timedelta64[ns]"))],
            columns=rr.VideoStream.columns(sample=[sample for sample, _ in packets], is_keyframe=[key for _, key in packets]),
        )
    device: str = select_decode_device()
    full = VideoDecoder(str(source), device=device, dimension_order="NHWC", seek_mode="exact")
    wanted = np.array([0, 3, 3, 20, 49, 50], dtype=np.int64)
    with rr.server.Server(datasets={"test": [recording]}) as server:
        video = open_decoders(server.client().get_dataset("test"), "segment", ["/phone/video"], "capture_time", torch.device(device), chunk_s=1.0)[0]
        batches = list(decode_wanted(video, wanted, batch=4))
        with pytest.raises(ValueError, match="sorted"):
            next(decode_wanted(video, np.array([3, 2]), batch=4))
        with pytest.raises(ValueError, match="inside"):
            next(decode_wanted(video, np.array([0, len(video)]), batch=4))
    assert [len(frames) for frames in batches] == [4, 2]
    assert batches[0].device.type == device
    assert torch.equal(torch.cat(batches), full.get_frames_at(wanted.tolist()).data)


def test_plan_uses_common_half_open_clock_and_checks_missing_overlap():
    def plan(**times):
        aligned = {phone: np.rint(np.array(t) * 1e9).astype(np.int64) for phone, t in times.items()}
        return plan_frames({phone: TimesOnly(t) for phone, t in aligned.items()}, aligned, 10.0)

    result = plan(a=[0.1, 0.2, 0.3, 0.4], b=[0.16, 0.26, 0.36])
    np.testing.assert_array_equal(result.times, [0.2, 0.3, 0.4])
    np.testing.assert_array_equal(result.indices["a"], [1, 2, 3])
    np.testing.assert_array_equal(result.indices["b"], [0, 1, 2])
    with pytest.raises(ValueError, match="two synced"):
        plan(a=[0.0, 1.0])
    with pytest.raises(ValueError, match="empty synced overlap"):
        plan(a=[0.0, 0.1], b=[1.0, 1.1])


def test_pose_boxes_cover_confident_keypoints_and_flag_lost_tracks():
    xy = torch.zeros((2, 133, 2))
    xy[0, :10] = torch.tensor([[100.0, 200.0], [300.0, 600.0]]).repeat(5, 1)
    scores = torch.zeros((2, 133))
    scores[0, :10] = 0.9
    scores[1, :5] = 0.9  # five confident keypoints: below the tracking minimum
    xy[0, 50] = torch.tensor([5000.0, 5000.0])  # an unconfident far point must not stretch the box
    boxes, box_scores, alive = pose_boxes(xy, scores, width=1920, height=1080)
    np.testing.assert_allclose(boxes[0].numpy(), [75.0, 150.0, 325.0, 650.0])  # centre (200, 400), half (125, 250) x1.25
    assert box_scores[0].item() == pytest.approx(0.9)
    assert alive.tolist() == [True, False]


def test_top_slots_rank_detections_within_each_frame():
    frames = torch.tensor([0, 1, 0, 0, 0, 1])
    scores = torch.tensor([0.5, 0.9, 0.8, 0.5, 0.95, 0.3])
    assert top_slots(frames, scores).tolist() == [2, 0, 1, 3, 0, 1]
