"""The labeler's frame source: the 5 fps grid, frame sets across cameras, and CPU decode of a synthetic H.264 video."""

import socket
from urllib.parse import urlparse

import av
import numpy as np
import pytest
from simplecv.catalog_video_codec import wrap_mp4

from posekit.labeler.frames import CameraVideo, FrameSource, frameset_samples, grid_indices, nearest_indices
from posekit.labeler.schema import CameraInfo

FRAME_NS: int = 33_344_444
"""RoboCap's measured frame period (slightly over 1/30 s, so a naive every-6th-frame grid drifts)."""
LEVEL_STEP: int = 4
"""Grey level step between synthetic frames: frame ``i`` is uniformly ``4 * i``."""


def synthetic_video(cam: int, n_frames: int, offset_ns: int = 0) -> CameraVideo:
    """A 64x48 H.264 camera whose frame ``i`` is uniformly grey ``4 * i`` (GOP 10, no B-frames)."""
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 64, 48, "yuv420p"
    codec.options = {"g": "10", "bf": "0", "qp": "4"}
    samples: list[bytes] = []
    keyframes: list[bool] = []
    frames = [av.VideoFrame.from_ndarray(np.full((48, 64, 3), LEVEL_STEP * i, np.uint8), format="rgb24").reformat(format="yuv420p") for i in range(n_frames)]
    for index, frame in enumerate(frames):
        frame.pts = index
        for packet in codec.encode(frame):
            samples.append(bytes(packet))
            keyframes.append(packet.is_keyframe)
    for packet in codec.encode(None):
        samples.append(bytes(packet))
        keyframes.append(packet.is_keyframe)
    times = (np.arange(n_frames, dtype=np.int64) * FRAME_NS + 1_000_000_000 + offset_ns).astype(np.int64)
    info = CameraInfo(cam=cam, name=f"cam{cam}", width=64, height=48)
    return CameraVideo(info, times, wrap_mp4(samples, keyframes, 30, codec="h264"))


def test_nearest_indices_respects_tolerance() -> None:
    times = np.array([0, 100, 200], dtype=np.int64)
    hits = nearest_indices(times, np.array([-5, 49, 51, 160, 260], dtype=np.int64), 50)
    assert hits.tolist() == [0, 0, 1, 2, -1]


def test_grid_follows_period_without_drift_and_skips_gaps() -> None:
    times = np.arange(300, dtype=np.int64) * FRAME_NS
    # A 3 s gap: frames 150.. move 3 s later.
    times[150:] += 3_000_000_000
    grid = grid_indices(times, 200_000_000, 50_000_000)
    grid_times = times[grid]
    steps = np.diff(grid_times)
    assert np.all(np.abs(steps[steps < 1_000_000_000] - 200_000_000) <= FRAME_NS)
    assert ((grid_times > times[149]) & (grid_times < times[150])).sum() == 0
    # Two runs of 150 frames (~5 s each) at 5 fps.
    assert 48 <= len(grid) <= 52


def test_frameset_time_is_the_latest_member_and_missing_cameras_are_minus_one() -> None:
    reference = synthetic_video(0, 61)
    later = synthetic_video(1, 61, offset_ns=20_000)
    short = synthetic_video(2, 20)
    grid_ns, samples = frameset_samples([reference, later, short], 0, 200_000_000, 10_000_000)
    assert len(grid_ns) == 11
    np.testing.assert_array_equal(grid_ns, later.times_ns[samples[1]])
    assert samples[0].tolist() == samples[1].tolist()
    assert (samples[2][samples[0] >= 20] == -1).all()


def test_frame_source_decodes_the_frame_set_member() -> None:
    source = FrameSource("seg", [synthetic_video(0, 61), synthetic_video(1, 61, offset_ns=20_000)], reference_cam=0, cache_frames=2)
    assert len(source.times_ns) == 11
    assert source.keyframes_ns == source.times_ns[::5]
    assert [cam.cam for cam in source.cams] == [0, 1]
    for column in (0, 4, 10, 7):
        t_ns = source.times_ns[column]
        expected = LEVEL_STEP * int(source.grid_samples[1, column])
        rgb = source.frame(1, t_ns)
        assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8
        assert abs(float(rgb.mean()) - expected) < 3
    assert len(source._cache) == 2
    # Off-grid times take the nearest frame; far-away times have none.
    off_grid = int(source.videos[0].times_ns[33]) + 1_000
    assert abs(float(source.frame(0, off_grid).mean()) - LEVEL_STEP * 33) < 3
    with pytest.raises(KeyError):
        source.frame(0, 0)
    info = source.segment_info("eval", "rerun+http://x/dataset/y")
    assert info.times_ns == source.times_ns and info.split == "eval"


def test_iter_grid_walks_every_grid_frame_in_order() -> None:
    source = FrameSource("seg", [synthetic_video(0, 61), synthetic_video(1, 50)], reference_cam=0)
    batches = list(source.iter_grid(batch_size=4, device="cpu"))
    assert [batch.cam for batch in batches] == [0, 0, 0, 1, 1, 1]
    walked = [int(t) for batch in batches if batch.cam == 1 for t in batch.times_ns]
    assert walked == [t for t, sample in zip(source.times_ns, source.grid_samples[1].tolist(), strict=True) if sample >= 0]
    first = batches[0]
    levels = first.frames_rgb.float().mean(dim=(1, 2, 3)).numpy()
    np.testing.assert_allclose(levels, LEVEL_STEP * source.grid_samples[0, :4], atol=3)
    capped = list(source.iter_grid([0], batch_size=8, device="cpu", max_frames=3))
    assert len(capped) == 1 and len(capped[0].times_ns) == 3


CATALOG_URL: str = "rerun+http://127.0.0.1:9897"
SEGMENT: str = "robocap__f408193e6447b3b0__s00000016"


def _catalog_reachable(url: str) -> bool:
    parsed = urlparse(url.replace("rerun+", ""))
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1.0):
            return True
    except OSError:
        return False


@pytest.mark.integration
def test_frame_source_from_the_labeling_catalog() -> None:
    if not _catalog_reachable(CATALOG_URL):
        pytest.skip(f"labeling catalog {CATALOG_URL} (dataset robocap-labeling) is not reachable")
    from rerun.catalog import CatalogClient

    dataset = CatalogClient(CATALOG_URL).get_dataset("robocap-labeling")
    source = FrameSource.from_catalog(dataset, SEGMENT)
    assert [cam.name for cam in source.cams] == ["left_front", "right_front", "left_eye", "right_eye", "left", "right"]
    duration_s = (source.times_ns[-1] - source.times_ns[0]) / 1e9
    assert abs(len(source.times_ns) - (5 * duration_s + 1)) <= 3
    assert len(source.keyframes_ns) == (len(source.times_ns) + 4) // 5 + (1 if (len(source.times_ns) - 1) % 5 else 0)
    assert source.keyframes_ns[-1] == source.times_ns[-1]
    rgb = source.frame(2, source.keyframes_ns[3])
    assert rgb.shape == (1080, 1920, 3) and rgb.std() > 5


def grid_only_camera(n_frames: int) -> CameraVideo:
    """A camera with frame times only (no video): enough for the grid, never decoded."""
    times = (np.arange(n_frames, dtype=np.int64) * FRAME_NS + 1_000_000_000).astype(np.int64)
    return CameraVideo(CameraInfo(cam=0, name="cam0", width=64, height=48), times, b"")


def test_the_last_grid_time_is_a_keyframe() -> None:
    # 61 frames at 30 fps -> 11 grid times: keyframes 0, 5, 10 already end on the last one.
    source = FrameSource("seg", [grid_only_camera(61)], reference_cam=0)
    assert len(source.times_ns) == 11 and source.keyframes_ns == source.times_ns[::5]
    # 73 frames -> 13 grid times: times 11 and 12 lie after keyframe 10, so the last time closes the run.
    source = FrameSource("seg", [grid_only_camera(73)], reference_cam=0)
    assert len(source.times_ns) == 13
    assert source.keyframes_ns == [source.times_ns[i] for i in (0, 5, 10, 12)]
