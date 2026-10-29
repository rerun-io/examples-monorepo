"""CPU contracts for GPU planning and the one-pass wanted-frame decode."""

import subprocess

import numpy as np
import pytest
import torch
from simplecv.catalog_video import CatalogVideo
from torchcodec.decoders import VideoDecoder

from selfcap_curate.keypoints_gpu import decode_wanted, nearest_indices, plan_frames


def test_nearest_frame_ties_choose_left():
    np.testing.assert_array_equal(nearest_indices(np.array([0.0, 0.25, 0.5]), np.array([-1.0, 0.125, 0.38, 1.0])), [0, 0, 2, 2])


def test_one_pass_decode_yields_wanted_frames_in_batches(tmp_path):
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

    full = VideoDecoder(str(source), dimension_order="NHWC", seek_mode="exact")
    wanted = np.array([0, 3, 3, 20, 49, 50], dtype=np.int64)
    batches = list(decode_wanted(VideoDecoder(str(source), dimension_order="NHWC", seek_mode="exact"), wanted, batch=4))
    assert [len(frames.data) for frames in batches] == [4, 2]
    assert torch.equal(torch.cat([frames.data for frames in batches]), full.get_frames_at(wanted.tolist()).data)
    np.testing.assert_allclose(torch.cat([frames.pts_seconds for frames in batches]).numpy(), wanted / 24.0, rtol=0.0, atol=1e-9)
    with pytest.raises(ValueError, match="sorted"):
        next(decode_wanted(full, np.array([3, 2]), batch=4))
    with pytest.raises(ValueError, match="inside"):
        next(decode_wanted(full, np.array([0, len(full)]), batch=4))


def test_plan_uses_common_half_open_clock_and_checks_missing_overlap():
    def video(times):
        return CatalogVideo(
            np.rint(np.array(times) * 1e9).astype("timedelta64[ns]"), [np.empty(0, dtype=np.uint8)] * len(times), [True] * len(times), "av1"
        )

    result = plan_frames({"a": video([0.1, 0.2, 0.3, 0.4]), "b": video([0.16, 0.26, 0.36])}, 10.0)
    np.testing.assert_array_equal(result.times, [0.2, 0.3, 0.4])
    np.testing.assert_array_equal(result.indices["a"], [1, 2, 3])
    np.testing.assert_array_equal(result.indices["b"], [0, 1, 2])
    with pytest.raises(ValueError, match="two synced"):
        plan_frames({"a": video([0.0, 1.0])}, 10.0)
    with pytest.raises(ValueError, match="empty synced overlap"):
        plan_frames({"a": video([0.0, 0.1]), "b": video([1.0, 1.1])}, 10.0)
