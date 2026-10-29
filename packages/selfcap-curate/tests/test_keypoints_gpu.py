"""CPU contracts for GPU planning and reference-preserving AV1 filtering."""

import subprocess

import av
import numpy as np
import pytest
from simplecv.catalog_video import CatalogVideo

from selfcap_curate.keypoints_av1 import filter_video
from selfcap_curate.keypoints_gpu import nearest_indices, plan_frames


def test_nearest_frame_ties_choose_left():
    np.testing.assert_array_equal(nearest_indices(np.array([0.0, 0.25, 0.5]), np.array([-1.0, 0.125, 0.38, 1.0])), [0, 0, 2, 2])


def test_av1_skip_is_bit_identical_to_full_decode(tmp_path):
    source, filtered = tmp_path / "source.mp4", tmp_path / "filtered.mp4"
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

    def frames(path):
        with av.open(str(path)) as container:
            return {float(f.pts * f.time_base): f.to_ndarray(format="rgb24") for f in container.decode(video=0)}

    full = frames(source)
    wanted = np.asarray(sorted(full)[3::17])
    indices = filter_video(source, filtered, wanted)
    candidate = frames(filtered)
    assert len(candidate) < len(full)
    assert len(indices) == len(wanted)
    for stamp in wanted:
        np.testing.assert_array_equal(candidate[stamp], full[stamp])


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
