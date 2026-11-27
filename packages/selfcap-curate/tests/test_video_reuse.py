"""Video reuse keeps capture timestamps and exact aligned phone clocks."""

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from rerun.catalog import DatasetEntry
from video_stubs import TimesOnly

from selfcap_curate.catalog_io import Segment, aligned_phone_times, median_rate
from selfcap_curate.records import CatalogHands, SelfcapLayout, TimeSync


def test_cached_phone_videos_keep_capture_clock_and_retime(fixture_sync: Callable[..., TimeSync], monkeypatch: pytest.MonkeyPatch) -> None:
    sync: TimeSync = fixture_sync(offsets_ms={"exo/A": 0.0000015, "exo/B": -0.0000025})
    video = TimesOnly(np.array([17, 33_333_350, 66_666_683]))
    segment: Segment = Segment(
        Mock(spec=DatasetEntry), sync.episode_key.segment, SelfcapLayout(2, 3, ("A", "B"))
    )
    captured = {"A": video, "B": video}
    aligned = aligned_phone_times(captured, sync)
    np.testing.assert_array_equal(aligned["A"], [19, 33_333_352, 66_666_685])
    np.testing.assert_array_equal(aligned["B"], [15, 33_333_348, 66_666_681])
    np.testing.assert_array_equal(video.index.t_ns, [17, 33_333_350, 66_666_683])
    assert median_rate(captured) == 30.0
    from selfcap_curate import keypoints_gpu

    def reread(*_args, **_kwargs):
        pytest.fail("inference planning must reuse the indexed videos")

    monkeypatch.setattr(keypoints_gpu, "read_phone_videos", reread)
    plan = keypoints_gpu.plan_episode(segment, 30.0, sync, videos=captured)
    assert plan.videos["A"] is video and plan.videos["B"] is video



def test_median_rate_takes_the_middle_phone_of_the_videos_it_is_given() -> None:
    def video(spacing_ns: int) -> TimesOnly:
        return TimesOnly(np.array([0, spacing_ns, 2 * spacing_ns]))

    assert median_rate({"A": video(41_666_667), "B": video(33_333_333), "C": video(16_666_667)}) == 30.0


def test_loaded_quest_samples_are_passed_to_overlay_without_changing_capture_clock(fixture_sync: Callable[..., TimeSync]) -> None:
    sync: TimeSync = fixture_sync()
    hands: CatalogHands = CatalogHands(
        np.array([1.0, 2.0]), np.ones((2, 133, 3)), np.array([1.0, 2.0]), np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    )
    from selfcap_curate.keypoints import ego_quest_hands

    track = ego_quest_hands(hands, sync)
    assert track is not None
    np.testing.assert_array_equal(hands.times_s, [1.0, 2.0])
