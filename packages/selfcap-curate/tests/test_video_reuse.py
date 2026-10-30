"""Packet reuse keeps capture timestamps and exact aligned phone clocks."""

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
from dataforge.schema import video_path
from jaxtyping import Shaped
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.catalog_video import CatalogVideo

from selfcap_curate.catalog_io import Segment, native_phone_rate, read_catalog_hands, read_phone_videos
from selfcap_curate.records import CatalogHands, SelfcapLayout, TimeSync


def test_cached_phone_packets_keep_capture_clock_and_retime_without_copies(fixture_sync: Callable[..., TimeSync]) -> None:
    sync: TimeSync = fixture_sync(offsets_ms={"exo/A": 0.0000015, "exo/B": -0.0000025})
    times: Shaped[ndarray, "3"] = np.array([17, 33_333_350, 66_666_683], dtype="timedelta64[ns]")
    video: CatalogVideo = CatalogVideo(times, [np.array([1, 2], dtype=np.uint8)] * 3, [True, False, False], "av1")
    segment: Segment = Segment(
        Mock(spec=DatasetEntry), sync.episode_key.segment, SelfcapLayout(2, 3, ("A", "B")), {video_path(0, 0): video, video_path(1, 0): video}
    )
    captured: dict[str, CatalogVideo] = read_phone_videos(segment)
    assert captured["A"] is video
    aligned: dict[str, CatalogVideo] = read_phone_videos(segment, timeline="video_time", sync=sync)
    np.testing.assert_array_equal(aligned["A"].t_ns, [19, 33_333_352, 66_666_685])
    np.testing.assert_array_equal(aligned["B"].t_ns, [15, 33_333_348, 66_666_681])
    np.testing.assert_array_equal(video.t_ns, [17, 33_333_350, 66_666_683])
    assert aligned["A"].samples is video.samples
    assert aligned["A"].keyframes is video.keyframes
    assert aligned["A"].codec == video.codec
    assert native_phone_rate(segment) == 30.0


def test_cached_quest_samples_keep_capture_clock(fixture_sync: Callable[..., TimeSync]) -> None:
    sync: TimeSync = fixture_sync()
    hands: CatalogHands = CatalogHands(
        np.array([1.0, 2.0]), np.ones((2, 133, 3)), np.array([1.0, 2.0]), np.zeros((2, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    )
    segment: Segment = Segment(Mock(spec=DatasetEntry), sync.episode_key.segment, SelfcapLayout(2, 3, ("A", "B")), hands=hands)
    assert read_catalog_hands(segment) is hands
