"""The Insta360 base's stored lens videos: their size, the camera model at that size (insta360-io), and the base as a sample source."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray

LENS: int = 3840
"""The X5's lens size."""


def x5_like_camera():  # noqa: ANN201 - insta360_io is imported in the test, not at collection
    """A Mei lens shaped like an X5 lens, with distortion terms, off-centre."""
    import insta360_io

    return insta360_io.MeiCamera("mei_v6", 1838.4, 1840.1, 1925.3, 1911.7, [2.31, np.deg2rad(100.0), 0.012, -0.004, 0.0007] + [0.0] * 10)


def rays() -> Float64[ndarray, "n 3"]:
    """Directions across the front hemisphere and past it."""
    angles: Float64[ndarray, " n"] = np.deg2rad(np.linspace(0.0, 95.0, 20))
    return np.stack([np.sin(angles) * np.cos(3.0 * angles), np.sin(angles) * np.sin(3.0 * angles), np.cos(angles)], axis=1)


def test_a_lens_is_halved_until_it_is_at_most_1920_wide() -> None:
    import insta360_io

    from dataforge.datasets.insta360_lens_video import LENS_MAX_SIZE

    assert [insta360_io.lens_halvings(width, LENS_MAX_SIZE) for width in (3840, 2880, 1920, 1440)] == [1, 1, 0, 0]
    assert insta360_io.lens_halvings(3840, None) == 0, "without a stored size the reader keeps the track size"


def test_the_halved_camera_projects_each_ray_into_the_stored_pixel_whose_box_holds_it() -> None:
    """A box filter puts camera pixel ``p`` at ``(p + 0.5) / 2 - 0.5`` of the stored image: the rule ``halved_camera`` follows."""
    import insta360_io

    camera = x5_like_camera()
    full, valid = camera.project(rays())
    half, half_valid = insta360_io.halved_camera(camera, 1).project(rays())
    assert np.array_equal(valid, half_valid) and valid.all()
    np.testing.assert_allclose(half, (full + 0.5) / 2.0 - 0.5, atol=1e-9)
    # A ramp image box-filtered: stored pixel j holds camera coordinate 2j + 0.5, the same rule read backwards.
    ramp: Float64[ndarray, " n"] = np.arange(16, dtype=np.float64)
    np.testing.assert_allclose(ramp.reshape(-1, 2).mean(axis=1), 2.0 * np.arange(8) + 0.5)


def write_lens_videos(path: Path, stamps: list[list[int]], keyframes: list[list[bool]], codec: rr.VideoCodec = rr.VideoCodec.AV1) -> Path:
    """A base holding only two lens videos: sample ``k`` of a lens is the bytes ``[lens, k]``."""
    from dataforge import schema

    recording = rr.RecordingStream("insta360-test", recording_id="lens-videos")
    recording.save(path)
    for lens in range(2):
        entity: str = schema.video_path(0, lens)
        recording.log(entity, rr.VideoStream(codec=codec), static=True)
        times = np.asarray(stamps[lens], dtype=np.int64)
        recording.send_columns(
            entity,
            indexes=[rr.TimeColumn("video_time", duration=times.astype("timedelta64[ns]"))],
            columns=rr.VideoStream.columns(sample=[bytes([lens, k]) for k in range(times.size)], is_keyframe=keyframes[lens]),
        )
    recording.flush()
    recording.disconnect()
    return path


def test_the_base_serves_each_lens_s_samples_gops_and_stamps(tmp_path: Path) -> None:
    from simplecv.catalog_video import CatalogVideo

    from dataforge.datasets.insta360_lens_video import read_base_lens_videos

    flags: list[bool] = [True, False, False, True, False]
    base = write_lens_videos(tmp_path / "base.rrd", [[0, 33, 66, 100, 133]] * 2, [flags, flags])
    source = read_base_lens_videos(base)
    assert all(isinstance(video, CatalogVideo) for video in source.videos), "simplecv's zero-copy packets, no dataforge copy"
    assert [bytes(sample) for sample in source.videos[0].samples] == [bytes([0, k]) for k in range(5)]
    assert (source.codec, source.num_frames) == ("av1", 5)
    np.testing.assert_array_equal(source.times_ns, [0, 33, 66, 100, 133])
    np.testing.assert_array_equal(source.gops(1), [[0, 3], [3, 5]])
    data, bounds, irap = source.read_packed(1, 2, 4)
    assert data.tobytes() == bytes([1, 2, 1, 3])
    np.testing.assert_array_equal(bounds, [0, 2, 4])
    np.testing.assert_array_equal(irap, [False, True])


def test_lens_videos_on_different_stamps_or_without_a_first_keyframe_are_refused(tmp_path: Path) -> None:
    from dataforge.datasets.insta360_lens_video import read_base_lens_videos

    shifted = write_lens_videos(tmp_path / "shifted.rrd", [[0, 33], [0, 34]], [[True, False]] * 2)
    with pytest.raises(ValueError, match="same stamps"):
        read_base_lens_videos(shifted)
    late = write_lens_videos(tmp_path / "late.rrd", [[0, 33]] * 2, [[False, True]] * 2)
    with pytest.raises(ValueError, match="keyframe"):
        read_base_lens_videos(late)
