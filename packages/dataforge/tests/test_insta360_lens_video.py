"""The Insta360 base's stored lens videos: their size, the camera model at that size, and the base as a sample source."""

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
    from dataforge.datasets.insta360_lens_video import lens_halvings

    assert [lens_halvings(width) for width in (3840, 2880, 1920, 1440)] == [1, 1, 0, 0]


def test_the_halved_camera_projects_each_ray_into_the_stored_pixel_whose_box_holds_it() -> None:
    """A box filter puts camera pixel ``p`` at ``(p + 0.5) / 2 - 0.5`` of the stored image: the rule ``halved_camera`` follows."""
    from dataforge.datasets.insta360_lens_video import halved_camera

    camera = x5_like_camera()
    full, valid = camera.project(rays())
    half, half_valid = halved_camera(camera, 1).project(rays())
    assert np.array_equal(valid, half_valid) and valid.all()
    np.testing.assert_allclose(half, (full + 0.5) / 2.0 - 0.5, atol=1e-9)
    # A ramp image box-filtered: stored pixel j holds camera coordinate 2j + 0.5, the same rule read backwards.
    ramp: Float64[ndarray, " n"] = np.arange(16, dtype=np.float64)
    np.testing.assert_allclose(ramp.reshape(-1, 2).mean(axis=1), 2.0 * np.arange(8) + 0.5)


def test_the_stored_camera_reads_back_as_the_halved_mei_model(tmp_path: Path) -> None:
    """``log_lens_camera`` writes what ``cameras_from_recording`` (the cube and panorama tables) reads: the halved model."""
    from dataforge import schema
    from dataforge.datasets.insta360 import RIG, cameras_from_recording
    from dataforge.datasets.insta360_lens_video import halved_camera, log_lens_camera

    camera = halved_camera(x5_like_camera(), 1)
    base: Path = tmp_path / "base.rrd"
    recording = rr.RecordingStream("insta360-test", recording_id="lens-camera")
    recording.save(base)
    for cam in range(2):
        recording.log(schema.cam_path(RIG, cam), rr.Transform3D(translation=[0.0, 0.0, 0.017 * cam], mat3x3=np.eye(3), relation=rr.TransformRelation.ChildFromParent), static=True)
        log_lens_camera(recording, schema.pinhole_path(RIG, cam), camera, LENS // 2, LENS // 2)
    recording.flush()
    recording.disconnect()
    lenses = cameras_from_recording(base)
    assert lenses.size == (LENS // 2, LENS // 2)
    stored = lenses.cameras[0]
    np.testing.assert_allclose([stored.fx, stored.fy, stored.cx, stored.cy], [camera.fx, camera.fy, camera.cx, camera.cy], rtol=1e-6)
    np.testing.assert_allclose(stored.project(rays())[0], camera.project(rays())[0], atol=2e-3)  # stored as float32


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
    from dataforge import schema
    from dataforge.datasets.insta360_lens_video import BaseLensVideos

    flags: list[bool] = [True, False, False, True, False]
    base = write_lens_videos(tmp_path / "base.rrd", [[0, 33, 66, 100, 133]] * 2, [flags, flags])
    source = BaseLensVideos(base, (schema.video_path(0, 0), schema.video_path(0, 1)))
    assert (source.codec, source.num_frames) == ("av1", 5)
    np.testing.assert_array_equal(source.times_ns, [0, 33, 66, 100, 133])
    np.testing.assert_array_equal(source.gops(1), [[0, 3], [3, 5]])
    data, bounds, irap = source.read_packed(1, 2, 4)
    assert data.tobytes() == bytes([1, 2, 1, 3])
    np.testing.assert_array_equal(bounds, [0, 2, 4])
    np.testing.assert_array_equal(irap, [False, True])


def test_lens_videos_on_different_stamps_or_without_a_first_keyframe_are_refused(tmp_path: Path) -> None:
    from dataforge import schema
    from dataforge.datasets.insta360_lens_video import BaseLensVideos

    entities: tuple[str, str] = (schema.video_path(0, 0), schema.video_path(0, 1))
    shifted = write_lens_videos(tmp_path / "shifted.rrd", [[0, 33], [0, 34]], [[True, False]] * 2)
    with pytest.raises(ValueError, match="same stamps"):
        BaseLensVideos(shifted, entities)
    late = write_lens_videos(tmp_path / "late.rrd", [[0, 33]] * 2, [[False, True]] * 2)
    with pytest.raises(ValueError, match="keyframe"):
        BaseLensVideos(late, entities)
