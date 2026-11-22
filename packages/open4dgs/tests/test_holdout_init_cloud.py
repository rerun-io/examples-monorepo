"""Held-out selection is pinned; an init cloud survives the .rrd round trip bit-exactly."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr

MAMMA_NAMES: tuple[str, ...] = tuple(f"rig_{index:02d}/cam_00" for index in range(1, 33))


def test_select_holdout_pins_mamma_seed0() -> None:
    from open4dgs.holdout import select_holdout

    train, holdout = select_holdout(tuple(reversed(MAMMA_NAMES)), num_holdout=3, seed=0)
    assert holdout == ("rig_25/cam_00", "rig_27/cam_00", "rig_03/cam_00")
    assert len(train) == 29 and not set(train) & set(holdout)
    assert list(train) == sorted(train)


def test_select_holdout_rejects_no_training_camera() -> None:
    from open4dgs.holdout import select_holdout

    with pytest.raises(ValueError):
        select_holdout(("a", "b"), num_holdout=2, seed=0)


def _cloud(with_optional: bool):
    from open4dgs.init_cloud import InitCloud

    rng = np.random.default_rng(3)
    frames = np.repeat(np.array([0, 3, 6], dtype=np.int32), [4, 5, 3])
    n = frames.size
    optional = (
        dict(
            durations=rng.random((n, 1), dtype=np.float32),
            has_velocity=rng.random(n) > 0.5,
            is_static=np.zeros(n, dtype=bool),
            dynamic_protect=rng.random(n) > 0.5,
            dynamic_allowed=np.ones(n, dtype=bool),
            stage_keep=np.ones(n, dtype=bool),
        )
        if with_optional
        else {}
    )
    return InitCloud(
        positions=rng.normal(size=(n, 3)).astype(np.float32),
        velocities=rng.normal(size=(n, 3)).astype(np.float32) * np.float32(1e-3),
        colors=rng.random((n, 3), dtype=np.float32),
        times=(frames / 6.0).astype(np.float32)[:, None],
        frames=frames,
        frame_start=0,
        frame_end=6,
        keyframe_step=3,
        fps=30.0,
        source_frame_start=99,
        dataset="ds",
        segment_id="seg",
        method="body_surface",
        **optional,
    )


@pytest.mark.parametrize("with_optional", [False, True])
def test_init_cloud_rrd_round_trip_is_bit_exact(tmp_path: Path, with_optional: bool) -> None:
    from open4dgs.init_cloud import log_init_cloud, read_init_cloud

    cloud = _cloud(with_optional)
    path = tmp_path / "init.rrd"
    recording = rr.RecordingStream("open4dgs-test", recording_id="init-cloud")
    recording.save(path)
    log_init_cloud(cloud, recording)
    recording.flush()
    del recording
    back = read_init_cloud(path)
    for name in (
        "positions",
        "velocities",
        "colors",
        "times",
        "frames",
        "durations",
        "has_velocity",
        "is_static",
        "dynamic_protect",
        "dynamic_allowed",
        "stage_keep",
    ):
        a, b = getattr(cloud, name), getattr(back, name)
        if a is None:
            assert b is None, name
            continue
        assert b is not None and a.dtype == b.dtype and a.shape == b.shape, name
        assert np.array_equal(a, b), name
    for name in ("frame_start", "frame_end", "keyframe_step", "fps", "source_frame_start", "dataset", "segment_id", "method"):
        assert getattr(cloud, name) == getattr(back, name), name


def test_to_trigs_mapping_has_the_trainer_keys() -> None:
    mapping = _cloud(True).to_trigs_mapping()
    assert {"positions", "velocities", "colors", "times", "frame_start", "frame_end", "keyframe_step", "durations"} <= set(mapping)
    assert "source_frame_start" not in mapping
    assert str(mapping["velocity_units"]) == "world_per_video_frame"
