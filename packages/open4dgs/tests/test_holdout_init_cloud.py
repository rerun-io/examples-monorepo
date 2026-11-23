"""Held-out selection is pinned; an init cloud survives the .rrd round trip bit-exactly."""

import math
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import torch

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


def _synthetic_clip(names: tuple[str, ...], holdout: tuple[str, ...], num_frames: int = 4):
    """Cameras on a ring around a tetrahedron at the origin; held-out frames red, training frames blue."""
    from open4dgs.catalog_clip import ClipPeople, MultiViewClip
    from open4dgs.evaluate import look_at

    vertices = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]) - 0.125
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.int32)
    poses = []
    for k in range(len(names)):
        angle = 2.0 * math.pi * k / len(names)
        eye = torch.tensor([3.0 * math.cos(angle), 3.0 * math.sin(angle), 1.0])
        poses.append(look_at(eye, torch.zeros(3), torch.tensor([0.0, 0.0, 1.0])))
    red, blue = torch.tensor([255, 0, 0], dtype=torch.uint8), torch.tensor([0, 0, 255], dtype=torch.uint8)
    frames = tuple((red if name in holdout else blue).expand(num_frames, 32, 32, 3).clone() for name in names)
    people = ClipPeople(
        source="gt",
        names=("person_00",),
        vertices=vertices.expand(num_frames, 1, 4, 3).clone(),
        present=torch.ones(num_frames, 1, dtype=torch.bool),
        faces=faces.to(torch.uint32),
    )
    K = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]]).expand(len(names), 3, 3).clone()
    return MultiViewClip(
        camera_names=names,
        frames=frames,
        valid=tuple(torch.ones(32, 32, dtype=torch.bool) for _ in names),
        K=K,
        world_T_cam=torch.stack(poses),
        fps=30.0,
        frame_indices=torch.arange(num_frames),
        people=people,
    )


def test_holdout_wiring_black_frames_zero_masks_positions_and_init_colours() -> None:
    pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")
    from open4dgs.body_init import body_surface_init
    from open4dgs.holdout import select_holdout
    from open4dgs.run_inputs import split_cameras
    from open4dgs.trigs_adapter import TrainConfig, trainer_frames

    names = ("rig_03/cam_00", "rig_00/cam_00", "rig_02/cam_00", "rig_01/cam_00")  # not sorted: positions are clip order
    train_names, holdout_names = split_cameras(names, ("rig_02/cam_00", "rig_03/cam_00"), num_holdout=1, seed=0)
    assert holdout_names == ("rig_02/cam_00", "rig_03/cam_00") and train_names == ("rig_00/cam_00", "rig_01/cam_00")
    with pytest.raises(ValueError, match="rig_09"):
        split_cameras(names, ("rig_09/cam_00",), num_holdout=1, seed=0)
    assert split_cameras(names, None, 1, 0) == select_holdout(names, 1, 0)

    clip = _synthetic_clip(names, holdout_names)
    masks = tuple(torch.ones(4, 32, 32) for _ in names)
    gt_images, gt_alphas, positions = trainer_frames(clip, masks, holdout_names)
    assert positions == (2, 0)
    assert TrainConfig().to_trigs(clip.fps, 3, positions).holdout_views == (2, 0)
    for f in range(4):
        for c in range(len(names)):
            if c in positions:
                assert int(gt_images[f][c].sum()) == 0 and float(gt_alphas[f][c].sum()) == 0.0
            else:
                assert torch.equal(gt_images[f][c], clip.frames[c][f]) and torch.equal(gt_alphas[f][c], masks[c][f])

    init = body_surface_init(clip, train_names, dataset="d", segment_id="s", samples_per_person=200, keyframe_step=1, device=torch.device("cpu"))
    assert float(init.colors[:, 0].max()) == 0.0 and float(init.colors[:, 2].min()) == 1.0  # only blue (training) pixels


def test_layer_init_must_match_the_run_split_and_window() -> None:
    from open4dgs.init_cloud import check_init_matches_run

    cloud = _cloud(with_optional=False)
    start, num = cloud.source_frame_start, cloud.frame_end + 1
    check_init_matches_run(cloud, ("rig_25/cam_00", "rig_03/cam_00"), ("rig_03/cam_00", "rig_25/cam_00"), start, num)  # order-free
    with pytest.raises(ValueError, match="would leak"):
        check_init_matches_run(cloud, ("rig_25/cam_00",), ("rig_03/cam_00",), start, num)
    with pytest.raises(ValueError, match="--num-frames"):
        check_init_matches_run(cloud, ("rig_25/cam_00",), ("rig_25/cam_00",), start, num + 1)
