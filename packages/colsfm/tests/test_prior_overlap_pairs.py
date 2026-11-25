"""Pair proposal by prior view overlap: cross-camera and long-range pairs at a fixed per-image budget."""

from __future__ import annotations

import dataclasses

import numpy as np
import pycolmap
import pytest
from mapping_helpers import SyntheticRig
from scipy.spatial.transform import Rotation

from colsfm.config import PriorOverlapConfig
from colsfm.database import ImagePair
from colsfm.frames_meta import FramesMeta, KeyframeMeta
from colsfm.pairs import consecutive_pairs, prior_overlap_pairs, prior_overlap_proposals, select_pairs

NUM_FACES: int = 4
"""Faces of the synthetic horizontal cube: yaw 0, 90, 180 and 270 deg."""

NUM_RIG_FRAMES: int = 24
"""Rig frames; at `YAW_STEP_DEG` per frame the rig turns twice, so every view is revisited."""

YAW_STEP_DEG: float = 30.0
"""Rig yaw per frame."""


def _cube_frames_meta(template: FramesMeta, centre_step_m: float = 0.0) -> FramesMeta:
    """A turning 4-face cube rig: faces never overlap inside one frame, but do across frames.

    Args:
        template: Any collection, for the fields the pair rules do not read.
        centre_step_m: Rig translation per frame along x.

    Returns:
        The collection, keyframe id = frame * NUM_FACES + face.
    """
    keyframes: list[KeyframeMeta] = []
    for frame in range(NUM_RIG_FRAMES):
        for face in range(NUM_FACES):
            yaw_deg: float = frame * YAW_STEP_DEG + face * 90.0
            # Camera z (optical axis) horizontal, rotated about world z by the yaw.
            world_R_cam: Rotation = Rotation.from_euler("z", yaw_deg, degrees=True) * Rotation.from_euler("x", -90.0, degrees=True)
            keyframes.append(
                KeyframeMeta(
                    keyframe_id=frame * NUM_FACES + face,
                    camera_params_id=face,
                    timestamp_microseconds=1_000_000 + frame * 100_000,
                    image_name=f"face_{face}/{frame:04d}.jpeg",
                    world_T_cam=pycolmap.Rigid3d(pycolmap.Rotation3d(world_R_cam.as_quat()), np.array([frame * centre_step_m, 0.0, 0.0])),
                    synced_sample_id=frame,
                    track_id=0,
                )
            )
    return dataclasses.replace(template, keyframes=tuple(keyframes))


def _axis(frames_meta: FramesMeta, keyframe_id: int) -> np.ndarray:
    return frames_meta.keyframe_by_id()[keyframe_id].world_T_cam.rotation.matrix()[:, 2]


@pytest.fixture
def cube(synthetic_rig: SyntheticRig) -> FramesMeta:
    """The turning cube."""
    return _cube_frames_meta(synthetic_rig.frames_meta)


def test_pairs_join_different_cameras_and_distant_frames(cube: FramesMeta) -> None:
    """The rule proposes the two kinds of pair CONSECUTIVE cannot: cross-camera and long-range."""
    pairs: list[ImagePair] = prior_overlap_pairs(cube, PriorOverlapConfig(pairs_per_image=4), connected_keyframe_num=2)
    keyframes: dict[int, KeyframeMeta] = cube.keyframe_by_id()
    cross_camera: list[ImagePair] = [pair for pair in pairs if keyframes[pair[0]].camera_params_id != keyframes[pair[1]].camera_params_id]
    long_range: list[ImagePair] = [
        pair for pair in pairs if abs(keyframes[pair[0]].synced_sample_id - keyframes[pair[1]].synced_sample_id) >= NUM_RIG_FRAMES // 2
    ]
    assert cross_camera
    assert long_range


def test_every_pair_overlaps_and_spans_rig_frames(cube: FramesMeta) -> None:
    """No pair inside one rig frame, none past the axis-angle gate, none CONSECUTIVE already holds."""
    config: PriorOverlapConfig = PriorOverlapConfig(pairs_per_image=4, max_axis_angle_deg=45.0)
    pairs: list[ImagePair] = prior_overlap_pairs(cube, config, connected_keyframe_num=2)
    keyframes: dict[int, KeyframeMeta] = cube.keyframe_by_id()
    consecutive: set[ImagePair] = set(consecutive_pairs(cube, 2))
    for first, second in pairs:
        assert first < second
        assert keyframes[first].synced_sample_id != keyframes[second].synced_sample_id
        angle_deg: float = float(np.degrees(np.arccos(np.clip(_axis(cube, first) @ _axis(cube, second), -1.0, 1.0))))
        assert angle_deg <= config.max_axis_angle_deg + 1e-6
        assert (first, second) not in consecutive


def test_the_budget_bounds_the_pair_count(cube: FramesMeta) -> None:
    """Each image proposes at most `pairs_per_image`, so the count grows with the budget and stays under budget x images."""
    counts: list[int] = [len(prior_overlap_pairs(cube, PriorOverlapConfig(pairs_per_image=budget), connected_keyframe_num=2)) for budget in (1, 2, 4)]
    assert counts[0] < counts[1] < counts[2]
    assert all(count <= budget * len(cube.keyframes) for count, budget in zip(counts, (1, 2, 4), strict=True))


def test_one_image_proposes_spread_out_frames(cube: FramesMeta) -> None:
    """The suppression window keeps one image's picks apart in time, so its neighbours cannot use its whole budget."""
    config: PriorOverlapConfig = PriorOverlapConfig(pairs_per_image=3, suppression_rig_frames=3)
    keyframes: dict[int, KeyframeMeta] = cube.keyframe_by_id()
    picks: list[int] = prior_overlap_proposals(cube, config, connected_keyframe_num=2)[0]
    frames: list[int] = sorted(keyframes[keyframe_id].synced_sample_id for keyframe_id in picks)
    assert len(frames) == config.pairs_per_image
    assert all(later - earlier >= config.suppression_rig_frames for earlier, later in zip(frames, frames[1:], strict=False))


def test_the_centre_distance_gate(synthetic_rig: SyntheticRig) -> None:
    """A rig that walks 1 m per frame cannot pair frames more than `max_centre_distance_m` apart."""
    walking: FramesMeta = _cube_frames_meta(synthetic_rig.frames_meta, centre_step_m=1.0)
    pairs: list[ImagePair] = prior_overlap_pairs(walking, PriorOverlapConfig(pairs_per_image=4, max_centre_distance_m=2.5), connected_keyframe_num=2)
    keyframes: dict[int, KeyframeMeta] = walking.keyframe_by_id()
    assert pairs
    assert all(abs(keyframes[first].synced_sample_id - keyframes[second].synced_sample_id) <= 2 for first, second in pairs)


def test_select_pairs_adds_the_overlap_pairs_only_when_asked(cube: FramesMeta) -> None:
    """Off by default: the pair list is CONSECUTIVE + EXTRINSIC as before."""
    assert select_pairs(cube, connected_keyframe_num=2) == consecutive_pairs(cube, 2)
    with_overlap: list[ImagePair] = select_pairs(cube, connected_keyframe_num=2, prior_overlap=PriorOverlapConfig(pairs_per_image=2))
    assert set(consecutive_pairs(cube, 2)) < set(with_overlap)
    assert with_overlap == sorted(set(with_overlap))
