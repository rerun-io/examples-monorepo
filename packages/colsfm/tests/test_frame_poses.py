"""Every input frame gets a pose: mapped keyframes keep theirs, the others are re-anchored on their neighbours.

The synthetic rig's start poses stand in for the input trajectory. "Optimising" every other rig frame by a known
correction lets each case state its answer exactly: a constant correction moves every frame by itself, and two
different corrections on either side of a frame blend by its time.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pycolmap
from mapping_helpers import SyntheticRig
from scipy.spatial.transform import Rotation

from colsfm.frames_meta import FramesMeta


def _matrix(pose: pycolmap.Rigid3d) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :] = pose.matrix()
    return matrix


def _optimised(rig: SyntheticRig, keep_every: int, correction_of: Callable[[int], np.ndarray]) -> tuple[FramesMeta, set[int]]:
    """The rig with every `keep_every`-th rig frame "mapped": its poses moved by `correction_of(rig_index)`."""
    from colsfm.geometry import rigid3d_from_matrix

    meta = rig.frames_meta
    rig_frames = meta.rig_frames()
    world_T_cam: dict[int, pycolmap.Rigid3d] = {}
    mapped: set[int] = set()
    for index, rig_frame in enumerate(rig_frames):
        if index % keep_every:
            continue
        correction = correction_of(index)
        for keyframe_id in rig_frame.keyframe_ids:
            keyframe = meta.keyframe_by_id()[keyframe_id]
            moved = correction @ _matrix(keyframe.world_T_cam)
            world_T_cam[keyframe_id] = rigid3d_from_matrix(moved[:3, :3], moved[:3, 3])
            mapped.add(keyframe_id)
    return meta.filtered(sorted(mapped)).with_camera_to_world(world_T_cam), mapped


def _shift(translation: list[float], yaw_deg: float = 0.0) -> np.ndarray:
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    correction[:3, 3] = translation
    return correction


def test_a_constant_correction_moves_every_frame_by_itself(synthetic_rig: SyntheticRig) -> None:
    """Every third rig frame mapped, all shifted by the same transform: every input frame comes out shifted by it."""
    from colsfm.frame_poses import every_frame_world_T_vehicle

    correction = _shift([0.3, -0.2, 0.1], yaw_deg=4.0)
    optimised, mapped = _optimised(synthetic_rig, 3, lambda index: correction)
    times_us, poses = every_frame_world_T_vehicle(synthetic_rig.frames_meta, optimised, mapped)
    rig_frames = sorted(synthetic_rig.frames_meta.rig_frames(), key=lambda rig: rig.timestamp_microseconds)
    assert times_us.tolist() == [rig.timestamp_microseconds for rig in rig_frames]
    for pose, rig in zip(poses, rig_frames, strict=True):
        np.testing.assert_allclose(pose, correction @ _matrix(rig.world_T_vehicle), atol=1e-9)


def test_a_frame_between_two_keyframes_blends_their_corrections_by_time(synthetic_rig: SyntheticRig) -> None:
    """Keyframes 0 and 2 shifted 0 m and 0.2 m in x: rig frame 1, halfway in time, is shifted 0.1 m; keyframes exactly."""
    from colsfm.frame_poses import every_frame_world_T_vehicle

    optimised, mapped = _optimised(synthetic_rig, 2, lambda index: _shift([0.1 * index, 0.0, 0.0]))
    times_us, poses = every_frame_world_T_vehicle(synthetic_rig.frames_meta, optimised, mapped)
    rig_frames = sorted(synthetic_rig.frames_meta.rig_frames(), key=lambda rig: rig.timestamp_microseconds)
    for index in (0, 1, 2):
        raw = _matrix(rig_frames[index].world_T_vehicle)
        t_left, t_right = rig_frames[0].timestamp_microseconds, rig_frames[2].timestamp_microseconds
        alpha = (rig_frames[index].timestamp_microseconds - t_left) / (t_right - t_left)
        np.testing.assert_allclose(poses[index][:3, 3], raw[:3, 3] + [0.2 * alpha, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(poses[index][:3, :3], raw[:3, :3], atol=1e-9)


def test_frames_past_the_last_keyframe_take_its_correction(synthetic_rig: SyntheticRig) -> None:
    """Only rig frame 0 mapped: every later frame has one neighbour and takes its correction whole."""
    from colsfm.frame_poses import every_frame_world_T_vehicle

    correction = _shift([0.0, 0.5, 0.0])
    count = len(synthetic_rig.frames_meta.rig_frames())
    optimised, mapped = _optimised(synthetic_rig, count, lambda index: correction)
    _, poses = every_frame_world_T_vehicle(synthetic_rig.frames_meta, optimised, mapped)
    rig_frames = sorted(synthetic_rig.frames_meta.rig_frames(), key=lambda rig: rig.timestamp_microseconds)
    np.testing.assert_allclose(poses[-1], correction @ _matrix(rig_frames[-1].world_T_vehicle), atol=1e-9)


def test_the_every_frame_collection_poses_every_keyframe_through_the_optimised_extrinsics(synthetic_rig: SyntheticRig) -> None:
    """`every_frame_meta` keeps the input's keyframes, and a mapped keyframe's camera pose is its optimised one."""
    from colsfm.frame_poses import every_frame_meta

    optimised, mapped = _optimised(synthetic_rig, 3, lambda index: _shift([0.3, 0.0, 0.0]))
    dense = every_frame_meta(synthetic_rig.frames_meta, optimised, mapped)
    assert len(dense.keyframes) == len(synthetic_rig.frames_meta.keyframes)
    by_id = optimised.keyframe_by_id()
    for keyframe in dense.keyframes:
        if keyframe.keyframe_id in mapped:
            np.testing.assert_allclose(_matrix(keyframe.world_T_cam), _matrix(by_id[keyframe.keyframe_id].world_T_cam), atol=1e-6)


def test_the_every_frame_collection_carries_the_optimised_intrinsics(synthetic_rig: SyntheticRig) -> None:
    """Under `--refine-camera-model` the optimised collection holds the solved focals: the every-frame collection must
    hold the same cameras, or the two `frames_meta.json` files give one camera two models."""
    import pycolmap

    from colsfm.cameras import colmap_cameras, with_colmap_intrinsics
    from colsfm.frame_poses import every_frame_meta

    optimised, mapped = _optimised(synthetic_rig, 3, lambda index: _shift([0.3, 0.0, 0.0]))
    solved: pycolmap.Reconstruction = pycolmap.Reconstruction()
    for camera in colmap_cameras(optimised).values():
        params = np.asarray(camera.params, dtype=np.float64).copy()
        params[:2] *= 1.012
        solved.add_camera(pycolmap.Camera(camera_id=camera.camera_id, model=camera.model, width=camera.width, height=camera.height, params=params))
    refined = with_colmap_intrinsics(optimised, solved)

    dense = every_frame_meta(synthetic_rig.frames_meta, refined, mapped)
    for camera_id, camera in colmap_cameras(dense).items():
        np.testing.assert_allclose(camera.params, solved.cameras[camera_id].params, rtol=1e-12)
    unrefined = every_frame_meta(synthetic_rig.frames_meta, optimised, mapped)
    assert {camera_id: camera.params.tolist() for camera_id, camera in colmap_cameras(unrefined).items()} == {
        camera_id: camera.params.tolist() for camera_id, camera in colmap_cameras(synthetic_rig.frames_meta).items()
    }, "cameras the solve did not move stay as the input has them"


def test_no_registered_keyframe_means_no_poses(synthetic_rig: SyntheticRig) -> None:
    from colsfm.frame_poses import every_frame_world_T_vehicle

    optimised, _ = _optimised(synthetic_rig, 3, lambda index: np.eye(4))
    times_us, poses = every_frame_world_T_vehicle(synthetic_rig.frames_meta, optimised, set())
    assert times_us.shape == (0,) and poses.shape == (0, 4, 4)
