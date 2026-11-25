"""A pose for every input frame, from the optimised keyframes: what the layer and the TUM export carry.

Keyframe selection keeps a subset of the rig frames (`colsfm.keyframe_selection`), and only those are mapped. Every
other input rig frame keeps its start pose, re-anchored on its optimised neighbouring keyframes the way RootSLAM's
`merge_frames_keyframes.py` does it: for a frame `f` between keyframes `L` and `R`,

    T_f_left  = T_L_opt * T_L_raw^-1 * T_f_raw
    T_f_right = T_R_opt * T_R_raw^-1 * T_f_raw
    T_f       = interpolate(T_f_left, T_f_right, alpha),  alpha = (t_f - t_L) / (t_R - t_L)

with the translation interpolated linearly and the rotation by slerp. A frame before the first or after the last
keyframe takes the one correction it has; a keyframe keeps its optimised pose exactly. The raw poses are the start
trajectory's (`world_T_vehicle` of the input collection), so the correction carries whatever the mapping changed,
the refined extrinsics included.
"""

from __future__ import annotations

from collections.abc import Set

import numpy as np
import pycolmap
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp

from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.geometry import rigid3d_from_matrix


def _matrix(pose: pycolmap.Rigid3d) -> Float64[ndarray, "4 4"]:
    matrix: Float64[ndarray, "4 4"] = np.eye(4)
    matrix[:3, :] = pose.matrix()
    return matrix


def every_frame_world_T_vehicle(
    input_meta: FramesMeta, optimised: FramesMeta, registered_keyframe_ids: Set[int]
) -> tuple[Int64[ndarray, " n"], Float64[ndarray, "n 4 4"]]:
    """One `world_T_vehicle` per input rig frame, re-anchored on the optimised keyframes around it.

    Args:
        input_meta: The input collection, every frame, with the start poses.
        optimised: The optimised collection; its keyframe ids are a subset of the input's.
        registered_keyframe_ids: Keyframes the reconstruction registered; the others are not anchors.

    Returns:
        Each input rig frame's reference timestamp (microseconds, ascending) and its pose. Empty when no keyframe
        was registered.

    Raises:
        ValueError: When two input rig frames share a timestamp.
    """
    optimised_by_id: dict[int, KeyframeMeta] = optimised.keyframe_by_id()
    rig_frames: list[RigFrame] = sorted(input_meta.rig_frames(), key=lambda rig: rig.timestamp_microseconds)
    times_us: Int64[ndarray, " n"] = np.asarray([rig.timestamp_microseconds for rig in rig_frames], dtype=np.int64)
    if len(times_us) > 1 and not np.all(np.diff(times_us) > 0):
        raise ValueError("two input rig frames share a timestamp; every-frame poses need strictly increasing times")
    raw: Float64[ndarray, "n 4 4"] = np.asarray([_matrix(rig.world_T_vehicle) for rig in rig_frames], dtype=np.float64).reshape(-1, 4, 4)

    # An input rig frame is an anchor when one of its keyframes was mapped; its optimised pose is that keyframe's
    # `world_T_vehicle`, which every camera of the rig frame shares.
    anchor_rows: list[int] = []
    corrections: list[Float64[ndarray, "4 4"]] = []
    for row, rig in enumerate(rig_frames):
        mapped: list[int] = [
            keyframe_id for keyframe_id in rig.keyframe_ids if keyframe_id in registered_keyframe_ids and keyframe_id in optimised_by_id
        ]
        if not mapped:
            continue
        optimised_pose: Float64[ndarray, "4 4"] = _matrix(optimised.world_T_vehicle(optimised_by_id[min(mapped)]))
        anchor_rows.append(row)
        corrections.append(optimised_pose @ np.linalg.inv(raw[row]))
    if not anchor_rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4, 4), dtype=np.float64)

    anchors: Int64[ndarray, " k"] = np.asarray(anchor_rows, dtype=np.int64)
    correction: Float64[ndarray, "k 4 4"] = np.asarray(corrections, dtype=np.float64)
    # Index of the anchor at or before each row, and the next one.
    left: Int64[ndarray, " n"] = np.clip(np.searchsorted(anchors, np.arange(len(rig_frames)), side="right") - 1, 0, len(anchors) - 1)
    right: Int64[ndarray, " n"] = np.clip(left + 1, 0, len(anchors) - 1)
    t_left: Float64[ndarray, " n"] = times_us[anchors[left]].astype(np.float64)
    t_right: Float64[ndarray, " n"] = times_us[anchors[right]].astype(np.float64)
    span: Float64[ndarray, " n"] = t_right - t_left
    alpha: Float64[ndarray, " n"] = np.where(span > 0, (times_us - t_left) / np.where(span > 0, span, 1.0), 0.0).clip(0.0, 1.0)

    from_left: Float64[ndarray, "n 4 4"] = np.einsum("nij,njk->nik", correction[left], raw)
    from_right: Float64[ndarray, "n 4 4"] = np.einsum("nij,njk->nik", correction[right], raw)
    poses: Float64[ndarray, "n 4 4"] = from_left.copy()
    poses[:, :3, 3] = (1.0 - alpha)[:, None] * from_left[:, :3, 3] + alpha[:, None] * from_right[:, :3, 3]
    for row in np.flatnonzero((alpha > 0.0) & (alpha < 1.0)):
        pair: Rotation = Rotation.from_matrix(np.stack([from_left[row, :3, :3], from_right[row, :3, :3]]))
        poses[row, :3, :3] = Slerp([0.0, 1.0], pair)([float(alpha[row])]).as_matrix()[0]
    return times_us, poses


def every_frame_meta(input_meta: FramesMeta, optimised: FramesMeta, registered_keyframe_ids: Set[int]) -> FramesMeta:
    """The input collection with every keyframe posed: mapped ones optimised, the others re-anchored.

    The cameras carry the optimised extrinsics, so `world_T_cam = world_T_vehicle * vehicle_T_cam` agrees with the
    optimised collection on every mapped keyframe, and the optimised intrinsics (`--refine-camera-model`), so both
    collections give a camera one model.

    Args:
        input_meta: The input collection, every frame, with the start poses.
        optimised: The optimised collection.
        registered_keyframe_ids: Keyframes the reconstruction registered.

    Returns:
        The every-frame collection; `input_meta` unchanged when no keyframe was registered.
    """
    times_us, world_T_vehicle = every_frame_world_T_vehicle(input_meta, optimised, registered_keyframe_ids)
    if len(times_us) == 0:
        return input_meta
    with_extrinsics: FramesMeta = input_meta.with_intrinsics(optimised.cameras).with_extrinsics(
        {camera_id: camera.vehicle_T_cam for camera_id, camera in optimised.cameras.items()}
    )
    row_by_time: dict[int, int] = {int(time_us): row for row, time_us in enumerate(times_us.tolist())}
    keyframe_by_id: dict[int, KeyframeMeta] = with_extrinsics.keyframe_by_id()
    world_T_cam: dict[int, pycolmap.Rigid3d] = {}
    for rig in with_extrinsics.rig_frames():
        pose: Float64[ndarray, "4 4"] = world_T_vehicle[row_by_time[rig.timestamp_microseconds]]
        world_T_rig: pycolmap.Rigid3d = rigid3d_from_matrix(pose[:3, :3], pose[:3, 3])
        for keyframe_id in rig.keyframe_ids:
            camera_params_id: int = keyframe_by_id[keyframe_id].camera_params_id
            world_T_cam[keyframe_id] = world_T_rig * with_extrinsics.cameras[camera_params_id].vehicle_T_cam
    return with_extrinsics.with_camera_to_world(world_T_cam, "ALIGNMENT")
