"""A pose for every input frame, from the optimised keyframes: what the layer and the TUM export carry.

Keyframe selection keeps a subset of the rig frames (`colsfm.keyframe_selection`), and only those are mapped. Every
other input rig frame keeps its start pose, re-anchored on its optimised neighbouring keyframes the way RootSLAM's
`merge_frames_keyframes.py` does it (`reanchor`): for a frame `f` between keyframes `L` and `R`,

    T_f_left  = T_L_opt * T_L_raw^-1 * T_f_raw
    T_f_right = T_R_opt * T_R_raw^-1 * T_f_raw
    T_f       = interpolate(T_f_left, T_f_right, alpha),  alpha = (t_f - t_L) / (t_R - t_L)

with the translation interpolated linearly and the rotation by slerp. A frame before the first or after the last
keyframe takes the one correction it has; a keyframe keeps its optimised pose exactly. The raw poses are the start
trajectory's (`world_T_vehicle` of the input collection), so the correction carries whatever the mapping changed,
the refined extrinsics included.

The layer's viewing rig (`view_rig_poses`) uses the same rule on the prior at its own rate: held out every other
registered rig frame of the X5 office clip (0.67 s apart), it predicts them within 0.146 cm / 0.035 deg (median),
where the rig-frame offset `raw^-1 opt` interpolated gives 0.150 cm / 0.036 deg and a slerp between keyframes 4.0 cm
/ 2.5 deg.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

import numpy as np
import pycolmap
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation, Slerp

from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.geometry import interpolate_poses, rigid3d_from_matrix


def _matrix(pose: pycolmap.Rigid3d) -> Float64[ndarray, "4 4"]:
    matrix: Float64[ndarray, "4 4"] = np.eye(4)
    matrix[:3, :] = pose.matrix()
    return matrix


def reanchor(
    prior_t: Int64[ndarray, " n"],
    world_T_prior: Float64[ndarray, "n 4 4"],
    anchor_t: Int64[ndarray, " k"],
    world_T_anchor: Float64[ndarray, "k 4 4"],
    query_t: Int64[ndarray, " m"],
) -> Float64[ndarray, "m 4 4"]:
    """The prior at ``query_t`` re-anchored on the optimised poses at ``anchor_t`` (module docstring, RootSLAM's rule).

    Each anchor's world-frame correction ``opt * prior^-1`` is applied to the prior at the query, and the two corrected
    poses of the anchors around it are blended (translation linearly, rotation by slerp) at the query's place between
    them; a query outside the anchors takes the nearest one's correction. A query at an anchor gets its pose exactly.

    Args:
        prior_t: Prior timestamps, increasing; they cover every anchor and query.
        world_T_prior: The prior's poses.
        anchor_t: Anchor timestamps, increasing (at least one).
        world_T_anchor: The optimised poses at the anchors.
        query_t: Where poses are wanted.

    Returns:
        One pose per query.
    """
    corrections: Float64[ndarray, "k 4 4"] = world_T_anchor @ np.linalg.inv(interpolate_poses(prior_t, world_T_prior, anchor_t))
    prior: Float64[ndarray, "m 4 4"] = interpolate_poses(prior_t, world_T_prior, query_t)
    left: Int64[ndarray, " m"] = np.clip(np.searchsorted(anchor_t, query_t, side="right") - 1, 0, len(anchor_t) - 1)
    right: Int64[ndarray, " m"] = np.clip(left + 1, 0, len(anchor_t) - 1)
    span: Float64[ndarray, " m"] = (anchor_t[right] - anchor_t[left]).astype(np.float64)
    alpha: Float64[ndarray, " m"] = np.where(span > 0, (query_t - anchor_t[left]) / np.where(span > 0, span, 1.0), 0.0).clip(0.0, 1.0)
    from_left: Float64[ndarray, "m 4 4"] = corrections[left] @ prior
    from_right: Float64[ndarray, "m 4 4"] = corrections[right] @ prior
    poses: Float64[ndarray, "m 4 4"] = from_left.copy()
    poses[:, :3, 3] = (1.0 - alpha)[:, None] * from_left[:, :3, 3] + alpha[:, None] * from_right[:, :3, 3]
    for row in np.flatnonzero((alpha > 0.0) & (alpha < 1.0)):
        pair: Rotation = Rotation.from_matrix(np.stack([from_left[row, :3, :3], from_right[row, :3, :3]]))
        poses[row, :3, :3] = Slerp([0.0, 1.0], pair)([float(alpha[row])]).as_matrix()[0]
    return poses


@dataclass(frozen=True)
class ViewRigSource:
    """What a layer's viewing rig is drawn from: the prior the run started from, at its own rate, and the viewing frames."""

    prior_t_ns: Int64[ndarray, " n"]
    """Prior timestamps, increasing, nanoseconds on the layer's timeline."""
    world_T_prior: Float64[ndarray, "n 4 4"]
    """The prior's `world_T_rig` at each."""
    view_t_ns: Int64[ndarray, " m"]
    """Stamps of the viewing frames (the samples of each camera's viewing stream), increasing."""
    view_entities: dict[str, str]
    """Each camera's viewing stream, by the entity of the stream the run read (`/<camera>/pinhole/video`)."""


def view_rig_poses(rig_t_ns: Int64[ndarray, " k"], world_T_rig: Float64[ndarray, "k 4 4"], source: ViewRigSource) -> tuple[Int64[ndarray, " m"], Float64[ndarray, "m 4 4"]]:
    """The rig at every viewing frame between the first and last anchor rig frame (`rig_t_ns`) inside the prior's span:
    the prior re-anchored on those rig frames (`reanchor`). A hand-held walk does not move like a slerp between them.

    Returns:
        Those viewing frames and the rig pose at each; nothing with fewer than two such rig frames.
    """
    keys = (rig_t_ns >= source.prior_t_ns[0]) & (rig_t_ns <= source.prior_t_ns[-1])
    if keys.sum() < 2:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4, 4))
    key_t: Int64[ndarray, " k"] = rig_t_ns[keys]
    inside: Int64[ndarray, " m"] = source.view_t_ns[(source.view_t_ns >= key_t[0]) & (source.view_t_ns <= key_t[-1])]
    return inside, reanchor(source.prior_t_ns, source.world_T_prior, key_t, world_T_rig[keys], inside)


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
    optimised_poses: list[Float64[ndarray, "4 4"]] = []
    for row, rig in enumerate(rig_frames):
        mapped: list[int] = [
            keyframe_id for keyframe_id in rig.keyframe_ids if keyframe_id in registered_keyframe_ids and keyframe_id in optimised_by_id
        ]
        if not mapped:
            continue
        optimised_pose: Float64[ndarray, "4 4"] = _matrix(optimised.world_T_vehicle(optimised_by_id[min(mapped)]))
        anchor_rows.append(row)
        optimised_poses.append(optimised_pose)
    if not anchor_rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4, 4), dtype=np.float64)

    anchors: Int64[ndarray, " k"] = np.asarray(anchor_rows, dtype=np.int64)
    poses: Float64[ndarray, "n 4 4"] = reanchor(times_us, raw, times_us[anchors], np.asarray(optimised_poses, dtype=np.float64), times_us)
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
