# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Assignment and reprojection helpers for multi-view tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lamptrack.third_party.lamp.core.se3 import compose, invert
from lamptrack.third_party.lamp.core.types import Person, PersonState, Skeleton
from lamptrack.third_party.lamp.io.sensor_io import PerCameraCalibration
from scipy.optimize import (
    linear_sum_assignment,  # pyright: ignore[reportUnknownVariableType]
)


@dataclass(slots=True)
class Assignment:
    """A single matched (or unmatched) detection -> track row."""

    detection_idx: int
    track_idx: int  # -1 for unassigned
    cost: float


@dataclass(slots=True)
class SensorRecord:
    """Per-camera per-timestamp record needed for cross-cam reprojection."""

    T_world_cam: np.ndarray  # 4x4 float32
    cam_model: PerCameraCalibration | None
    cam_idx: int


def hungarian_assign(cost: np.ndarray) -> list[Assignment]:
    """Wrap `scipy.optimize.linear_sum_assignment` for our rectangular case."""
    if cost.size == 0:
        return []
    rows_arr, cols_arr = linear_sum_assignment(cost)  # pyright: ignore[reportUnknownVariableType]
    rows: list[int] = [int(r) for r in rows_arr]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    cols: list[int] = [int(c) for c in cols_arr]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    matched: dict[int, int] = dict(zip(rows, cols, strict=True))
    out: list[Assignment] = []
    for det_idx in range(cost.shape[0]):
        if det_idx in matched:
            track_idx = matched[det_idx]
            out.append(Assignment(det_idx, track_idx, float(cost[det_idx, track_idx])))
        else:
            out.append(Assignment(det_idx, -1, float("inf")))
    return out


def _clip_xy(xy: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Clamp a `(2,)` pixel coord into `[0, W-1] x [0, H-1]`."""
    width, height = image_size
    return np.array(
        [
            min(max(float(xy[0]), 0.0), float(width - 1)),
            min(max(float(xy[1]), 0.0), float(height - 1)),
        ],
        dtype=np.float32,
    )


def _box_corners_xyxy(box: np.ndarray) -> np.ndarray:
    """Return the 4 corners of an xyxy box as a `(4, 2)` array."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return np.array(
        [[x1, y1], [x2, y1], [x1, y2], [x2, y2]],
        dtype=np.float32,
    )


def _bbox_from_points(points: list[np.ndarray]) -> np.ndarray | None:
    """Axis-aligned bbox `[xmin, ymin, xmax, ymax]` from a list of `(2,)` pixels."""
    if not points:
        return None
    arr = np.stack(points, axis=0)
    return np.array(
        [arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()],
        dtype=np.float32,
    )


def transform_detection_2d(
    person: Person,
    old_cam: PerCameraCalibration | None,
    new_cam: PerCameraCalibration | None,
    T_world_oldcam: np.ndarray,
    T_world_newcam: np.ndarray,
) -> np.ndarray | None:
    """Reproject a person's last detection into a new camera frame."""
    if not person.ts_to_states or person.last_obs_ts < 0:
        return None
    # Use the last observed state, which may differ from the latest lifted ts.
    last_state = person.ts_to_states.get(person.last_obs_ts)
    if last_state is None or not last_state.detection2ds:
        return None
    old_det = last_state.detection2ds[-1]

    # Path 1: project the lifted 3D keypoints into the new cam.
    if person.last_lifted_ts != -1 and new_cam is not None:
        skeleton = person.ts_to_states.get(
            person.last_lifted_ts, PersonState([])
        ).skeleton
        if skeleton is None:
            return None
        return _project_skeleton_box(skeleton, new_cam, T_world_newcam)

    # Path 2: unlifted person. Keypoints give a tighter reprojection box; bbox
    # corners are a fallback for bare-box detections.
    if old_cam is None or new_cam is None:
        return None
    if old_cam.unproject is None or new_cam.project is None:
        return None

    T_new_old = compose(invert(T_world_newcam), T_world_oldcam)
    # The old detection depth is unknown, so only relative rotation is useful.
    R_new_old = T_new_old[:3, :3]
    unproject = old_cam.unproject
    project = new_cam.project

    def _reproject(xy: np.ndarray) -> np.ndarray | None:
        """Unproject `xy` in the old cam, rotate, and reproject into the new cam."""
        ray3 = unproject(xy)
        if ray3 is None:
            return None
        ray3 = np.asarray(ray3, dtype=np.float32)
        if ray3.shape != (3,):
            return None
        new_xy = project(R_new_old @ ray3)
        if new_xy is None:
            return None
        new_xy = np.asarray(new_xy, dtype=np.float32)
        if new_xy.shape != (2,):
            return None
        return _clip_xy(new_xy, new_cam.image_size)

    projected: list[np.ndarray] = []
    if old_det.has_keypoints and old_det.keypoints.size > 0:
        # Per-keypoint path.
        for kp_idx in range(old_det.keypoints.shape[0]):
            kp = old_det.keypoints[kp_idx]
            # Skip suppressed keypoints (score == 0 was zeroed by the detector).
            if float(kp[2]) <= 0.0:
                continue
            kp_xy = np.asarray([float(kp[0]), float(kp[1])], dtype=np.float32)
            new_xy = _reproject(kp_xy)
            if new_xy is not None:
                projected.append(new_xy)

    if not projected:
        # 4-corner fallback for bare-box detections or when no keypoint projected.
        for corner in _box_corners_xyxy(old_det.box_xyxy):
            new_xy = _reproject(corner.astype(np.float32))
            if new_xy is not None:
                projected.append(new_xy)

    box = _bbox_from_points(projected)
    if box is None:
        return None
    if (box[2] - box[0]) * (box[3] - box[1]) < 10.0:
        return None
    return box


def _project_skeleton_box(
    skeleton: Skeleton,
    new_cam: PerCameraCalibration,
    T_world_newcam: np.ndarray,
) -> np.ndarray | None:
    """Project a 3D skeleton into a camera and return an in-image bbox."""
    if new_cam.project is None:
        return None
    T_newcam_world = invert(T_world_newcam)
    width, height = new_cam.image_size
    in_image: list[np.ndarray] = []
    for kp_world in skeleton.kp_world:
        kp_world_4 = np.array(
            [float(kp_world[0]), float(kp_world[1]), float(kp_world[2]), 1.0],
            dtype=np.float32,
        )
        kp_cam = (T_newcam_world @ kp_world_4)[:3]
        xy = new_cam.project(kp_cam)
        if xy is None:
            continue
        xy = np.asarray(xy, dtype=np.float32)
        if xy.shape != (2,):
            continue
        if not (0.0 <= float(xy[0]) <= float(width - 1)) or not (
            0.0 <= float(xy[1]) <= float(height - 1)
        ):
            continue
        in_image.append(_clip_xy(xy, new_cam.image_size))

    if len(in_image) < 5:
        return None
    return _bbox_from_points(in_image)


def cam_params_16(cam: PerCameraCalibration) -> np.ndarray:
    """Build the per-frame camera vector LAMP's unprojector expects.

    Fisheye624 cameras return a length-16 vector `[fx, fy, cx, cy, k0..k11]`
    (`cam.projection_params` is `[fx, cx, cy, k0..k11]` with `fy == fx`).

    Linear / pinhole cameras return a length-4 vector `[fx, fy, cx, cy]`. The
    length of 4 is exactly how `get_cam_ray` routes a camera to the pinhole
    unprojector instead of the fisheye one, so a pinhole calibration must not be
    padded out to 16 (that would silently route it through fisheye undistortion).
    """
    params = cam.projection_params
    if params.size >= 15:
        # FISHEYE624: [fx, cx, cy, k0..k11].
        out = np.zeros(16, dtype=np.float32)
        out[0] = params[0]
        out[1] = params[0]  # fy == fx
        out[2] = params[1]
        out[3] = params[2]
        out[4:16] = params[3:15]
        return out
    if cam.intrinsic_matrix.size >= 9:
        # Linear / pinhole: [fx, fy, cx, cy] straight from K.
        K = cam.intrinsic_matrix
        return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    # Unknown model: keep the historical zero-fisheye shape.
    return np.zeros(16, dtype=np.float32)
