"""Generic logging of a COLMAP-style rig skeleton + motion into Rerun.

This is the exoego-side counterpart to ``live_rerun.rerun_video_logger``: it logs
the rig/camera *skeleton* (rig metadata, per-camera ``rig_T_cam`` + pinhole,
reference-camera tint, sensor metadata) and, for a moving rig, the per-frame
``world_T_rig(t)`` on the rig node so every child camera moves rigidly with it.

It deliberately knows nothing about COCO keypoints, MANO, depth, or video — those
stay in :mod:`simplecv.apis.view_exoego`. It consumes the generic
:class:`simplecv.rig.Rig` produced by the dataset-agnostic builders.

Entity layout (see ``packages/simplecv/docs/exoego_schema.md``)::

    /world                              ViewCoordinates (logged by the caller)
      /rig_NN                           AnyValues{schema_version, reference, num_cameras}
                                        + (moving rig only) Transform3D = world_T_rig(t)
        /cam_MM                         Transform3D = rig_T_cam (static)
                                        + AnyValues{name, kind}
          /pinhole                      PinholeWithDistortion (static)
            /video                      VideoStream (logged by view_exoego)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Int
from numpy import ndarray

from simplecv.rerun_log_utils import log_pinhole
from simplecv.rig import Rig, entity_id

#: Schema identifier logged on every rig node so loaders can validate the layout.
SCHEMA_VERSION: str = "exoego:v2"

#: Frustum wireframe colour for the reference camera (the rig origin) in a
#: multi-camera rig, so it is trivially distinguishable. Matches slam-evals' /
#: live-rerun's green. Single-camera rigs are not tinted (every rig would be
#: green otherwise, which carries no information).
REFERENCE_CAM_COLOR: tuple[int, int, int, int] = (40, 200, 80, 255)


def log_rig_static(rig: Rig, *, world_path: str = "world", recording: rr.RecordingStream | None = None) -> None:
    """Log a rig's static skeleton (no video, no labels).

    Logs, under ``/{world_path}/rig_<index>``:

    - static ``AnyValues(schema_version, reference, num_cameras)`` on the rig node.
      **No** static transform is logged here: a moving rig receives a *temporal*
      ``world_T_rig`` from :func:`log_rig_pose_stream`, and a static
      world-anchored rig stays at implicit identity. (``AnyValues`` is a distinct
      component, so logging it static never conflicts with the temporal pose.)
    - per camera: ``log_pinhole`` → the static ``rig_T_cam`` ``Transform3D`` plus
      the ``PinholeWithDistortion`` at ``cam_<MM>/pinhole``.
    - the reference camera of a *multi-camera* rig gets a green frustum tint
      (partial ``Pinhole`` update).
    - per camera: static ``AnyValues(name, kind)``.
    """
    rig_path: str = f"{world_path}/{entity_id('rig', rig.index)}"
    calibration = rig.calibration
    rr.log(
        rig_path,
        rr.AnyValues(
            schema_version=SCHEMA_VERSION,
            reference=entity_id("cam", calibration.reference_index),
            num_cameras=len(calibration.cameras),
        ),
        static=True,
        recording=recording,
    )
    is_multi_camera: bool = len(calibration.cameras) > 1
    for cam in calibration.cameras:
        cam_path: str = f"{rig_path}/{entity_id('cam', cam.index)}"
        # log_pinhole logs the rig_T_cam Transform3D at cam_path (from_parent=True)
        # and the PinholeWithDistortion at cam_path/pinhole.
        log_pinhole(
            cam.pinhole,
            cam_log_path=Path(cam_path),
            image_plane_distance=rig.image_plane_distance,
            static=True,
            recording=recording,
            include_distortion=True,
        )
        if is_multi_camera and cam.index == calibration.reference_index:
            rr.log(
                f"{cam_path}/pinhole",
                rr.Pinhole.from_fields(image_plane_distance=rig.image_plane_distance, color=REFERENCE_CAM_COLOR),
                static=True,
                recording=recording,
            )
        rr.log(cam_path, rr.AnyValues(name=cam.name, kind=cam.kind), static=True, recording=recording)


def log_rig_pose_stream(
    rig: Rig,
    *,
    timestamps_ns: Int[ndarray, "n_frames"],
    world_path: str = "world",
    timeline: str = "video_time",
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log ``world_T_rig(t)`` on the rig node, moving the whole rig rigidly.

    No-op when ``rig.pose_stream`` is ``None`` (a static world-anchored rig). The
    pose is logged on the rig node ``/{world_path}/rig_<index>`` (NOT
    ``from_parent``, so the stored value is ``world_T_rig``), exactly as the v1 ego
    writer logged ``world_T_cam`` on the camera node — but here it rides on the rig
    so all child cameras follow. Frames whose pose is **NaN** (a tracking dropout)
    render no frustum, so the whole rig disappears for the duration of the gap.

    Caller contract: ``timestamps_ns`` must already be trimmed to the demuxed
    video frames (see ``view_exoego``); the pose arrays are sliced to the common
    length defensively.
    """
    pose_stream = rig.pose_stream
    if pose_stream is None:
        return
    n_frames: int = min(len(timestamps_ns), len(pose_stream.world_t_rig), len(pose_stream.world_R_rig))
    if n_frames == 0:
        return
    rig_path: str = f"{world_path}/{entity_id('rig', rig.index)}"
    timestamps_s: np.ndarray = 1e-9 * np.asarray(timestamps_ns[:n_frames])
    rr.send_columns(
        rig_path,
        indexes=[rr.TimeColumn(timeline, duration=timestamps_s)],
        columns=[
            *rr.Transform3D.columns(
                translation=np.asarray(pose_stream.world_t_rig[:n_frames]),
                mat3x3=np.asarray(pose_stream.world_R_rig[:n_frames]),
            ),
        ],
        recording=recording,
    )
