"""`frames_meta.json` to the rig types `colsfm.loop_pose` measures against.

`colsfm.loop_pose` deliberately knows nothing about cuSFM metadata: it takes a `RigGeometry`
(cameras, `cam_T_vehicle` extrinsics, stereo partnerships) and a `RigFrameIndex` (rig frames
in time order, their prior poses, their images) and nothing else, so it can be driven from a
synthetic scene in a test as easily as from a real run. This module is the one adapter that
turns a parsed `FramesMeta` into those two, and it lives apart from `colsfm.loop_closure`
because the conversion is not part of the loop-closure funnel — the pose-graph and
extrinsics work needs the same rig.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pycolmap

from colsfm.cameras import colmap_cameras
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame


@dataclass(frozen=True, slots=True)
class RigGeometry:
    """The rig itself: what the two generalized estimators take as their fixed arguments.

    `cams_from_rig` and `cameras` are parallel to `camera_ids`, and a camera's position in
    those tuples is the `camera_idx` both pycolmap estimators expect.
    """

    camera_ids: tuple[int, ...]
    """cuSFM `camera_params_id` per rig sensor, ascending."""
    cams_from_rig: tuple[pycolmap.Rigid3d, ...]
    """`cam_T_vehicle` per camera, i.e. the inverse of the metadata's `vehicle_T_cam`; the
    same convention `colsfm.reconstruction.build_rig` gives COLMAP."""
    cameras: tuple[pycolmap.Camera, ...]
    """Calibrated `pycolmap.Camera` per camera, `has_prior_focal_length` set."""
    stereo_partner: Mapping[int, int] = field(default_factory=dict)
    """The other camera of a declared stereo pair, per `camera_params_id`. Empty when the
    metadata declares no pair, in which case the local map triangulates from the temporal
    neighbours alone."""

    def position_of(self, camera_id: int) -> int:
        """Index of one camera in the parallel tuples.

        Args:
            camera_id: cuSFM `camera_params_id`.

        Returns:
            Its `camera_idx` for the generalized estimators.

        Raises:
            KeyError: When the rig holds no such camera.
        """
        try:
            return self.camera_ids.index(camera_id)
        except ValueError as error:
            raise KeyError(f"camera {camera_id} is not part of this rig") from error


@dataclass(frozen=True, slots=True)
class RigFrameIndex:
    """Rig frames in time order, their prior poses and their member images."""

    sequence: tuple[int, ...]
    """Rig ids in capture order; `neighbours` walks this, not the id arithmetic, because
    `synced_sample_id` need not be contiguous."""
    world_T_rig: Mapping[int, pycolmap.Rigid3d]
    """Prior rig pose per rig id. Only *relative* poses inside a neighbourhood are read, so
    session-scale drift in these is exactly what the loop edge is allowed to contradict."""
    keyframe_by_rig_camera: Mapping[tuple[int, int], int]
    """Image id per `(rig id, camera_params_id)`; a camera may be absent from a rig frame."""

    def keyframe(self, rig_id: int, camera_id: int) -> int | None:
        """The image one camera contributed to one rig frame.

        Args:
            rig_id: The rig frame's `synced_sample_id`.
            camera_id: cuSFM `camera_params_id`.

        Returns:
            The image id, or None when that camera did not fire in that rig frame.
        """
        return self.keyframe_by_rig_camera.get((rig_id, camera_id))

    def neighbours(self, rig_id: int, span: int) -> tuple[int, ...]:
        """Rig frames within `span` steps of one rig frame, in time order.

        Args:
            rig_id: The rig frame to look around.
            span: How many steps to reach in each direction.

        Returns:
            The neighbouring rig ids, excluding `rig_id` itself; empty when `span` is 0 or
            the rig id is unknown.

        Raises:
            ValueError: When `span` is negative.
        """
        if span < 0:
            raise ValueError(f"neighbour span must not be negative, got {span}")
        try:
            position: int = self.sequence.index(rig_id)
        except ValueError:
            return ()
        lower: int = max(0, position - span)
        upper: int = min(len(self.sequence), position + span + 1)
        return tuple(self.sequence[step] for step in range(lower, upper) if step != position)


def build_rig_geometry(
    cameras: Mapping[int, pycolmap.Camera], vehicle_T_cam: Mapping[int, pycolmap.Rigid3d], stereo_partner: Mapping[int, int]
) -> RigGeometry:
    """Assemble the fixed arguments of the generalized estimators.

    Args:
        cameras: Calibrated `pycolmap.Camera` per `camera_params_id`.
        vehicle_T_cam: Rig extrinsic per `camera_params_id`, camera to vehicle.
        stereo_partner: The other camera of a declared stereo pair, per camera id.

    Returns:
        The rig geometry, cameras ordered by ascending id.

    Raises:
        KeyError: When a camera has no extrinsic.
    """
    camera_ids: tuple[int, ...] = tuple(sorted(cameras))
    return RigGeometry(
        camera_ids=camera_ids,
        cams_from_rig=tuple(vehicle_T_cam[camera_id].inverse() for camera_id in camera_ids),
        cameras=tuple(cameras[camera_id] for camera_id in camera_ids),
        stereo_partner=dict(stereo_partner),
    )


def calibrated_cameras(frames_meta: FramesMeta) -> dict[int, pycolmap.Camera]:
    """Build the `pycolmap.Camera`s the generalized estimators need.

    `Camera.has_prior_focal_length` defaults to **False**, and with it false COLMAP takes
    the uncalibrated path: a pinhole pair degrades to a fundamental matrix with no relative
    pose, and a fisheye pair comes back `DEGENERATE` with zero inliers, silently
    (`docs/spec/pycolmap-capabilities.md` §5). `colsfm.cameras.colmap_camera` sets it on
    every camera it builds, so this is `colmap_cameras` and the name is kept only to say
    at the call site which property the estimators depend on.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Calibrated cameras keyed by `camera_params_id`.
    """
    return colmap_cameras(frames_meta)


def rig_geometry(frames_meta: FramesMeta) -> RigGeometry:
    """The rig the metric estimator resections against.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        Calibrated cameras, their `cam_T_vehicle` extrinsics and the declared stereo
        partnerships, in `colsfm.loop_pose`'s parallel-tuple form.
    """
    partners: dict[int, int] = {}
    for pair in frames_meta.stereo_pairs:
        partners[pair.left_camera_params_id] = pair.right_camera_params_id
        partners[pair.right_camera_params_id] = pair.left_camera_params_id
    return build_rig_geometry(
        calibrated_cameras(frames_meta),
        {camera_id: camera.vehicle_T_cam for camera_id, camera in frames_meta.cameras.items()},
        partners,
    )


def rig_frame_index(frames_meta: FramesMeta) -> RigFrameIndex:
    """The rig frames, their prior poses and their images, in time order.

    Args:
        frames_meta: The parsed metadata.

    Returns:
        The index `colsfm.loop_pose` walks to find a rig frame's neighbours and images.
    """
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    keyframe_by_rig_camera: dict[tuple[int, int], int] = {}
    for rig in rig_frames:
        for keyframe_id in rig.keyframe_ids:
            keyframe_by_rig_camera.setdefault((rig.synced_sample_id, keyframe_by_id[keyframe_id].camera_params_id), keyframe_id)
    return RigFrameIndex(
        sequence=tuple(rig.synced_sample_id for rig in rig_frames),
        world_T_rig={rig.synced_sample_id: rig.world_T_vehicle for rig in rig_frames},
        keyframe_by_rig_camera=keyframe_by_rig_camera,
    )
