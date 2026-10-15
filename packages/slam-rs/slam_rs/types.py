"""slam-rs edge values: the rig the estimator is built from, and its poses with their Rerun mapping (per frameset, or a
segment's whole ``slam_rs`` pose table)."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import STATIC, TimeNs
from dagkit.types import Log
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde

from slam_rs.catalog_feed import rig_trajectory
from slam_rs.rig import CameraCalib, ImuCalib
from slam_rs.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class VioRig:
    """The fed cameras and the IMU of a segment's rig, as the feed reads them from the base layer (static, memory only)."""

    cameras: tuple[CameraCalib, ...]
    """Fed cameras in the order of a frameset's images, at the feed's downscale."""
    imu: ImuCalib
    """IMU calibration and noise model; ``cam_time_offset_ns`` takes ``video_time`` to the estimator's clock."""
    export_offset_ns: int
    """What an exported trajectory adds to its ``video_time`` stamps to land on the device clock."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RigPose:
    """The estimated world-from-rig pose of one frameset."""

    position_m: Float64[ndarray, " 3"]
    quaternion_wxyz: Float64[ndarray, " 4"]

    def to_rerun(self, root: str) -> list[Log]:
        return [Log(root, rr.Transform3D(translation=self.position_m, quaternion=rr.Quaternion(xyzw=np.roll(self.quaternion_wxyz, -1))))]

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "RigPose"]]:
        poses: Trajectory = rig_trajectory(table, root, root, timeline)
        for t_ns, position, quaternion in zip(poses.t_ns.tolist(), poses.position_m, poses.quaternion_wxyz, strict=True):
            yield t_ns, cls(position, quaternion)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class CatalogPoses:
    """A segment's whole SLAM pose table (``slam_rs``: the rig's Transform3D rows), sorted, finite; one STATIC value."""

    t_ns: Int64[ndarray, " p"]
    world_from_rig: Float64[ndarray, "p 4 4"]

    def to_rerun(self, root: str) -> list[Log]:
        raise NotImplementedError("CatalogPoses is read from the slam_rs layer, which RigPose writes one frameset at a time")

    @classmethod
    def contents(cls, root: str) -> list[str]:
        return [root]

    @classmethod
    def from_layer(cls, table: pa.Table, root: str, timeline: str) -> Iterator[tuple[TimeNs, "CatalogPoses"]]:
        """The rows ``RigPose.from_layer`` reads (one null rule: a row needs both components), without non-finite or zero
        rotations; nothing when no pose is usable."""
        poses: Trajectory = rig_trajectory(table, root, root, timeline)
        valid: Bool[ndarray, " r"] = (
            np.isfinite(poses.quaternion_wxyz).all(axis=1) & np.isfinite(poses.position_m).all(axis=1) & (np.linalg.norm(poses.quaternion_wxyz, axis=1) > 0)
        )
        order: Int64[ndarray, " p"] = np.flatnonzero(valid)[np.argsort(poses.t_ns[valid], kind="stable")]
        if not order.size:  # no usable pose: no table (the hands node then refuses the segment)
            return
        world_from_rig: Float64[ndarray, "p 4 4"] = np.tile(np.eye(4), (len(order), 1, 1))
        world_from_rig[:, :3, :3] = Rotation.from_quat(poses.quaternion_wxyz[order][:, [1, 2, 3, 0]]).as_matrix()  # xyzw, normalised
        world_from_rig[:, :3, 3] = poses.position_m[order]
        yield STATIC, cls(poses.t_ns[order], world_from_rig)
