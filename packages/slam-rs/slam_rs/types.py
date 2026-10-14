"""slam-rs edge values: the rig the estimator is built from, and its poses with their Rerun mapping."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import TimeNs
from dagkit.types import Log
from jaxtyping import Float64
from numpy import ndarray
from serde import serde

from slam_rs.catalog_feed import _rig_trajectory
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
        poses: Trajectory = _rig_trajectory(table, root, root, timeline)
        for t_ns, position, quaternion in zip(poses.t_ns.tolist(), poses.position_m, poses.quaternion_wxyz, strict=True):
            yield t_ns, cls(position, quaternion)
