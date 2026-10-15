"""slam-rs edge values: the rig the estimator is built from, and its poses with their Rerun mapping (per frameset, or a
segment's whole ``slam_rs`` pose table); with the trajectory those poses form and the decoder of a layer's rig pose rows."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.time import STATIC, TimeNs
from dagkit.types import Log, flat
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import serde

from slam_rs.rig import RIG_ENTITY, TIMELINE, CameraCalib, ImuCalib


@dataclass(slots=True, frozen=True)
class Trajectory:
    """A time-stamped rigid-body trajectory in one world frame."""

    t_ns: Int64[ndarray, " n"]
    """Pose timestamps, integer nanoseconds on the estimator's clock."""
    position_m: Float64[ndarray, "n 3"]
    """World-frame positions, metres."""
    quaternion_wxyz: Float64[ndarray, "n 4"]
    """World-from-body rotations, **w-first** unit quaternions."""

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])


def empty_trajectory() -> Trajectory:
    """A trajectory with no poses.

    What a reference a run does not have looks like: a segment with no ``gt``
    layer. Callers then test :func:`len` rather than ``None``, and the one
    array shape is right for :func:`numpy.concatenate`.
    """
    return Trajectory(t_ns=np.zeros(0, dtype=np.int64), position_m=np.zeros((0, 3)), quaternion_wxyz=np.zeros((0, 4)))


def rig_trajectory(table: pa.Table, segment_id: str, entity: str = RIG_ENTITY, timeline: str = TIMELINE) -> Trajectory:
    """Return the rig trajectory of a pose table (a ``slam_rs`` layer, a ``gt`` window) on ``timeline``, Rerun XYZW to w-first.
    ``table`` holds one ``entity`` Transform3D per row, indexed by ``timeline``; ``segment_id`` identifies errors.
    Empty if neither component exists or the window has no pose. Raise ValueError naming the row if translation and rotation
    validity differs: a pose needs both from the same row. Dropping column nulls separately can give equal lengths but pair
    different rows. Rerun stores components independently and permits logging/clearing either alone, so valid recordings can
    have this layout; refusal is the only safe reading (S25 review).
    """
    translation_column: str = f"{entity}:Transform3D:translation"
    quaternion_column: str = f"{entity}:Transform3D:quaternion"
    if translation_column not in table.column_names or quaternion_column not in table.column_names:
        return empty_trajectory()
    row_t_ns: Int64[ndarray, " n_rows"] = np.asarray(table[timeline].combine_chunks().cast(pa.int64()))
    translations: pa.Array = table[translation_column].combine_chunks()
    quaternions: pa.Array = table[quaternion_column].combine_chunks()
    valid: Bool[ndarray, " n_rows"] = translations.is_valid().to_numpy(zero_copy_only=False)
    quaternion_valid: Bool[ndarray, " n_rows"] = quaternions.is_valid().to_numpy(zero_copy_only=False)
    if not np.array_equal(valid, quaternion_valid):
        disagreeing: Int64[ndarray, " n_disagreeing"] = np.flatnonzero(valid != quaternion_valid)
        raise ValueError(
            f"{segment_id}: {disagreeing.size} of {row_t_ns.size} rig rows carry a translation without a quaternion or "
            f"a quaternion without a translation, the first at {int(row_t_ns[disagreeing[0]])} ns; a pose needs both, "
            f"and reading each component off its own rows would give it another row's rotation"
        )
    # Equal masks: each component's null removal drops exactly `valid`'s false rows, matching the timestamps' mask.
    position_m: Float64[ndarray, "n_poses 3"] = flat(translations, np.float64).reshape(-1, 3)
    if position_m.shape[0] == 0:
        return empty_trajectory()
    quaternion_xyzw: Float64[ndarray, "n_poses 4"] = flat(quaternions, np.float64).reshape(-1, 4)
    return Trajectory(
        t_ns=row_t_ns[valid],
        position_m=position_m,
        quaternion_wxyz=np.column_stack([quaternion_xyzw[:, 3], quaternion_xyzw[:, 0:3]]),
    )


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
