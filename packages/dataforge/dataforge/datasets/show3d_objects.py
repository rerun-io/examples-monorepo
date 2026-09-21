"""Independent object pose and geometry layers on the SHOW3D frame clocks."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import perspective_projection
from simplecv.umetrack_temp.generic_hand_model_numpy import LANDMARK

from dataforge import schema, writing
from dataforge.datasets.show3d_calibration import HeadsetCalibration, HeadsetPose
from dataforge.datasets.show3d_hands import HandFrame
from dataforge.datasets.show3d_mesh_source import MESH_REPO
from dataforge.datasets.show3d_object_source import ObjectFrame
from dataforge.datasets.show3d_source import OBJECT_POSE_VERSION, FrameClock, sparse_rows
from dataforge.identity import SequenceIdentity


@dataclass(frozen=True, slots=True)
class ObjectSanity:
    """Census metrics; none of these numbers suppresses a source row."""

    coverage: float
    """Fraction of all frames with positive object confidence."""
    in_ego_fov_fraction: float
    """Fraction of posed frames in either headset image; NaN if no poses."""
    palm_dist_median_m: float
    """Median distance to nearest available palm; NaN if no paired palms."""


def object_sanity(frames: list[ObjectFrame], cameras: Sequence[HeadsetCalibration], hands: list[HandFrame]) -> ObjectSanity:
    """Measure visibility and nearest-palm distance without changing pose validity."""
    posed: list[ObjectFrame] = [frame for frame in frames if frame.posed]
    hand_by_index: dict[int, HandFrame] = {frame.index: frame for frame in hands}
    visible: int = 0
    distances: list[float] = []
    intrinsics: list[Float64[ndarray, "3 3"]] = [
        np.array([[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        for camera in cameras
    ]
    for frame in posed:
        world_T_object: Float64[ndarray, "4 4"] | None = frame.world_T_object
        assert world_T_object is not None
        centre: Float64[ndarray, "3"] = world_T_object[:3, 3]
        for camera, k_matrix in zip(cameras, intrinsics, strict=True):
            pose: HeadsetPose | None = camera.T_WorldFromCamera_by_index.get(str(frame.index))
            if pose is None or pose.T_WorldFromCamera is None or pose.is_pose_valid is False:
                continue
            transform: Float64[ndarray, "4 4"] = pose.T_WorldFromCamera
            xyz: Float64[ndarray, "3"] = transform[:3, :3].T @ (centre - transform[:3, 3])
            if xyz[2] <= 0.0:
                continue
            uv: Float64[ndarray, "1 2"] = perspective_projection(xyz[None, :], k_matrix)
            u, v = uv[0]
            if 0.0 <= u < camera.ImageSizeX and 0.0 <= v < camera.ImageSizeY:
                visible += 1
                break
        hand: HandFrame | None = hand_by_index.get(frame.index)
        if hand is not None:
            palm_distances: list[float] = [
                float(np.linalg.norm(centre - pose.landmarks_3d_mm[LANDMARK.PALM_CENTER])) * 0.001
                for pose in hand.hand_poses.values()
                if pose.landmarks_3d_mm is not None
            ]
            if palm_distances:
                distances.append(min(palm_distances))
    return ObjectSanity(
        len(posed) / len(frames),
        visible / len(posed) if posed else float("nan"),
        float(np.median(distances)) if distances else float("nan"),
    )


def write_object_pose_layer(
    identity: SequenceIdentity,
    alias: str,
    clock: FrameClock,
    frames: list[ObjectFrame],
    metrics: ObjectSanity,
    target: Path,
    *,
    clock_offset_s: float,
) -> None:
    """Write every confidence, sparse proper transforms, and typed census metrics."""
    positions, values = sparse_rows(frames, lambda frame: frame.world_T_object)
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        transforms: Float64[ndarray, "n 4 4"] = np.asarray(values, dtype=np.float64).reshape(-1, 4, 4)
        clock.send_sparse(
            recording, schema.objects_path(alias), positions,
            rr.Transform3D.columns(
                translation=transforms[:, :3, 3] * 0.001,
                quaternion=Rotation.from_matrix(transforms[:, :3, :3]).as_quat(),
            ),
        )
        rr.send_columns(
            schema.object_confidence_path(alias),
            indexes=clock.indexes(slice(None)),
            columns=rr.Scalars.columns(scalars=[frame.confidence for frame in frames]),
            recording=recording,
        )
        recording.send_property(
            "object_pose",
            rr.AnyValues(
                version=pa.array([OBJECT_POSE_VERSION], type=pa.string()),
                clock_offset_s=pa.array([clock_offset_s], type=pa.float64()),
                **{field.name: pa.array([getattr(metrics, field.name)], type=pa.float64()) for field in fields(ObjectSanity)},
            ),
        )


def write_object_mesh_layer(
    identity: SequenceIdentity, alias: str, clock: FrameClock, frames: list[ObjectFrame], mesh_id: int, mesh: Path, target: Path
) -> None:
    """Publish the GLB in the object frame, retaining its metre node scale, visible only where trusted.

    The pose stream on the parent entity is sparse (posed frames only, as shipped), so the
    viewer's latest-at would keep the static mesh at the last pose through every unposed
    frame. The mesh entity therefore carries a temporal ``albedo_factor``: opaque white where the
    frame is trusted (confidence above the Hub's default threshold), fully transparent otherwise.
    Rows exist only where visibility changes; latest-at carries them. ``Clear`` cannot serve
    (a cleared parent puts the static mesh at the rig origin); a scale of 0 warns and still draws.
    The mesh entity also repeats the shipped confidence as ``Scalars`` on every frame, so the
    value behind the alpha is one click away in the viewer and one column away in a query.
    """
    with writing.atomic_recording(target, recording_id=identity.recording_id, send_properties=False) as recording:
        path: str = schema.object_mesh_path(alias)
        rr.log(path, rr.Asset3D(path=mesh), static=True, recording=recording)
        trusted: list[bool] = [frame.trusted for frame in frames]
        changes: list[int] = [i for i in range(len(frames)) if i == 0 or trusted[i] != trusted[i - 1]]
        albedo: Float32[ndarray, "k 4"] = np.asarray([[1.0, 1.0, 1.0, 1.0 if trusted[i] else 0.0] for i in changes], dtype=np.float32)
        rr.send_columns(path, indexes=clock.indexes(changes), columns=rr.Asset3D.columns(albedo_factor=albedo), recording=recording)
        rr.send_columns(
            path, indexes=clock.indexes(slice(None)), columns=rr.Scalars.columns(scalars=[frame.confidence for frame in frames]), recording=recording
        )
        recording.send_property(
            "object_mesh", rr.AnyValues(mesh_id=pa.array([mesh_id], type=pa.int64()), mesh_source=pa.array([MESH_REPO], type=pa.string()))
        )
