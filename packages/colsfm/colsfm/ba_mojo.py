"""The bridge between a `pycolmap.Reconstruction` and the Mojo solver's array contract (`caspar_mojo.problem`).

`problem_from_reconstruction` applies the rules of COLMAP's CASPAR adapter (`bundle_adjustment_caspar.cc`) so both
solvers see the same problem: the observations of the config's images whose point has a 3D point (and a track at least
`min_track_length` long), ordered camera-first by `(camera_id, image_id)` and then by point2D index; points in id
order; a point free only when every observation of its track is in the problem; the gauge frame and the config's
constant poses fixed; `sensor_from_rig` free only for a non-reference sensor of a rig whose reference sensor has an
image in the problem; focal and distortion free together, the principal point never. An observation with nothing free
is left out, as CASPAR leaves it out.

What the contract does not carry raises `ValueError` instead of solving a different problem: a camera model outside
the four, a refined principal point, focal and distortion refined separately, the config's explicit variable or
constant points (CASPAR's "external" observations), the THREE_POINTS gauge, a frame whose images mix camera models
(CASPAR then keeps one unlinked pose per model), and a camera that is the reference sensor in one frame and not in
another.

Extraction reads the observations from `images.bin` and `points3D.bin` written to a RAM disk, because pycolmap has no
bulk accessor for points2D and a per-observation Python loop costs about 1 µs each (2.5 s for P1's 2.4 M).
"""

from __future__ import annotations

import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from caspar_mojo.problem import MAX_INTRINSICS, BaProblem, BaResult, CameraModel
from jaxtyping import Bool, Float64, Int32, Int64
from numpy import ndarray

CAMERA_MODEL_BY_NAME: Final[dict[str, CameraModel]] = {model.name: model for model in CameraModel}
"""COLMAP's model name to the contract's enum; the four models CASPAR projects."""

PRINCIPAL_POINT_COLUMNS: Final[dict[CameraModel, tuple[int, ...]]] = {
    CameraModel.PINHOLE: (2, 3),
    CameraModel.SIMPLE_RADIAL: (1, 2),
    CameraModel.OPENCV_FISHEYE: (2, 3),
    CameraModel.FULL_OPENCV: (2, 3),
}
"""Columns of an intrinsics row that stay as they are on write-back: the principal point is never refined."""

RAM_DISK: Final[Path] = Path("/dev/shm")

_POINT2D_RECORD: Final[np.dtype] = np.dtype([("xy", "<f8", 2), ("point3D_id", "<u8")])
_INVALID_POINT3D_ID: Final[int] = pycolmap.INVALID_POINT3D_ID


@dataclass(frozen=True, slots=True)
class ProblemIds:
    """COLMAP's ids for the rows of a `BaProblem`, for write-back and comparison."""

    camera_ids: Int64[ndarray, " n_cameras"]
    frame_ids: Int64[ndarray, " n_frames"]
    point3D_ids: Int64[ndarray, " n_points"]
    observation_image_ids: Int64[ndarray, " n_observations"]
    observation_point2D_idxs: Int64[ndarray, " n_observations"]
    """Index of the observation in its image's `points2D`."""
    camera_sensor_ids: Int64[ndarray, " n_cameras"]
    """`sensor_t.id` of each camera row, the key of its `sensor_from_rig` in its rig."""
    camera_rig_ids: Int64[ndarray, " n_cameras"]


@dataclass(frozen=True, slots=True)
class BridgedProblem:
    """A `BaProblem` together with the ids it came from."""

    problem: BaProblem
    ids: ProblemIds


@dataclass(frozen=True, slots=True)
class _Observations:
    """Every image's points2D, flattened in images.bin order."""

    image_ids: Int64[ndarray, " n"]
    point2D_idxs: Int64[ndarray, " n"]
    xy: Float64[ndarray, "n 2"]
    point3D_ids: Int64[ndarray, " n"]
    """-1 where the point2D has no 3D point."""


def _read_images_bin(path: Path) -> _Observations:
    data: bytes = path.read_bytes()
    (num_images,) = struct.unpack_from("<Q", data, 0)
    offset: int = 8
    image_ids: list[int] = []
    counts: list[int] = []
    blocks: list[ndarray] = []
    for _ in range(num_images):
        (image_id,) = struct.unpack_from("<I", data, offset)
        # image_id, qvec, tvec, camera_id, then the null-terminated name.
        name_end: int = data.index(b"\0", offset + 4 + 56 + 4)
        (num_points2D,) = struct.unpack_from("<Q", data, name_end + 1)
        start: int = name_end + 9
        blocks.append(np.frombuffer(data, _POINT2D_RECORD, num_points2D, start))
        offset = start + num_points2D * _POINT2D_RECORD.itemsize
        image_ids.append(image_id)
        counts.append(num_points2D)
    records: ndarray = np.concatenate(blocks) if blocks else np.zeros(0, _POINT2D_RECORD)
    counts_array: Int64[ndarray, " m"] = np.asarray(counts, np.int64)
    starts: Int64[ndarray, " m"] = np.cumsum(counts_array) - counts_array
    raw_ids: ndarray = records["point3D_id"]
    return _Observations(
        image_ids=np.repeat(np.asarray(image_ids, np.int64), counts_array),
        point2D_idxs=np.arange(len(records), dtype=np.int64) - np.repeat(starts, counts_array),
        xy=np.ascontiguousarray(records["xy"]),
        point3D_ids=np.where(raw_ids == _INVALID_POINT3D_ID, -1, raw_ids.astype(np.int64)),
    )


def _read_points3D(path: Path) -> tuple[Int64[ndarray, " p"], Float64[ndarray, "p 3"], Int64[ndarray, " p"]]:
    data: bytes = path.read_bytes()
    (num_points,) = struct.unpack_from("<Q", data, 0)
    offsets: Int64[ndarray, " p"] = np.empty(num_points, np.int64)
    offset: int = 8
    unpack = struct.Struct("<Q").unpack_from
    for index in range(num_points):
        offsets[index] = offset
        # id u64, xyz 3 f64, rgb 3 u8, error f64, track length u64, then (image_id u32, point2D_idx u32) per element.
        offset += 51 + 8 * unpack(data, offset + 43)[0]
    buffer: ndarray = np.frombuffer(data, np.uint8)
    ids: Int64[ndarray, " p"] = buffer[offsets[:, None] + np.arange(8)].copy().view("<u8").ravel().astype(np.int64)
    xyz: Float64[ndarray, "p 3"] = buffer[offsets[:, None] + 8 + np.arange(24)].copy().view("<f8").reshape(-1, 3)
    track_lengths: Int64[ndarray, " p"] = buffer[offsets[:, None] + 43 + np.arange(8)].copy().view("<u8").ravel().astype(np.int64)
    return ids, xyz, track_lengths


def _pose_row(pose: pycolmap.Rigid3d) -> Float64[ndarray, " 7"]:
    return np.concatenate([np.asarray(pose.rotation.quat), np.asarray(pose.translation)])


def _normalised(row: Float64[ndarray, " 7"]) -> pycolmap.Rigid3d:
    quat: Float64[ndarray, " 4"] = row[:4] / np.linalg.norm(row[:4])
    return pycolmap.Rigid3d(pycolmap.Rotation3d(quat), row[4:].copy())


IDENTITY_POSE: Final[tuple[float, ...]] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def problem_from_reconstruction(
    reconstruction: pycolmap.Reconstruction, config: pycolmap.BundleAdjustmentConfig, ba_options: pycolmap.BundleAdjustmentOptions
) -> BridgedProblem:
    """Read the bundle adjustment CASPAR would solve into the array contract.

    Args:
        reconstruction: The model; it is not changed.
        config: Images, constant poses, constant intrinsics and constant rig extrinsics, as for `create_default_bundle_adjuster`.
        ba_options: The refinement switches and `min_track_length`.

    Returns:
        The problem and the COLMAP ids of its rows.

    Raises:
        ValueError: For a problem the contract does not carry (module docstring), naming what.
    """
    if ba_options.refine_principal_point:
        raise ValueError("refine_principal_point is not supported: the Mojo solver keeps the principal point constant")
    if ba_options.refine_focal_length != ba_options.refine_extra_params:
        raise ValueError("refine_focal_length != refine_extra_params is not supported (CASPAR's merged focal_and_extra block)")
    if config.num_variable_points() or config.num_constant_points():
        raise ValueError("explicit variable/constant points in the config are not supported (CASPAR's external observations)")
    if config.fixed_gauge == pycolmap.BundleAdjustmentGauge.THREE_POINTS:
        raise ValueError("the THREE_POINTS gauge is not supported")
    image_ids: list[int] = sorted(config.images)
    images: dict[int, pycolmap.Image] = {image_id: reconstruction.image(image_id) for image_id in image_ids}

    camera_ids: list[int] = sorted({image.camera_id for image in images.values()})
    camera_row: dict[int, int] = {camera_id: row for row, camera_id in enumerate(camera_ids)}
    camera_model: Int32[ndarray, " c"] = np.zeros(len(camera_ids), np.int32)
    intrinsics: Float64[ndarray, "c 12"] = np.zeros((len(camera_ids), MAX_INTRINSICS))
    intrinsics_free: Bool[ndarray, " c"] = np.zeros(len(camera_ids), bool)
    refine_intrinsics: bool = ba_options.refine_focal_length and ba_options.refine_extra_params
    for row, camera_id in enumerate(camera_ids):
        camera: pycolmap.Camera = reconstruction.camera(camera_id)
        model: CameraModel | None = CAMERA_MODEL_BY_NAME.get(camera.model.name)
        if model is None:
            raise ValueError(f"camera {camera_id}: camera model {camera.model.name} is not supported by the Mojo solver")
        camera_model[row] = model
        params: Float64[ndarray, " k"] = np.asarray(camera.params)
        intrinsics[row, : len(params)] = params
        intrinsics_free[row] = refine_intrinsics and not config.has_constant_cam_intrinsics(camera_id)

    frame_ids: list[int] = sorted({image.frame_id for image in images.values()})
    frame_row: dict[int, int] = {frame_id: row for row, frame_id in enumerate(frame_ids)}
    frame_models: dict[int, int] = {}
    for image in images.values():
        if frame_models.setdefault(image.frame_id, int(camera_model[camera_row[image.camera_id]])) != camera_model[camera_row[image.camera_id]]:
            raise ValueError(f"frame {image.frame_id} mixes camera models: CASPAR keeps one unlinked pose per model")
    gauge_frames: set[int] = set()
    if (
        config.fixed_gauge == pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD
        and ba_options.refine_rig_from_world
        and not any(config.has_constant_rig_from_world_pose(image.frame_id) for image in images.values())
    ):
        first_ref: pycolmap.Image | None = next((images[i] for i in image_ids if images[i].is_ref_in_frame()), None)
        if first_ref is not None:
            gauge_frames.add(first_ref.frame_id)
    rig_from_world: Float64[ndarray, "f 7"] = np.stack([_pose_row(reconstruction.frame(frame_id).rig_from_world) for frame_id in frame_ids])
    pose_free: Bool[ndarray, " f"] = np.array(
        [
            ba_options.refine_rig_from_world and not config.has_constant_rig_from_world_pose(frame_id) and frame_id not in gauge_frames
            for frame_id in frame_ids
        ],
        bool,
    )

    rigs_with_ref: set[int] = {reconstruction.frame(image.frame_id).rig_id for image in images.values() if image.is_ref_in_frame()}
    sensor_from_rig: Float64[ndarray, "c 7"] = np.tile(np.array(IDENTITY_POSE), (len(camera_ids), 1))
    sensor_free: Bool[ndarray, " c"] = np.zeros(len(camera_ids), bool)
    camera_sensor_ids: Int64[ndarray, " c"] = np.full(len(camera_ids), -1, np.int64)
    camera_rig_ids: Int64[ndarray, " c"] = np.full(len(camera_ids), -1, np.int64)
    camera_is_ref: dict[int, bool] = {}
    for image in images.values():
        row = camera_row[image.camera_id]
        is_ref: bool = image.is_ref_in_frame()
        if camera_is_ref.setdefault(image.camera_id, is_ref) != is_ref:
            raise ValueError(f"camera {image.camera_id} is the reference sensor in some frames and not in others")
        if camera_sensor_ids[row] >= 0:
            continue
        sensor: pycolmap.sensor_t = image.data_id.sensor_id
        rig_id: int = reconstruction.frame(image.frame_id).rig_id
        camera_sensor_ids[row] = sensor.id
        camera_rig_ids[row] = rig_id
        if is_ref:
            continue
        free: bool = ba_options.refine_sensor_from_rig and not config.has_constant_sensor_from_rig_pose(sensor) and rig_id in rigs_with_ref
        sensor_row: Float64[ndarray, " 7"] = _pose_row(reconstruction.rig(rig_id).sensor_from_rig(sensor))
        if free:
            # CASPAR normalises a refined extrinsic before the solve; a constant one is used as stored.
            sensor_row[:4] /= np.linalg.norm(sensor_row[:4])
        sensor_from_rig[row] = sensor_row
        sensor_free[row] = free

    with tempfile.TemporaryDirectory(dir=RAM_DISK if RAM_DISK.is_dir() else None) as scratch:
        reconstruction.write_binary(scratch)
        observations: _Observations = _read_images_bin(Path(scratch) / "images.bin")
        all_point_ids, all_xyz, all_track_lengths = _read_points3D(Path(scratch) / "points3D.bin")

    by_id: Int64[ndarray, " q"] = np.argsort(all_point_ids)
    has_point: Bool[ndarray, " n"] = observations.point3D_ids >= 0
    candidate: Bool[ndarray, " n"] = has_point & np.isin(observations.image_ids, np.asarray(image_ids, np.int64))
    if ba_options.min_track_length > 0:
        length_of: Int64[ndarray, " n"] = np.zeros(len(candidate), np.int64)
        length_of[has_point] = all_track_lengths[by_id[np.searchsorted(all_point_ids, observations.point3D_ids[has_point], sorter=by_id)]]
        candidate &= length_of >= ba_options.min_track_length
    point3D_ids, num_in_problem = np.unique(observations.point3D_ids[candidate], return_counts=True)
    point_rows: Int64[ndarray, " p"] = by_id[np.searchsorted(all_point_ids, point3D_ids, sorter=by_id)]
    point_free: Bool[ndarray, " p"] = (all_track_lengths[point_rows] <= num_in_problem) & bool(ba_options.refine_points3D)
    points: Float64[ndarray, "p 3"] = all_xyz[point_rows]

    selected: Int64[ndarray, " o"] = np.flatnonzero(candidate)
    obs_image_ids: Int64[ndarray, " o"] = observations.image_ids[selected]
    image_camera_row: dict[int, int] = {image_id: camera_row[image.camera_id] for image_id, image in images.items()}
    image_frame_row: dict[int, int] = {image_id: frame_row[image.frame_id] for image_id, image in images.items()}
    lookup_ids: Int64[ndarray, " m"] = np.asarray(image_ids, np.int64)
    position: Int64[ndarray, " o"] = np.searchsorted(lookup_ids, obs_image_ids)
    obs_camera: Int32[ndarray, " o"] = np.asarray([image_camera_row[i] for i in image_ids], np.int32)[position]
    obs_frame: Int32[ndarray, " o"] = np.asarray([image_frame_row[i] for i in image_ids], np.int32)[position]
    obs_point: Int32[ndarray, " o"] = np.searchsorted(point3D_ids, observations.point3D_ids[selected]).astype(np.int32)
    keep: Bool[ndarray, " o"] = pose_free[obs_frame] | sensor_free[obs_camera] | intrinsics_free[obs_camera] | point_free[obs_point]
    obs_camera_ids: Int64[ndarray, " o"] = np.asarray(camera_ids, np.int64)[obs_camera]
    # Camera-first, then image, then point2D index: CASPAR's factor order (stable sort keeps images.bin's point2D order).
    ordering: Int64[ndarray, " k"] = np.lexsort((observations.point2D_idxs[selected], obs_image_ids, obs_camera_ids))
    ordering = ordering[keep[ordering]]

    problem = BaProblem(
        camera_model=camera_model,
        intrinsics=intrinsics,
        intrinsics_free=intrinsics_free,
        sensor_from_rig=sensor_from_rig,
        sensor_free=sensor_free,
        rig_from_world=rig_from_world,
        pose_free=pose_free,
        points=np.ascontiguousarray(points),
        point_free=point_free,
        pixels=np.ascontiguousarray(observations.xy[selected][ordering]),
        observation_frame=np.ascontiguousarray(obs_frame[ordering]),
        observation_camera=np.ascontiguousarray(obs_camera[ordering]),
        observation_point=np.ascontiguousarray(obs_point[ordering]),
    )
    ids = ProblemIds(
        camera_ids=np.asarray(camera_ids, np.int64),
        frame_ids=np.asarray(frame_ids, np.int64),
        point3D_ids=point3D_ids.astype(np.int64),
        observation_image_ids=obs_image_ids[ordering],
        observation_point2D_idxs=observations.point2D_idxs[selected][ordering],
        camera_sensor_ids=camera_sensor_ids,
        camera_rig_ids=camera_rig_ids,
    )
    return BridgedProblem(problem=problem, ids=ids)


def apply_result(reconstruction: pycolmap.Reconstruction, bridged: BridgedProblem, result: BaResult) -> None:
    """Write a solve back the way CASPAR's adapter does: only free nodes change, quaternions normalised.

    Args:
        reconstruction: The model the problem was read from, changed in place.
        bridged: The problem and its ids.
        result: The solver's refined state.
    """
    problem, ids = bridged.problem, bridged.ids
    for row in np.flatnonzero(problem.point_free):
        reconstruction.point3D(int(ids.point3D_ids[row])).xyz = result.points[row]
    for row in np.flatnonzero(problem.pose_free):
        reconstruction.frame(int(ids.frame_ids[row])).rig_from_world = _normalised(result.rig_from_world[row])
    for row in np.flatnonzero(problem.sensor_free):
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, int(ids.camera_sensor_ids[row]))
        reconstruction.rig(int(ids.camera_rig_ids[row])).set_sensor_from_rig(sensor, _normalised(result.sensor_from_rig[row]))
    for row in np.flatnonzero(problem.intrinsics_free):
        camera: pycolmap.Camera = reconstruction.camera(int(ids.camera_ids[row]))
        params: Float64[ndarray, " k"] = np.asarray(camera.params).copy()
        refined: Float64[ndarray, " k"] = result.intrinsics[row, : len(params)].copy()
        keep = list(PRINCIPAL_POINT_COLUMNS[CameraModel(int(problem.camera_model[row]))])
        refined[keep] = params[keep]
        camera.params = refined
