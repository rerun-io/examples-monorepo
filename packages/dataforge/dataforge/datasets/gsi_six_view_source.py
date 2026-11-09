"""GSI Egocentric Six-View, reading side: the shipped rrds as exoego:v2 base chunks, and one episode's hands, poses and calibration.

GSI's cameras are FLU and its rig frame is left_front's FLU body frame; exoego:v2 wants RDF cameras and an identity
reference camera, so ``base_stream``'s lenses turn every pinhole, ``rig_T_cam`` and the rig to left_front's RDF frame
without decoding a sample. ``read_episode`` reads the hands, the SLAM poses and the calibration column-wise for the
derived layers (``gsi_six_view_layers``). See docs/gsi-six-view.md.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Bool, Float32, Float64, Int64, UInt32
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import hands, schema
from dataforge.logging_toolkit import (
    FRAME_INDEX_FIELD,
    time_column,
)

FILES: tuple[str, str] = ("hands.rrd", "slam.rrd")
"""The two files GSI ships per episode."""
VIEWS: tuple[str, ...] = ("left_front", "right_front", "left_eye", "right_eye", "left_side", "right_side")
"""GSI's views in its calibration order, which fixes ``cam_00`` to ``cam_05``; ``cam_00`` (left_front) is the rig frame."""
RIG: int = 0
"""The one moving rig."""
RUN_SOURCE: str = "slam"
"""``/world/runs/slam``: GSI's SLAM path, the only trajectory the release ships."""
SIDES: tuple[hands.Side, ...] = ("left", "right")
"""GSI's two hands, ``/hand/<side>``."""
OPTIMIZE: rrc.OptimizationProfile = dataclasses.replace(rrc.OptimizationProfile.OBJECT_STORE, fix_keyframe=True)
"""Catalog compaction. GSI marks every non-key video sample ``is_keyframe=false``, which Rerun 0.42 rejects (the flag is a
sparse marker) and then leaves the video un-batched; ``fix_keyframe`` derives the flags from the H.265 samples instead."""
PINHOLE_COMPONENTS: list[str] = ["Pinhole:image_from_camera", "Pinhole:resolution", "Pinhole:camera_xyz", "Pinhole:image_plane_distance"]
"""What GSI logs for a pinhole on the entity that also carries its video."""
FLU_R_RDF: Float64[ndarray, "3 3"] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
"""An RDF camera direction in GSI's FLU camera frame (GSI's ``flu_from_opt``)."""
FLU: list[int] = [5, 4, 1]
"""``ViewCoordinates`` of GSI's camera frames: forward, left, up."""
RDF: list[int] = [3, 2, 5]
"""``ViewCoordinates`` of an exoego camera frame: right, down, forward."""
CALIBRATION_IN_ENTITIES: frozenset[str] = frozenset(
    {"K", "resolution", "views", "flu_from_opt", "P_view_from_left_front", "T_view_from_left_front", "T_view_optical_from_left_front"}
)
"""``/meta/calibration`` fields the camera entities already state (pinholes, ``rig_T_cam``, camera names); the rest becomes
the ``gsi_calibration`` property."""
SOURCE_SIZE: tuple[int, int] = (1920, 1080)
"""Every view's image, width by height; ``read_episode`` checks it against the pinholes."""

VIDEO_TIME_FIELD: pa.Field = (
    rrc.Chunk.from_columns("/", indexes=[time_column(np.array([0], dtype=np.int64))], columns=rr.Scalars.columns(scalars=[0.0]))
    .to_record_batch()
    .schema.field(schema.TIMELINE)
)
"""Rerun-derived index field of the ``video_time`` duration timeline."""


@dataclass(frozen=True, slots=True)
class Retimer:
    """Moves a GSI chunk onto the exoego clocks: ``video_time`` from the first frame, ``frame`` renamed ``frame_index``.

    ``capture_time`` stays. ``source_frame`` goes: it is ``frame`` plus the episode's ``source_frame_start``; so does an SDK ``log_time``. Rows
    without a ``frame`` go too: ``slam.rrd`` logs two such poses besides its one pose per frame.
    """

    t0_ns: int
    """``capture_time`` of the episode's first frame."""

    def __call__(self, chunk: rrc.Chunk) -> list[rrc.Chunk]:
        batch: pa.RecordBatch = chunk.to_record_batch()
        if schema.CAPTURE_TIMELINE not in batch.schema.names:
            return [chunk]
        if "frame" not in batch.schema.names:
            return []
        batch = batch.filter(batch.column("frame").is_valid())
        if batch.num_rows == 0:
            return []
        index: int = batch.schema.get_field_index("frame")
        batch = batch.set_column(index, FRAME_INDEX_FIELD, batch.column("frame").cast(pa.int64()))
        for dropped in ("source_frame", "log_time", "log_tick"):
            if dropped in batch.schema.names:
                batch = batch.remove_column(batch.schema.get_field_index(dropped))
        capture: Int64[ndarray, "n"] = np.asarray(batch.column(schema.CAPTURE_TIMELINE).cast(pa.int64()))
        video_time: pa.Array = pa.array(capture - self.t0_ns, type=pa.duration("ns"))
        return rrc.Chunk.from_record_batch(batch.add_column(1, VIDEO_TIME_FIELD, video_time))  # after the row ids


def renamed(mapping: dict[str, str]) -> Callable[[rrc.Chunk], rrc.Chunk]:
    """A chunk at its exoego path; every chunk the stream carries must have one."""
    return lambda chunk: chunk.with_entity_path(mapping[str(chunk.entity_path)])


def rdf_camera_xyz(camera_xyz: pa.FixedSizeListArray) -> pa.Array:
    """GSI's FLU camera frame as the exoego RDF one; the intrinsics already describe the optical frame."""
    if camera_xyz.to_pylist() != [FLU] * len(camera_xyz):
        raise ValueError(f"GSI pinholes changed their camera frame: {camera_xyz.to_pylist()}")
    return pa.array([RDF] * len(camera_xyz), type=camera_xyz.type)


def rdf_rig_T_cam_rotation(mat3x3: pa.FixedSizeListArray) -> pa.Array:
    """``rig_T_cam`` rotation between RDF frames: FLU_R_RDFᵀ · R · FLU_R_RDF (column-major in and out)."""
    rotations: Float64[ndarray, "n 3 3"] = np.asarray(mat3x3.flatten(), dtype=np.float64).reshape(-1, 3, 3).transpose(0, 2, 1)
    turned: Float64[ndarray, "n 3 3"] = FLU_R_RDF.T @ rotations @ FLU_R_RDF
    return pa.FixedSizeListArray.from_arrays(pa.array(turned.transpose(0, 2, 1).reshape(-1), type=mat3x3.type.value_type), 9)


def rdf_rig_T_cam_translation(translation: pa.FixedSizeListArray) -> pa.Array:
    """``rig_T_cam`` translation in the RDF rig frame: FLU_R_RDFᵀ · t."""
    offsets: Float64[ndarray, "n 3"] = np.asarray(translation.flatten(), dtype=np.float64).reshape(-1, 3)
    return pa.FixedSizeListArray.from_arrays(pa.array((offsets @ FLU_R_RDF).reshape(-1), type=translation.type.value_type), 3)


def base_stream(hands_rrd: Path, slam_rrd: Path, t0_ns: int) -> rrc.LazyChunkStream:
    """GSI's cameras, pinholes, videos and SLAM path at their exoego paths, retimed and turned to RDF; no sample is decoded.

    The rig pose is not here: ``write_base`` writes it dense, so a SLAM dropout gets its NaN row (exoego:v2).
    """
    retimer: Retimer = Retimer(t0_ns)
    cameras: rrc.LazyChunkStream = rrc.RrdReader(hands_rrd).stream().filter(content="/camera/**").flat_map(retimer)
    pinholes, cameras = cameras.split(components=PINHOLE_COMPONENTS)
    pinholes = pinholes.map(renamed({f"/camera/{view}/image": schema.pinhole_path(RIG, cam) for cam, view in enumerate(VIEWS)}))
    pinholes = pinholes.lenses(rrc.MutateLens("Pinhole:camera_xyz", rrc.Selector(".").pipe(rdf_camera_xyz)), output_mode="forward_unmatched")
    cameras = cameras.map(
        renamed(
            {
                **{f"/camera/{view}": schema.cam_path(RIG, cam) for cam, view in enumerate(VIEWS)},
                **{f"/camera/{view}/image": schema.video_path(RIG, cam) for cam, view in enumerate(VIEWS)},
            }
        )
    )
    cameras = cameras.lenses(
        [
            rrc.MutateLens("Transform3D:mat3x3", rrc.Selector(".").pipe(rdf_rig_T_cam_rotation)),
            rrc.MutateLens("Transform3D:translation", rrc.Selector(".").pipe(rdf_rig_T_cam_translation)),
        ],
        content=[schema.cam_path(RIG, cam) for cam in range(len(VIEWS))],
        output_mode="forward_unmatched",
    )
    run: str = schema.run_path(RUN_SOURCE)
    slam: rrc.LazyChunkStream = rrc.RrdReader(slam_rrd).stream().filter(content=["/slam/path", "/slam/index", "/slam/meta"]).flat_map(retimer)
    slam = slam.map(renamed({"/slam/path": schema.trajectory_path(RUN_SOURCE), "/slam/index": f"{run}/index", "/slam/meta": f"{run}/meta"}))
    return rrc.LazyChunkStream.merge(pinholes, cameras, slam)


@dataclass(frozen=True, slots=True)
class Episode:
    """What the derived layers need from one episode, read column-wise from both files; positions in metres."""

    frames: Int64[ndarray, "f"]
    """``frame_index``: 0 to f-1, every frame."""
    times_ns: Int64[ndarray, "f"]
    """``video_time`` of each frame."""
    rig_keypoints: Float32[ndarray, "f 2 21 3"]
    """Both hands' 21 joints in GSI's order (COCO-WholeBody hand order), in GSI's rig frame (left_front, FLU)."""
    rig_vertices: Float32[ndarray, "f 2 778 3"]
    """Both MANO meshes as GSI skinned them, in the rig frame."""
    faces: tuple[UInt32[ndarray, "t 3"], UInt32[ndarray, "t 3"]]
    """Each hand's triangle list; GSI logs one per frame and they never change."""
    mano: dict[str, pa.Array]
    """GSI's ``/hand/<side>/mano`` columns of both hands, stacked as ``<field>_<side>``, one row per frame."""
    usable: Bool[ndarray, "f 2"]
    """GSI's ``is_usable`` flag of each hand."""
    world_T_rig: Float64[ndarray, "f 4 4"]
    """The SLAM pose of GSI's rig frame on every frame; NaN where SLAM has none (a tracking dropout)."""

    @property
    def posed(self) -> Bool[ndarray, "f"]:
        """Frames with a SLAM pose."""
        return np.isfinite(self.world_T_rig).all(axis=(1, 2))
    rig_T_cams: Float64[ndarray, "c 4 4"]
    """Each view's shipped ``rig_T_cam``, FLU to FLU."""
    intrinsics: Float64[ndarray, "c 3 3"]
    """Each view's shipped K."""


def static_value(store: rrc.ChunkStore, entity: str, component: str) -> list:
    """The static value of one component."""
    table: pa.Table = store.reader(index=None, contents=entity).to_arrow_table()
    return table.column(next(name for name in table.column_names if name.endswith(component)))[0].as_py()


def frame_rows(store: rrc.ChunkStore, entity: str) -> pa.Table:
    """One entity's rows in ``frame`` order; every frame from 0 must be present once."""
    table: pa.Table = store.reader(index="frame", contents=entity).to_arrow_table()
    table = table.filter(table.column("frame").is_valid()).sort_by("frame")
    frames: Int64[ndarray, "f"] = np.asarray(table.column("frame").cast(pa.int64()))
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError(f"{entity}: GSI rows do not cover every frame once")
    return table


def slam_poses(store: rrc.ChunkStore, entity: str, num_frames: int) -> Float64[ndarray, "f 4 4"]:
    """One SLAM entity's pose on each of ``num_frames`` frames; NaN on a frame it skips (GSI's SLAM drops single frames).

    Raises:
        ValueError: A frame has two poses, or a pose sits outside the episode's frames.
    """
    table: pa.Table = store.reader(index="frame", contents=entity).to_arrow_table()
    table = table.filter(table.column("frame").is_valid())
    frames: Int64[ndarray, "n"] = np.asarray(table.column("frame").cast(pa.int64()))
    if len(np.unique(frames)) != len(frames) or frames.min() < 0 or frames.max() >= num_frames:
        raise ValueError(f"{entity}: GSI poses repeat a frame or fall outside frames 0-{num_frames - 1}")
    poses: Float64[ndarray, "f 4 4"] = np.full((num_frames, 4, 4), np.nan)
    poses[frames] = np.eye(4)
    poses[frames, :3, :3] = Rotation.from_quat(stacked(column(table, "Transform3D:quaternion"), 4).astype(np.float64)).as_matrix()
    poses[frames, :3, 3] = stacked(column(table, "Transform3D:translation"), 3)
    return poses


def column(table: pa.Table, component: str) -> pa.ChunkedArray:
    """A component column, whatever entity prefix the reader gave it."""
    return table.column(next(name for name in table.column_names if name.endswith(component)))


def stacked(values: pa.ChunkedArray, width: int) -> Float32[ndarray, "f k"]:
    """One fixed-size batch per row (a joint set, a mesh) as a dense float32 array."""
    return np.asarray(values.combine_chunks().flatten().flatten(), dtype=np.float32).reshape(len(values), width)


def read_episode(hands_rrd: Path, slam_rrd: Path) -> Episode:
    """Both hands, the SLAM poses and the shipped calibration of one episode.

    Raises:
        ValueError: The SLAM poses of ``/hand`` and ``/slam/rig`` differ, so GSI's hands would not sit in its rig frame;
            or a view's calibration differs from what the layout assumes.
    """
    store: rrc.ChunkStore = (
        rrc.RrdReader(hands_rrd)
        .stream()
        .filter(content=["/hand/**", "/camera/**", "/meta/**", "/__properties/episode"])
        .drop(components=["VideoStream:sample", "VideoStream:codec", "Mesh3D:vertex_normals"])
        .collect()
    )
    slam: rrc.ChunkStore = rrc.RrdReader(slam_rrd).stream().filter(content=["/slam/rig", "/hand"]).collect()
    times: pa.Table = frame_rows(store, "/hand/left/keypoints3d")
    num_frames: int = times.num_rows
    world_T_rig: Float64[ndarray, "f 4 4"] = slam_poses(slam, "/slam/rig", num_frames)
    if not np.array_equal(world_T_rig, slam_poses(slam, "/hand", num_frames), equal_nan=True):
        raise ValueError(f"{slam_rrd}: /hand and /slam/rig carry different poses")
    if static_value(store, "/meta/calibration", "views") != list(VIEWS):
        raise ValueError(f"{hands_rrd}: GSI's views are not {VIEWS}")
    rig_T_cams: Float64[ndarray, "c 4 4"] = np.tile(np.eye(4), (len(VIEWS), 1, 1))
    intrinsics: Float64[ndarray, "c 3 3"] = np.zeros((len(VIEWS), 3, 3))
    for cam, view in enumerate(VIEWS):
        rig_T_cams[cam, :3, :3] = np.asarray(static_value(store, f"/camera/{view}", "Transform3D:mat3x3"), dtype=np.float64).reshape(3, 3).T
        rig_T_cams[cam, :3, 3] = np.asarray(static_value(store, f"/camera/{view}", "Transform3D:translation"), dtype=np.float64).reshape(3)
        intrinsics[cam] = np.asarray(static_value(store, f"/camera/{view}/image", "Pinhole:image_from_camera"), dtype=np.float64).reshape(3, 3).T
        if tuple(np.asarray(static_value(store, f"/camera/{view}/image", "Pinhole:resolution")).reshape(2).astype(int).tolist()) != SOURCE_SIZE:
            raise ValueError(f"{hands_rrd}: {view} is not {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]}")
    keypoints: list[Float32[ndarray, "f 21 3"]] = []
    vertices: list[Float32[ndarray, "f 778 3"]] = []
    faces: list[UInt32[ndarray, "t 3"]] = []
    mano: dict[str, pa.Array] = {}
    usable: list[Bool[ndarray, "f"]] = []
    for side in SIDES:
        keypoints.append(stacked(column(frame_rows(store, f"/hand/{side}/keypoints3d"), "Points3D:positions"), 21 * 3).reshape(-1, 21, 3))
        mesh: pa.Table = frame_rows(store, f"/hand/{side}/vertices")
        vertices.append(stacked(column(mesh, "Mesh3D:vertex_positions"), 778 * 3).reshape(-1, 778, 3))
        triangles: pa.ListArray = column(mesh, "Mesh3D:triangle_indices").combine_chunks()
        counts: Int64[ndarray, "f"] = np.diff(np.asarray(triangles.offsets, dtype=np.int64))
        indices: UInt32[ndarray, "n"] = np.asarray(triangles.flatten().flatten(), dtype=np.uint32)
        if not (counts == counts[0]).all() or not (indices.reshape(len(counts), -1) == indices[: counts[0] * 3]).all():
            raise ValueError(f"{hands_rrd}: the {side} hand changes its triangle list")
        faces.append(indices[: counts[0] * 3].reshape(-1, 3))
        flags: pa.Table = frame_rows(store, f"/hand/{side}/mano")
        for name in ("hand_pose", "betas", "global_orient", "transl", "is_detected", "is_interpolated", "is_usable"):
            values: pa.Array = column(flags, name).combine_chunks()
            if name.startswith("is_") and len(values.flatten()) != len(values):
                raise ValueError(f"{hands_rrd}: the {side} hand's {name} is not one flag per frame")
            mano[f"{name}_{side}"] = values.flatten() if name.startswith("is_") else values
        usable.append(np.asarray(mano[f"is_usable_{side}"], dtype=bool))
    start_ns: int = int(static_value(store, "/__properties/episode", "start_ns")[0])
    capture: Int64[ndarray, "f"] = np.asarray(times.column(schema.CAPTURE_TIMELINE).cast(pa.int64()))
    return Episode(
        frames=np.arange(len(times), dtype=np.int64),
        times_ns=capture - start_ns,
        rig_keypoints=np.stack(keypoints, axis=1),
        rig_vertices=np.stack(vertices, axis=1),
        faces=(faces[0], faces[1]),
        mano=mano,
        usable=np.stack(usable, axis=1),
        world_T_rig=world_T_rig,
        rig_T_cams=rig_T_cams,
        intrinsics=intrinsics,
    )


