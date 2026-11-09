"""GSI Egocentric Six-View: download through GSI's API, and repack its shipped rrds into the exoego:v2 layout.

GSI ships every episode as two Rerun 0.37 rrds under one recording id (application ``gsi_off_the_shelf``, recording
id = episode id): ``hands.rrd`` holds six H.265 camera streams with their pinholes and rig extrinsics, both hands'
MANO meshes, keypoints and parameters, a calibration blob and the episode properties; ``slam.rrd`` holds the rig
trajectory. Convert decodes no video. Lenses move the cameras, pinholes, videos and the SLAM pose onto
``/world/rig_00`` and turn GSI's FLU camera frames into RDF. The hands need each frame's SLAM pose to reach the world
frame, so they are read column-wise and written through the shared hand helpers. See docs/gsi-six-view.md.

``gsi_six_view_native`` keeps GSI's own layout as a temporary comparison mirror and reuses this module's download.
"""

import dataclasses
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from jaxtyping import Bool, Float32, Float64, Int64, UInt32
from numpy import ndarray
from rerun.encodings import EntityPathLike
from scipy.spatial.transform import Rotation
from simplecv.sensors.camera.base_camera import pixels_in_image

from dataforge import blueprints, hands, meshes, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig, RemoteSequence
from dataforge.datasets.gsi_six_view_remote import API_URL, GsiSession, Manifest, ManifestEpisode, fetch_episodes, fetch_manifest, read_manifest
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    FRAME_INDEX_FIELD,
    annotation_context,
    log_camera_source,
    log_dense_pose_track,
    log_rig_node,
    log_trail_segments,
    time_column,
)

FILES: tuple[str, str] = ("hands.rrd", "slam.rrd")
"""The two files GSI ships per episode."""
VIEWS: tuple[str, ...] = ("left_front", "right_front", "left_eye", "right_eye", "left_side", "right_side")
"""GSI's views in its calibration order, which fixes ``cam_00`` to ``cam_05``; ``cam_00`` (left_front) is the rig frame."""
PANE_ORDER: tuple[int, ...] = (2, 3, 0, 1, 4, 5)
"""The panes in eye, front and side pairs, left before right."""
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
MESH_BATCH: int = 256
"""Frames per hand-mesh chunk."""
SOURCE_SIZE: tuple[int, int] = (1920, 1080)
"""Every view's image, width by height; ``read_episode`` checks it against the pinholes."""
POSE_LAYER: str = "pose"
"""Hand keypoints, wrists, confidence and MANO records; the name DAS-Ego's pose layer uses."""
FOLLOW_EYE: rrb.EyeControls3D = blueprints.headset_eye_controls((0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
"""A chase eye in the rig frame (left_front's RDF frame: forward +z, up -y), aimed down at the hands."""


@dataclass
class GsiSixViewConfig(DataforgeDatasetConfig):
    """GSI's partner edition of Egocentric Six-View, fetched with the partner sign-in in ``GSI_PARTNER_USERNAME``/``GSI_PARTNER_PASSWORD``."""

    command: ClassVar[str] = "gsi-six-view"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: GsiSixViewDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "gsi-six-view")
    """Download root: ``manifest.json`` and ``episodes/<episode_id>/{hands,slam}.rrd`` until each episode is converted."""
    sequences: tuple[str, ...] | None = None
    """Episode ids to download and convert; None selects the whole edition."""
    edition: str = "01a0f166-7994-7b43-a622-a325541c4356"
    """GSI edition id: Egocentric Six-View, 10 h, release 2026-09-v2."""
    api_url: str = API_URL
    """GSI's API."""
    keep_download: bool = False
    """Keep each episode's download once its layers are published. Set it on the first of two datasets that convert the
    same downloads (``gsi-six-view`` and its native mirror); the second one deletes them."""


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


VIDEO_TIME_FIELD: pa.Field = (
    rrc.Chunk.from_columns("/", indexes=[time_column(np.array([0], dtype=np.int64))], columns=rr.Scalars.columns(scalars=[0.0]))
    .to_record_batch()
    .schema.field(schema.TIMELINE)
)
"""Rerun-derived index field of the ``video_time`` duration timeline."""


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


def rdf_world_T_rig_quaternion(quaternion: pa.FixedSizeListArray) -> pa.Array:
    """``world_T_rig`` for the RDF rig frame: world_R_flu · FLU_R_RDF; the rig origin does not move."""
    world_R_flu: Rotation = Rotation.from_quat(np.asarray(quaternion.flatten(), dtype=np.float64).reshape(-1, 4))
    turned: Float64[ndarray, "n 4"] = (world_R_flu * Rotation.from_matrix(FLU_R_RDF)).as_quat()
    return pa.FixedSizeListArray.from_arrays(pa.array(turned.reshape(-1), type=quaternion.type.value_type), 4)


def base_stream(hands_rrd: Path, slam_rrd: Path, t0_ns: int) -> rrc.LazyChunkStream:
    """GSI's cameras, pinholes, videos and SLAM rig at their exoego paths, retimed and turned to RDF; no sample is decoded."""
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
    slam: rrc.LazyChunkStream = rrc.RrdReader(slam_rrd).stream().filter(content=["/slam/rig", "/slam/path", "/slam/index", "/slam/meta"]).flat_map(retimer)
    slam = slam.map(
        renamed({"/slam/rig": schema.rig_path(RIG), "/slam/path": schema.trajectory_path(RUN_SOURCE), "/slam/index": f"{run}/index", "/slam/meta": f"{run}/meta"})
    )
    slam = slam.lenses(
        rrc.MutateLens("Transform3D:quaternion", rrc.Selector(".").pipe(rdf_world_T_rig_quaternion)), content=schema.rig_path(RIG), output_mode="forward_unmatched"
    )
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
    """The SLAM pose of GSI's rig frame on every frame."""
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
    store: rrc.ChunkStore = rrc.RrdReader(hands_rrd).stream().filter(content=["/hand/**", "/camera/**", "/meta/**", "/__properties/episode"]).collect()
    slam: rrc.ChunkStore = rrc.RrdReader(slam_rrd).stream().filter(content=["/slam/rig", "/hand"]).collect()
    poses: pa.Table = frame_rows(slam, "/slam/rig")
    hand_poses: pa.Table = frame_rows(slam, "/hand")
    quaternions: Float64[ndarray, "f 4"] = stacked(column(poses, "Transform3D:quaternion"), 4).astype(np.float64)
    translations: Float64[ndarray, "f 3"] = stacked(column(poses, "Transform3D:translation"), 3).astype(np.float64)
    if not (np.array_equal(quaternions, stacked(column(hand_poses, "Transform3D:quaternion"), 4)) and np.array_equal(translations, stacked(column(hand_poses, "Transform3D:translation"), 3))):
        raise ValueError(f"{slam_rrd}: /hand and /slam/rig carry different poses")
    world_T_rig: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (len(poses), 1, 1))
    world_T_rig[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    world_T_rig[:, :3, 3] = translations
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
    times: pa.Table = frame_rows(store, "/hand/left/keypoints3d")
    for side in SIDES:
        keypoints.append(stacked(column(frame_rows(store, f"/hand/{side}/keypoints3d"), "Points3D:positions"), 21 * 3).reshape(-1, 21, 3))
        mesh: pa.Table = frame_rows(store, f"/hand/{side}/vertices")
        vertices.append(stacked(column(mesh, "Mesh3D:vertex_positions"), 778 * 3).reshape(-1, 778, 3))
        triangles: list = column(mesh, "Mesh3D:triangle_indices").to_pylist()
        if any(rows != triangles[0] for rows in triangles):
            raise ValueError(f"{hands_rrd}: the {side} hand changes its triangle list")
        faces.append(np.asarray(triangles[0], dtype=np.uint32))
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


def to_world(world_T_rig: Float64[ndarray, "f 4 4"], points: Float32[ndarray, "f k 3"]) -> Float32[ndarray, "f k 3"]:
    """Rig-frame points of every frame in the world frame."""
    moved: Float64[ndarray, "f k 3"] = np.einsum("fij,fkj->fki", world_T_rig[:, :3, :3], points.astype(np.float64)) + world_T_rig[:, None, :3, 3]
    return moved.astype(np.float32)


def coco133(joints: Float32[ndarray, "f 2 21 d"]) -> Float32[ndarray, "f 133 d"]:
    """GSI's joints in their COCO-133 hand slots (91-111 left, 112-132 right); nothing derived."""
    return np.stack([hands.coco133_from_coco_hands(frame) for frame in joints])


def project(rig_T_cam: Float64[ndarray, "4 4"], intrinsics: Float64[ndarray, "3 3"], points: Float32[ndarray, "f k 3"]) -> Float32[ndarray, "f k 2"]:
    """Rig-frame points through one view's shipped pinhole; NaN behind the camera or off the image."""
    cam_T_rig: Float64[ndarray, "4 4"] = np.linalg.inv(rig_T_cam)
    in_cam: Float64[ndarray, "f k 3"] = points.astype(np.float64) @ cam_T_rig[:3, :3].T + cam_T_rig[:3, 3]
    optical: Float64[ndarray, "f k 3"] = in_cam @ FLU_R_RDF
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels: Float64[ndarray, "f k 2"] = (optical @ intrinsics.T)[..., :2] / optical[..., 2:]
    flat: Float64[ndarray, "n 2"] = pixels.reshape(-1, 2)
    inside: Bool[ndarray, "n"] = pixels_in_image(flat, *SOURCE_SIZE) & (optical.reshape(-1, 3)[:, 2] > 0.0)
    return np.where(inside[:, None], flat, np.nan).reshape(pixels.shape).astype(np.float32)


def write_pose_layer(recording: rr.RecordingStream, episode: Episode) -> None:
    """The ``pose`` layer: COCO-133 hand keypoints and both wrists in the world, each hand's confidence and GSI's MANO record."""
    world_keypoints: Float32[ndarray, "f 2 21 3"] = to_world(episode.world_T_rig, episode.rig_keypoints.reshape(len(episode.frames), -1, 3)).reshape(-1, 2, 21, 3)
    hands.log_keypoints3d(recording, times_ns=episode.times_ns, frame_indices=episode.frames, positions=coco133(world_keypoints), confidence=None)
    indexes: list[rr.TimeColumn] = [time_column(episode.times_ns), rr.TimeColumn(schema.FRAME_INDEX, sequence=episode.frames)]
    for side_index, side in enumerate(SIDES):
        hands.log_hand_confidence(recording, side, times_ns=episode.times_ns, frame_indices=episode.frames, confidence=episode.usable[:, side_index].astype(np.float64))
        rig_R_wrist: Float64[ndarray, "f 3 3"] = np.asarray(episode.mano[f"global_orient_{side}"].flatten(), dtype=np.float64).reshape(-1, 3, 3)
        world_T_wrist: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (len(episode.frames), 1, 1))
        world_T_wrist[:, :3, :3] = episode.world_T_rig[:, :3, :3] @ rig_R_wrist
        world_T_wrist[:, :3, 3] = world_keypoints[:, side_index, 0]
        log_dense_pose_track(recording, schema.hand_wrist_path(side), times_ns=episode.times_ns, frame_indices=episode.frames, transforms=world_T_wrist)
        rr.send_columns(
            schema.hand_mano_path(side),
            indexes=indexes,
            columns=rr.AnyValues.columns(**{name: episode.mano[f"{name}_{side}"] for name in ("hand_pose", "betas", "global_orient", "transl", "is_detected", "is_interpolated", "is_usable")}),
            recording=recording,
        )
    recording.send_property(
        POSE_LAYER,
        rr.AnyValues(
            keypoints="GSI's 21 joints per hand in COCO-133 slots 91-132 (GSI ships them in COCO-WholeBody hand order); no derived slots",
            keypoint_confidence="1.0 for every finite joint: GSI ships no keypoint score",
            hand_confidence="GSI's is_usable flag",
            wrist="translation = joint 0 in the world, rotation = world_R_rig · MANO global_orient",
            mano_frame="as shipped: global_orient and transl in GSI's rig frame (left_front, FLU), not the world",
            coverage_left=float(episode.usable[:, 0].mean()),
            coverage_right=float(episode.usable[:, 1].mean()),
        ),
    )


def write_hand_mesh_layer(recording: rr.RecordingStream, episode: Episode) -> None:
    """The ``hand_mesh`` layer: GSI's skinned MANO meshes, moved to the world frame, on the frames GSI marks usable."""
    for side_index, side in enumerate(SIDES):
        path: str = schema.hand_mesh_path(side)
        meshes.log_mesh_static(recording, path, albedo_factor=hands.HAND_ALBEDO[side])
        for start in range(0, len(episode.frames), MESH_BATCH):
            batch: slice = slice(start, start + MESH_BATCH)
            trusted: Bool[ndarray, "n"] = episode.usable[batch, side_index]
            world_vertices: Float32[ndarray, "n 778 3"] = to_world(episode.world_T_rig[batch], episode.rig_vertices[batch, side_index])
            meshes.log_mesh_batch(
                recording, path, times_ns=episode.times_ns[batch], frame_indices=episode.frames[batch],
                vertices=world_vertices[trusted], trusted=trusted.tolist(), topology=episode.faces[side_index],
            )  # fmt: skip
    recording.send_property(paths.HAND_MESH_LAYER, rr.AnyValues(model="MANO vertices as GSI skinned them, moved from its rig frame to the world by the SLAM pose"))


def write_projections_layer(recording: rr.RecordingStream, episode: Episode) -> None:
    """The ``projections`` layer: the COCO-133 hand keypoints through each view's shipped pinhole (GSI's videos are undistorted)."""
    rig_coco: Float32[ndarray, "f 133 3"] = coco133(episode.rig_keypoints)
    confidence: Float32[ndarray, "f 133"] = np.isfinite(rig_coco).all(axis=-1).astype(np.float32)

    def pixels_by_camera() -> Iterator[tuple[str, Float32[ndarray, "f 133 2"]]]:
        for cam in range(len(VIEWS)):
            yield schema.coco133_uv_projected_path(RIG, cam), project(episode.rig_T_cams[cam], episode.intrinsics[cam], rig_coco)

    hands.log_projections_property(recording, camera_model="pinhole", calibration_source="GSI's Pinhole per view (= /meta/calibration K)")
    hands.log_projections(recording, pixels_by_camera(), times_ns=episode.times_ns, frame_indices=episode.frames, confidence=confidence)


def trail_overrides() -> dict[EntityPathLike, rrb.VisibleTimeRanges | rr.LineStrips3D]:
    """The SLAM trail through the shared cursor window over the whole path, drawn thin and dim as in ``blueprints.rig_blueprint``'s follow view."""
    return {
        schema.trail_path(RUN_SOURCE): blueprints.trail_window_range(),
        schema.trajectory_path(RUN_SOURCE): rr.LineStrips3D.from_fields(
            colors=blueprints.DIM_TRAJECTORY_COLOR, radii=rr.Radius.ui_points(blueprints.DIM_TRAJECTORY_RADIUS_UI_POINTS)
        ),
    }


class GsiSixViewDataset(DataforgeDataset[GsiSixViewConfig, Path]):
    """One catalog dataset in the exoego:v2 layout; one segment per episode."""

    layers: tuple[str, ...] = (paths.BASE_LAYER, POSE_LAYER, paths.HAND_MESH_LAYER, paths.PROJECTIONS_LAYER)

    @property
    def manifest_path(self) -> Path:
        """Where ``download`` keeps the edition manifest."""
        return self.config.root / "manifest.json"

    def download(self) -> None:
        """Fetch the manifest, then every selected episode that is neither on disk nor converted."""
        session: GsiSession = GsiSession.sign_in(self.config.api_url)
        fetch_manifest(session, self.config.edition, self.manifest_path)
        episodes: list[str] = [episode.episode_id for episode in read_manifest(self.manifest_path).episodes]
        selected: list[str] = episodes if self.config.sequences is None else list(dict.fromkeys(self.config.sequences))
        unknown: list[str] = sorted(set(selected) - set(episodes))
        if unknown:
            raise ValueError(f"{self.config.command}: the edition has no episode(s) {unknown}; see `dataforge-download {self.config.command} --list-remote`")
        wanted: set[str] = {episode for episode in selected if not all(target.is_file() for target in self.targets(self.identity(episode)).values())}
        print(f"{self.config.command}: {len(wanted)} of {len(selected)} selected episode(s) still to fetch into {self.config.root}")
        if wanted:
            report: transports.FetchReport = fetch_episodes(session, self.config.edition, wanted, lambda episode, name: self.config.root / "episodes" / episode / name)
            print(f"{self.config.command}: {report.summary()}")

    def remote_sequences(self) -> list[RemoteSequence]:
        """Every episode of the manifest, fetched first when it is not on disk."""
        if not self.manifest_path.is_file():
            fetch_manifest(GsiSession.sign_in(self.config.api_url), self.config.edition, self.manifest_path)
        return [
            RemoteSequence(
                episode.episode_id,
                episode.hands_rrd.size_bytes + episode.slam_rrd.size_bytes,
                tuple(f"episodes/{episode.episode_id}/{name}" for name in FILES),
            )
            for episode in read_manifest(self.manifest_path).episodes
        ]

    def identity(self, episode_id: str) -> SequenceIdentity:
        return SequenceIdentity(self.config.name, (episode_id,))

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Every downloaded episode holding both files; a converted one has none left."""
        selected: set[str] | None = None if self.config.sequences is None else set(self.config.sequences)
        return [
            (self.identity(directory.name), directory)
            for directory in sorted((self.config.root / "episodes").glob("*"))
            if (selected is None or directory.name in selected) and all((directory / name).is_file() for name in FILES)
        ]

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write each pending layer, then delete the episode's download once every layer is published (unless ``keep_download``)."""
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        manifest: Manifest = read_manifest(self.manifest_path)
        episode: ManifestEpisode = next(entry for entry in manifest.episodes if entry.episode_id == identity.parts[0])
        self.timer.capture_s = episode.duration_s
        if pending:
            self.write_episode(identity, source, targets, pending, manifest, episode)
        if not self.config.keep_download and all(target.is_file() for target in targets.values()):
            for name in FILES:
                (source / name).unlink(missing_ok=True)
            source.rmdir()
        return targets[paths.BASE_LAYER]

    def write_episode(
        self, identity: SequenceIdentity, source: Path, targets: dict[str, Path], pending: list[str], manifest: Manifest, episode: ManifestEpisode
    ) -> None:
        """The base layer from GSI's cameras and SLAM rig, then the hand layers from one column-wise read."""
        hands_rrd, slam_rrd = (source / name for name in FILES)
        with self.timer.stage("read"):
            track: Episode = read_episode(hands_rrd, slam_rrd)
        if paths.BASE_LAYER in pending:
            with self.timer.stage(f"write:{paths.BASE_LAYER}"):
                self.write_base(identity, hands_rrd, slam_rrd, targets[paths.BASE_LAYER], manifest, episode, track)
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            POSE_LAYER: lambda recording: write_pose_layer(recording, track),
            paths.HAND_MESH_LAYER: lambda recording: write_hand_mesh_layer(recording, track),
            paths.PROJECTIONS_LAYER: lambda recording: write_projections_layer(recording, track),
        }
        self.write_layers(identity, targets, [layer for layer in pending if layer != paths.BASE_LAYER], writers)

    def write_base(
        self, identity: SequenceIdentity, hands_rrd: Path, slam_rrd: Path, target: Path, manifest: Manifest, episode: ManifestEpisode, track: Episode
    ) -> None:
        """The rig, its six cameras with GSI's pinholes and H.265 samples, the SLAM path and trail, and the capture properties.

        Written outside ``write_layers`` so that the base keeps GSI's ``RecordingInfo`` start time, the capture's own.
        """
        shipped: rrc.ChunkStore = rrc.RrdReader(hands_rrd).stream().filter(content=["/meta/calibration", "/__properties/episode"]).collect()
        start_ns: int = int(static_value(shipped, "/__properties/episode", "start_ns")[0])
        frames: int = int(static_value(shipped, "/__properties/episode", "n_frames")[0])
        compacted: rrc.ChunkStore = base_stream(hands_rrd, slam_rrd, start_ns).collect(optimize=OPTIMIZE)
        calibration: pa.Table = shipped.reader(index=None, contents="/meta/calibration").to_arrow_table()
        extras: dict[str, pa.Array] = {
            name.rsplit(":", 1)[1]: calibration.column(name).combine_chunks().flatten()
            for name in calibration.column_names
            if name.startswith("/meta/calibration:") and name.rsplit(":", 1)[1] not in CALIBRATION_IN_ENTITIES
        }
        with (
            writing.atomic_write(target) as temp_path,
            writing.recording_to(temp_path, recording_id=identity.recording_id, default_blueprint=self.default_blueprint(), send_properties=False) as recording,
        ):
            rr.log("/", rr.ViewCoordinates.FLU, static=True, recording=recording)
            rr.log("/", annotation_context(), static=True, recording=recording)
            log_rig_node(recording, RIG, reference=schema.cam_path(RIG, 0).rsplit("/", 1)[1], num_cameras=len(VIEWS), name="gsi-six-view", kind="ego")
            resolutions: list[str] = []
            for cam, view in enumerate(VIEWS):
                rr.log(schema.cam_path(RIG, cam), rr.AnyValues(name=view, kind="rgb", camera_model="pinhole"), static=True, recording=recording)
                resolutions.append(
                    log_camera_source(recording, RIG, cam, name=view, source_width=SOURCE_SIZE[0], source_height=SOURCE_SIZE[1], video_codec="h265", source_num_frames=frames)
                )
            recording.send_chunks(compacted)
            # The rig origin is the same in GSI's FLU frame and the RDF one, so the trail reads GSI's translations.
            log_trail_segments(recording, schema.trail_path(RUN_SOURCE), times_ns=track.times_ns, translations_xyz=track.world_T_rig[:, :3, 3])
            recording.send_chunks(rrc.RrdReader(hands_rrd).stream().filter(content="/__properties/episode"))
            recording.send_chunks(rrc.RrdReader(hands_rrd).stream().filter(content="/__properties", components="RecordingInfo:start_time"))
            recording.send_property("gsi_calibration", rr.AnyValues(**extras))
            writing.send_capture_properties(
                recording,
                identity,
                num_cameras=len(VIEWS),
                num_frames=frames,
                category=episode.category,
                camera_names=",".join(VIEWS),
                source_resolution=resolutions,
                gsi_edition=self.config.edition,
                gsi_source_release=manifest.source_release,
                gsi_rrd_contract=manifest.rrd_contract,
                gsi_privacy_status=manifest.privacy_status,
                hands_rrd_sha256=episode.hands_rrd.sha256,
                slam_rrd_sha256=episode.slam_rrd.sha256,
            )

    def world_contents(self) -> list[str]:
        """The rig with its frustums and videos, the SLAM path, the hands; no 2D projections on the image planes."""
        return ["+ /world/**", *(f"- {schema.coco133_uv_projected_path(RIG, cam)}" for cam in range(len(VIEWS)))]

    def pane_contents(self, cam: int) -> list[str]:
        """A view's video, its projected keypoints, and both meshes drawn live through its pinhole."""
        return [schema.video_path(RIG, cam), schema.coco133_uv_projected_path(RIG, cam), *(schema.hand_mesh_path(side) for side in SIDES)]

    def default_blueprint(self) -> rrb.Blueprint:
        """The wearer's hands and path in 3D, riding the rig, beside the six views in eye, front and side pairs."""
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="GSI six-view", origin=schema.rig_path(RIG), contents=self.world_contents(), eye_controls=FOLLOW_EYE, line_grid=False, overrides=trail_overrides()
                ),
                rrb.Grid(
                    *(blueprints.camera_view(VIEWS[cam], RIG, cam, contents=self.pane_contents(cam)) for cam in PANE_ORDER),
                    grid_columns=2,
                    name="Cameras: eye, front, side",
                ),
                column_shares=[2, 3],
            ),
            rrb.TimePanel(timeline=schema.TIMELINE, state="collapsed"),
            collapse_panels=True,
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """The hands in 3D without any video, beside left_front with its hands: the one stream a card decodes."""
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Hands",
                origin=schema.rig_path(RIG),
                contents=[*self.world_contents(), *blueprints.video_exclusions((RIG, cam) for cam in range(len(VIEWS)))],
                eye_controls=FOLLOW_EYE,
                line_grid=False,
                overrides=trail_overrides(),
            ),
            blueprints.camera_view(VIEWS[0], RIG, 0, contents=self.pane_contents(0)),
        )

    def table_fields(self) -> writing.TableFields:
        shown: tuple[writing.TableField, ...] = (
            writing.TableField("property:capture:category", "category"),
            writing.TableField("property:episode:caption", "caption"),
            writing.TableField("property:episode:duration_s", "seconds"),
        )
        return writing.TableFields(cards=shown, table=shown)

