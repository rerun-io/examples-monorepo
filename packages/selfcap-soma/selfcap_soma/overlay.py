"""Write a layer aligned with SelfCap's original recording and camera tree."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
import trimesh
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader

from selfcap_soma.camera_geometry import project, project_ego
from selfcap_soma.catalog import ExoSource
from selfcap_soma.evaluation_io import first, missing_camera_calibration, nearest_indices, read_egohand_rgb, read_views
from selfcap_soma.observations import overlay_points, require_full_rate_mesh, video_time_column
from selfcap_soma.overlay_style import observed_links, observed_points, residual_tolerance


@dataclass(frozen=True, slots=True)
class Config:
    upper_body_only: bool = True
    """Crop the mesh and display only head/ears, arms/hands and hips."""
    native_exo: bool = False
    """Use native exo/Quest projection: omit baked exo overlays; Quest has none; OAK stays baked."""
    exo_source: ExoSource = "sapiens2"
    """Per-phone image evidence used to classify observed versus inferred joints."""
    source: Path = Path("/tmp/soma-fit/full-segment/fit.rrd")
    """Fitted geometry recording; no input videos are copied."""
    catalog_url: str = "rerun+http://dgx-spark:51235"
    """Catalog to read, without changing any source layer."""
    dataset: str = "selfcap-post-dev"
    """Dataset containing this segment's inputs and calibrated cameras."""
    segment: str = "selfcap__2025-12-03__d31bf1da__ep001"
    """Recording/segment ID used by the SelfCap converter."""
    application_id: str = "dataforge"
    """Application ID written by the SelfCap converter."""
    reference: Path | None = None
    """Original layer defining the application and recording IDs."""
    output: Path = Path("/tmp/soma-fit/layer/selfcap__2025-12-03__d31bf1da__ep001.rrd")
    """Local layer file; registration is a separate explicit action."""


def log_joint_overlay(
    recording: rr.RecordingStream, path: str, times: Float64[ndarray, "n"], pixels: Float32[ndarray, "n 133 2"], seen: Bool[ndarray, "n 133"]
) -> None:
    """Keep points and individually styled links on the existing joint entity."""
    body_edges = [(3, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
    hand_edges = [(base, base + 1 + finger * 4) for base in (91, 112) for finger in range(5)] + [
        (base + 1 + finger * 4 + joint, base + 2 + finger * 4 + joint) for base in (91, 112) for finger in range(5) for joint in range(3)
    ]
    edges = np.asarray(body_edges + hand_edges, dtype=np.int64)
    recording.log(
        path,
        rr.AnnotationContext(rr.ClassDescription(info=rr.AnnotationInfo(id=0))),
        static=True,
    )
    recording.log(path, rr.Points2D.from_fields(keypoint_ids=np.arange(133), class_ids=0), static=True)
    recording.send_columns(
        path,
        indexes=[video_time_column(times)],
        columns=rr.Points2D.columns(
            positions=pixels.reshape(-1, 2),
            colors=np.where(seen[..., None], np.array([0, 255, 255, 255]), np.array([140, 140, 140, 110])).astype(np.uint8).reshape(-1, 4),
            radii=np.where(seen, 2.5, 1.1).astype(np.float32).ravel(),
        ).partition([133] * len(times)),
    )
    supported = observed_links(seen, edges)
    recording.send_columns(
        path,
        indexes=[video_time_column(times)],
        columns=rr.LineStrips2D.columns(
            strips=pixels[:, edges].reshape(-1, 2, 2),
            colors=np.where(supported[..., None], np.array([0, 255, 255, 255]), np.array([130, 130, 130, 90])).astype(np.uint8).reshape(-1, 4),
            radii=np.where(supported, 1.2, 0.55).astype(np.float32).ravel(),
        ).partition([len(edges)] * len(times)),
    )


def mesh_vertex_rows(column: pa.Array | pa.ChunkedArray) -> Float32[ndarray, "n v 3"]:
    """View uniform mesh rows in Arrow numeric buffers, retaining slice offsets."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        raise ValueError("mesh vertex rows must not be null")
    lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
    if len(lengths) and np.any(lengths != lengths[0]):
        raise ValueError("mesh vertex rows must have uniform vertex counts")
    vectors = pc.list_flatten(array)
    if vectors.null_count:
        raise ValueError("mesh vertices must not be null")
    values = pc.list_flatten(vectors).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    count = int(lengths[0]) if len(lengths) else 0
    return values.reshape(len(array), count, 3)


def log_cropped_mesh(recording: rr.RecordingStream, source: RrdReader, times: Float64[ndarray, "n"]) -> None:
    """Retain the fixed rest-pose waist selection, with newly computed normals."""
    selection = source.stream().filter(content="/parameters/upper_body_vertices").to_chunks()
    if not selection:
        raise ValueError("fit is missing its rest-pose upper-body vertex selection")
    ids = np.asarray(selection[0].to_record_batch().column("Tensor:data")[0].as_py()[0]["buffer"], dtype=np.int64)
    faces = None
    batches = []
    for chunk in source.stream().filter(content="/world/fit/mesh").to_chunks():
        batch = chunk.to_record_batch()
        if "Mesh3D:triangle_indices" in batch.schema.names:
            faces = np.asarray(batch.column("Mesh3D:triangle_indices")[0].as_py(), dtype=np.int32)
        if "video_time" in batch.schema.names and "Mesh3D:vertex_positions" in batch.schema.names:
            batches.append(batch)
    if faces is None or not batches:
        raise ValueError("fit mesh has no faces or vertex frames")
    mesh_times = np.concatenate([batch.column("video_time").cast("int64").to_numpy() for batch in batches])
    require_full_rate_mesh(mesh_times, np.rint(times * 1e9).astype(np.int64))
    remap = np.full(int(faces.max()) + 1, -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    faces = remap[faces[(remap[faces] >= 0).all(1)]].astype(np.int32)
    recording.log("/world/pred/soma_x/mesh", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[0.25, 0.65, 0.8, 0.5]), static=True)
    for batch in batches:
        stamps = batch.column("video_time").cast("int64").to_numpy().astype(np.float64) / 1e9
        vertices = mesh_vertex_rows(batch.column("Mesh3D:vertex_positions"))[:, ids]
        normals = np.stack([trimesh.Trimesh(v, faces, process=False).vertex_normals for v in vertices]).astype(np.float32)
        recording.send_columns(
            "/world/pred/soma_x/mesh",
            indexes=[video_time_column(stamps)],
            columns=rr.Mesh3D.columns(vertex_positions=vertices.reshape(-1, 3), vertex_normals=normals.reshape(-1, 3)).partition(
                [len(ids)] * len(stamps)
            ),
        )


def main(config: Config) -> None:
    source = RrdReader(config.source)
    segment = config.segment
    application_id = config.application_id
    if config.reference is not None:
        identity = RrdReader(config.reference).recordings()[0]
        if identity.recording_id != segment:
            raise ValueError("reference recording must be the requested SelfCap segment")
        application_id = identity.application_id
    rows = []
    for chunk in source.stream().filter(content="/world/fit/joints").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            times = table.column("video_time").cast("int64").to_numpy().astype(np.float64) / 1e9
            points = np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32)
            rows.append((times, points))
    times = np.concatenate([row[0] for row in rows])
    points = np.concatenate([row[1] for row in rows])
    order = np.argsort(times)
    times, points = times[order], points[order]
    name_chunks = source.stream().filter(content="/parameters/joint_names").to_chunks()
    if not name_chunks:
        raise ValueError("fit recording lacks joint names; rerun soma-fit before exporting overlays")
    names = tuple(name_chunks[0].to_record_batch().column("TextDocument:text")[0].as_py()[0].splitlines())
    world_joints = points.copy()
    points = overlay_points(points, names, config.upper_body_only)
    ear_rows = []
    for chunk in source.stream().filter(content="/world/fit/ears").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            ear_rows.append(
                (table.column("video_time").cast("int64").to_numpy(), np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32))
            )
    if ear_rows:
        ear_times = np.concatenate([row[0] for row in ear_rows])
        ears = np.concatenate([row[1] for row in ear_rows])[np.argsort(ear_times)]
        if len(ears) != len(times):
            raise ValueError("ear and joint streams differ in length")
        points[:, 3:5] = ears
    foot_rows = []
    quest_rows = []
    for entity, destination in (("/world/fit/feet", foot_rows), ("/world/inputs/quest", quest_rows)):
        for chunk in source.stream().filter(content=entity).to_chunks():
            table = chunk.to_record_batch()
            if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
                destination.append(
                    (
                        table.column("video_time").cast("int64").to_numpy(),
                        np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32),
                    )
                )
    if foot_rows:
        feet = np.concatenate([row[1] for row in foot_rows])[np.argsort(np.concatenate([row[0] for row in foot_rows]))]
        if len(feet) != len(times):
            raise ValueError("foot and joint streams differ in length")
        points[:, 17:23] = feet
    quest = np.full_like(points, np.nan)
    if quest_rows:
        quest = np.concatenate([row[1] for row in quest_rows])[np.argsort(np.concatenate([row[0] for row in quest_rows]))]
        if len(quest) != len(times):
            raise ValueError("Quest and joint streams differ in length")
    # New fits persist the exact COCO-equivalent points consumed by their losses.
    # Older fits use the equivalent joint mapping plus separately saved surface ears.
    landmark_rows = []
    for chunk in source.stream().filter(content="/world/fit/coco_landmarks").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            landmark_rows.append(
                (table.column("video_time").cast("int64").to_numpy(), np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float32))
            )
    if landmark_rows:
        landmark_times = np.concatenate([row[0] for row in landmark_rows])
        landmark_order = np.argsort(landmark_times)
        if not np.array_equal(landmark_times[landmark_order], np.rint(times * 1e9).astype(np.int64)):
            raise ValueError("COCO landmarks and joints must share exact timestamps")
        points = np.concatenate([row[1] for row in landmark_rows])[landmark_order]
    if config.upper_body_only:
        points[:, 13:91] = np.nan
    dataset = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    phones, rgb = read_views(dataset, segment, times, config.exo_source)
    rgb_evidence = np.full((len(times), 133, 2), np.nan)
    egos = []
    if rgb is not None:
        for track in read_egohand_rgb(dataset, segment, times):
            if not len(track.times):
                continue
            indices = nearest_indices(track.times, times)
            valid = abs(track.times[indices] - times) <= 0.05
            base = 91 if track.name.endswith("/left") else 112
            rgb_evidence[valid, base : base + 21] = track.points[indices[valid]]
        paths = [f"/world/rig_{rgb.rig:02d}/cam_{cam:02d}" for cam in (1, 2)]
        table = dataset.filter_segments(segment).filter_contents([*paths, *(f"{path}/pinhole" for path in paths)]).reader(index=None).to_arrow_table()
        egos.append(rgb)
        for cam, path in enumerate(paths, start=1):
            missing = missing_camera_calibration(table, path, distorted=True)
            if missing:
                print(f"SKIP_CAMERA {segment} {path}: missing calibration {missing}", flush=True)
                continue
            transform = np.eye(4)
            transform[:3, :3] = first(table, f"{path}:Transform3D:mat3x3").reshape(3, 3).T
            transform[:3, 3] = first(table, f"{path}:Transform3D:translation")
            size = first(table, f"{path}/pinhole:Pinhole:resolution")
            egos.append(
                replace(
                    rgb,
                    cam=cam,
                    K=first(table, f"{path}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T,
                    distortion=first(table, f"{path}/pinhole:simplecv.components.DistortionCoefficients"),
                    size=(int(size[0]), int(size[1])),
                    rig_T_cam=transform,
                )
            )
    config.output.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(application_id, recording_id=segment)
    recording.save(config.output)
    if config.upper_body_only:
        static = source.stream().filter(content="/parameters/visible_joints").to_chunks()
        if not static:
            raise ValueError("upper-body output requires a fit with a rest-pose display selection")
        visible_ids = np.asarray(static[0].to_record_batch().column("Tensor:data")[0].as_py()[0]["buffer"], dtype=np.int64)
        hidden = np.ones(world_joints.shape[1], dtype=bool)
        hidden[visible_ids] = False
        world_joints[:, hidden] = np.nan
        for chunk in source.stream().filter(content="/world/fit/joints").to_chunks():
            if "video_time" not in chunk.to_record_batch().schema.names:
                recording.send_chunks(chunk.with_entity_path("/world/pred/soma_x/joints"))
        recording.send_columns(
            "/world/pred/soma_x/joints",
            indexes=[video_time_column(times)],
            columns=rr.Points3D.columns(positions=world_joints.reshape(-1, 3)).partition([world_joints.shape[1]] * len(times)),
        )
        log_cropped_mesh(recording, source, times)
    else:
        for old, new in (("/world/fit/joints", "/world/pred/soma_x/joints"), ("/world/fit/mesh", "/world/pred/soma_x/mesh")):
            recording.send_chunks(source.stream().filter(content=old).map(lambda chunk, new=new: chunk.with_entity_path(new)))
    recording.send_chunks(source.stream().filter(content="/world/fit/ears").map(lambda chunk: chunk.with_entity_path("/world/pred/soma_x/ears")))
    if config.native_exo:
        for path in ("/world/pred/soma_x/joints", "/world/pred/soma_x/ears"):
            recording.log(path, rr.Points3D.from_fields(radii=rr.Radius.ui_points(2.5)), static=True)
    exo_rows = []
    for chunk in source.stream().filter(content="/world/inputs/exo").to_chunks():
        table = chunk.to_record_batch()
        if "video_time" in table.schema.names and "Points3D:positions" in table.schema.names:
            exo_rows.append(
                (table.column("video_time").cast("int64").to_numpy(), np.asarray(table.column("Points3D:positions").to_pylist(), dtype=np.float64))
            )
    reference = np.full(points.shape, np.nan, dtype=np.float64)
    if exo_rows:
        exo_times = np.concatenate([row[0] for row in exo_rows])
        exo_order = np.argsort(exo_times)
        if not np.array_equal(exo_times[exo_order], np.rint(times * 1e9).astype(np.int64)):
            raise ValueError("source landmarks and fitted landmarks must share exact timestamps")
        reference = np.concatenate([row[1] for row in exo_rows])[exo_order]
    projections = []
    if not config.native_exo:
        for view in phones:
            projected = project(view.camera, points.reshape(-1, 3).astype(np.float64)).reshape(len(times), 133, 3)
            pixels = np.where(projected[:, :, 2:] > 0.001, projected[:, :, :2], np.nan).astype(np.float32)
            tolerance = residual_tolerance(
                points.astype(np.float64), reference, projected[:, :, 2], float(max(view.camera.K[0, 0], view.camera.K[1, 1]))
            )
            seen = observed_points(pixels.astype(np.float64), view.pixels, view.scores, tolerance)
            projections.append((f"{view.camera.entity}/pinhole", pixels, seen))
    for ego in egos:
        pixels = project_ego(ego, times, points)
        evidence = rgb_evidence if ego.cam == 0 else project_ego(ego, times, quest).astype(np.float64)
        # Mono images have no EgoHand detector: require RGB image support as
        # well as a nearby valid Quest projection, rather than claiming that
        # tracking alone establishes visibility through an occluder.
        if ego.cam != 0:
            evidence[~np.isfinite(rgb_evidence).all(-1)] = np.nan
        seen = observed_points(pixels.astype(np.float64), evidence, np.isfinite(evidence).all(-1).astype(np.float64))
        projections.append((f"/world/rig_{ego.rig:02d}/cam_{ego.cam:02d}/pinhole", pixels, seen))
    for pinhole, pixels, seen in projections:
        path = f"{pinhole}/pred/soma_x/joints_uv"
        log_joint_overlay(recording, path, times, pixels, seen)
        print(path, "observed:", int(seen.sum()), "finite joints:", int(np.isfinite(pixels).all(-1).sum()))
    recording.flush()
    recording.disconnect()
    print(config.output)
