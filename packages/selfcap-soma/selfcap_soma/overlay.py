"""Write a layer aligned with SelfCap's original recording and camera tree, from a fit in memory or a legacy fit.rrd."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dataforge.schema import SOMA_MESH
from dataforge.triangulation import project, project_ego
from jaxtyping import Bool, Float32, Float64, Int32, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient
from rerun.chunk import RrdReader
from scipy.sparse import csr_matrix

from selfcap_soma.catalog import ExoSource, Observations
from selfcap_soma.evaluation_io import first, missing_camera_calibration, nearest_indices, read_egohand_rgb, read_views
from selfcap_soma.fitting import FitResult
from selfcap_soma.observations import overlay_points, require_full_rate_mesh, video_time_column
from selfcap_soma.output import joint_edges, log_joint_style, mapped
from selfcap_soma.overlay_style import observed_links, observed_points, residual_tolerance
from selfcap_soma.quality import SomaQuality, assess_quality, log_quality, read_quality

SOMA_JOINTS: str = "/world/pred/soma_x/joints"
"""Fitted joints in the soma layer."""
SOMA_EARS: str = "/world/pred/soma_x/ears"
"""Fitted ear landmarks in the soma layer."""
SOMA_CROP: str = "/__soma/upper_body_vertices"
"""Static low-LOD vertex ids of the stored crop, component ``vertex_ids``: outside /world and without a visualizer, so no view shows it."""
NORMAL_FRAMES: int = 128
"""Frames per normals batch and per mesh chunk; bounds the float64 working arrays."""
TRIMESH_ZERO_NORM: float = 1e-13
"""trimesh.util.TOL_ZERO: a shorter vector has no direction."""
TRIMESH_DEGENERATE_ANGLE: float = 1e-8
"""trimesh.constants.tol.merge: a triangle with a smaller corner angle contributes nothing."""


@dataclass(frozen=True, slots=True)
class Config:
    upper_body_only: bool = True
    """Crop the mesh and display only head/ears, arms/hands and hips."""
    native_exo: bool = False
    """Use native exo/Quest projection: omit baked exo overlays; Quest has none; OAK stays baked."""
    exo_source: ExoSource = "rtmw"
    """Per-phone image evidence used to classify observed versus inferred joints."""
    source: Path = Path("/tmp/soma-fit/full-segment/fit.rrd")
    """Legacy fit recording read by soma-overlay; soma-batch passes its fit in memory and ignores this."""
    catalog_url: str = "rerun+http://dgx-spark:51235"
    """Catalog to read, without changing any source layer."""
    dataset: str = "selfcap-post-dev"
    """Dataset containing this segment's inputs and calibrated cameras."""
    segment: str = "selfcap__2025-12-03__d31bf1da__ep001"
    """Recording/segment ID used by the SelfCap converter."""
    output: Path = Path("/tmp/soma-fit/layer/selfcap__2025-12-03__d31bf1da__ep001.rrd")
    """Local layer file; registration is a separate explicit action."""


@dataclass(frozen=True, slots=True)
class LayerFit:
    """What a soma layer is made of, independent of where the fit came from."""

    times: Float64[ndarray, "n"]
    """Fitted frame times in seconds."""
    names: tuple[str, ...]
    """Joint names, matching joints."""
    joints: Float32[ndarray, "n j 3"]
    """Fitted world joints."""
    edges: list[tuple[int, int]]
    """Rig connections between joints."""
    ears: Float32[ndarray, "n 2 3"] | None
    """Fitted ear landmarks; older fits lack them."""
    points: Float32[ndarray, "n 133 3"]
    """COCO-133 equivalent points consumed by the fit's losses."""
    exo: Float64[ndarray, "n 133 3"]
    """Triangulated source landmarks; NaN when absent."""
    quest: Float32[ndarray, "n 133 3"]
    """Quest hand landmarks; NaN when absent."""
    quality: SomaQuality
    """Fit-residual screening record."""
    vertices: Float32[ndarray, "m v 3"]
    """Whole low-LOD surface per mesh frame."""
    mesh_stamps: Int64[ndarray, "m"]
    """Mesh frame times in nanoseconds."""
    faces: Int32[ndarray, "f 3"]
    """Whole low-LOD triangles."""
    upper_body_vertices: Int64[ndarray, "c"] | None
    """Rest-pose vertex selection for the upper-body surface."""
    visible_joints: Int64[ndarray, "k"] | None
    """Joint indices shown in upper-body displays."""


def layer_fit(data: Observations, result: FitResult) -> LayerFit:
    """Take a layer's contents directly from a finished fit."""
    points: Float64[ndarray, "n 133 3"] = mapped(result, data)
    return LayerFit(
        data.times,
        result.names,
        result.joints,
        joint_edges(result.parents),
        result.ear_landmarks,
        points.astype(np.float32),
        data.exo,
        data.hands.astype(np.float32),
        assess_quality(points, data.exo, data.hands, result.scale, data.confidence),
        result.vertices,
        np.rint(data.times * 1e9).astype(np.int64),
        result.faces,
        result.upper_body_vertices,
        result.visible_joints,
    )


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
    lengths = array.value_lengths().to_numpy(zero_copy_only=False)
    if len(lengths) and np.any(lengths != lengths[0]):
        raise ValueError("mesh vertex rows must have uniform vertex counts")
    vectors = array.flatten()
    if vectors.null_count:
        raise ValueError("mesh vertices must not be null")
    values = vectors.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    count = int(lengths[0]) if len(lengths) else 0
    return values.reshape(len(array), count, 3)


def read_layer_crop(source: RrdReader) -> Int64[ndarray, "v"]:
    """Low-LOD vertex ids of the upper-body crop that a soma layer stores, in stored order."""
    selection = source.stream().filter(content=SOMA_CROP).to_chunks()
    if not selection:
        raise ValueError(f"layer is missing its upper-body vertex selection {SOMA_CROP}; it predates this entity, so refit it")
    return selection[0].to_record_batch().column("vertex_ids").flatten().to_numpy(zero_copy_only=False).astype(np.int64)


def read_fit_ids(source: RrdReader, entity: str) -> Int64[ndarray, "v"] | None:
    """A static id tensor of a legacy fit recording."""
    selection = source.stream().filter(content=entity).to_chunks()
    if not selection:
        return None
    return np.asarray(selection[0].to_record_batch().column("Tensor:data")[0].as_py()[0]["buffer"], dtype=np.int64)


def read_fit_points(source: RrdReader, entity: str) -> tuple[Int64[ndarray, "n"], Float32[ndarray, "n p 3"]] | None:
    """Time-ordered point rows of one entity of a legacy fit recording."""
    stamps: list[Int64[ndarray, "m"]] = []
    rows: list[Float32[ndarray, "m p 3"]] = []
    for chunk in source.stream().filter(content=entity, is_static=False, components="Points3D:positions").to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        stamps.append(batch.column("video_time").cast("int64").to_numpy())
        rows.append(mesh_vertex_rows(batch.column("Points3D:positions")))
    if not rows:
        return None
    joined: Int64[ndarray, "n"] = np.concatenate(stamps)
    order: Int64[ndarray, "n"] = np.argsort(joined)
    return joined[order], np.concatenate(rows)[order]


def read_fit_mesh(source: RrdReader) -> tuple[Int64[ndarray, "m"], Float32[ndarray, "m v 3"], Int32[ndarray, "f 3"]]:
    """Time-ordered surface frames and triangles of a legacy fit recording."""
    faces: Int32[ndarray, "f 3"] | None = None
    stamps: list[Int64[ndarray, "k"]] = []
    rows: list[Float32[ndarray, "k v 3"]] = []
    for chunk in source.stream().filter(content="/world/fit/mesh").to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "Mesh3D:triangle_indices" in batch.schema.names:
            faces = np.asarray(batch.column("Mesh3D:triangle_indices")[0].as_py(), dtype=np.int32)
        if "video_time" in batch.schema.names and "Mesh3D:vertex_positions" in batch.schema.names:
            stamps.append(batch.column("video_time").cast("int64").to_numpy())
            rows.append(mesh_vertex_rows(batch.column("Mesh3D:vertex_positions")))
    if faces is None or not rows:
        raise ValueError("fit mesh has no faces or vertex frames")
    joined: Int64[ndarray, "m"] = np.concatenate(stamps)
    order: Int64[ndarray, "m"] = np.argsort(joined)
    return joined[order], np.concatenate(rows)[order], faces


def read_fit_edges(source: RrdReader) -> list[tuple[int, int]]:
    """Rig connections stored on a legacy fit recording's joint entity."""
    for chunk in source.stream().filter(content="/world/fit/joints", is_static=True).to_chunks():
        batch: pa.RecordBatch = chunk.to_record_batch()
        if "AnnotationContext:context" in batch.schema.names:
            connections = batch.column("AnnotationContext:context")[0].as_py()[0][0]["class_description"]["keypoint_connections"]
            return [(int(edge["keypoint0"]), int(edge["keypoint1"])) for edge in connections]
    return []


def read_fit(path: Path) -> LayerFit:
    """Read a layer's contents from a legacy fit.rrd, written by soma-fit and the research tools."""
    source: RrdReader = RrdReader(path)
    joint_rows = read_fit_points(source, "/world/fit/joints")
    if joint_rows is None:
        raise ValueError(f"{path} has no fitted joints")
    stamps: Int64[ndarray, "n"] = joint_rows[0]
    joints: Float32[ndarray, "n j 3"] = joint_rows[1]

    def aligned(entity: str) -> Float32[ndarray, "n p 3"] | None:
        read = read_fit_points(source, entity)
        if read is not None and not np.array_equal(read[0], stamps):
            raise ValueError(f"{entity} and the fitted joints must share exact timestamps")
        return None if read is None else read[1]

    name_chunks = source.stream().filter(content="/parameters/joint_names").to_chunks()
    if not name_chunks:
        raise ValueError("fit recording lacks joint names; rerun soma-fit before exporting overlays")
    names: tuple[str, ...] = tuple(name_chunks[0].to_record_batch().column("TextDocument:text")[0].as_py()[0].splitlines())
    ears: Float32[ndarray, "n 2 3"] | None = aligned("/world/fit/ears")
    feet: Float32[ndarray, "n 6 3"] | None = aligned("/world/fit/feet")
    # New fits persist the exact COCO-equivalent points consumed by their losses.
    # Older fits use the equivalent joint mapping plus separately saved surface ears and feet.
    points: Float32[ndarray, "n 133 3"] | None = aligned("/world/fit/coco_landmarks")
    if points is None:
        points = overlay_points(joints, names, upper_body_only=False)
        if ears is not None:
            points[:, 3:5] = ears
        if feet is not None:
            points[:, 17:23] = feet
    quest: Float32[ndarray, "n 133 3"] | None = aligned("/world/inputs/quest")
    if quest is None:
        quest = np.full_like(points, np.nan)
    exo: Float32[ndarray, "n 133 3"] | None = aligned("/world/inputs/exo")
    reference: Float64[ndarray, "n 133 3"] = np.full(points.shape, np.nan, dtype=np.float64) if exo is None else exo.astype(np.float64)
    try:
        quality: SomaQuality = read_quality(path)
    except ValueError:  # fits from before the quality record
        quality = assess_quality(points.astype(np.float64), reference, quest.astype(np.float64))
    mesh = read_fit_mesh(source)
    return LayerFit(
        stamps.astype(np.float64) / 1e9,
        names,
        joints,
        read_fit_edges(source),
        ears,
        points,
        reference,
        quest,
        quality,
        mesh[1],
        mesh[0],
        mesh[2],
        read_fit_ids(source, "/parameters/upper_body_vertices"),
        read_fit_ids(source, "/parameters/visible_joints"),
    )


def angle_weighted_normals(vertices: Float32[ndarray, "n v 3"], faces: Int32[ndarray, "f 3"]) -> Float32[ndarray, "n v 3"]:
    """Trimesh's vertex normals for every frame at once: unit face normals weighted by corner angle.

    A vertex in no valid face gets a zero normal. Float64 inside, as trimesh, because needle triangles lose their angles in float32.
    """
    corners: Float64[ndarray, "n f 3 3"] = vertices[:, faces].astype(np.float64)
    first_edge: Float64[ndarray, "n f 3"] = corners[:, :, 1] - corners[:, :, 0]
    last_edge: Float64[ndarray, "n f 3"] = corners[:, :, 2] - corners[:, :, 1]
    face_normals: Float64[ndarray, "n f 3"] = unit_vectors(np.cross(first_edge, last_edge))
    u: Float64[ndarray, "n f 3"] = unit_vectors(first_edge)
    v: Float64[ndarray, "n f 3"] = unit_vectors(corners[:, :, 2] - corners[:, :, 0])
    w: Float64[ndarray, "n f 3"] = unit_vectors(last_edge)
    angles: Float64[ndarray, "n f 3"] = np.empty(face_normals.shape, dtype=np.float64)
    angles[:, :, 0] = np.arccos(np.clip((u * v).sum(axis=-1), -1.0, 1.0))
    angles[:, :, 1] = np.arccos(np.clip((-u * w).sum(axis=-1), -1.0, 1.0))
    angles[:, :, 2] = np.pi - angles[:, :, 0] - angles[:, :, 1]
    angles[(angles < TRIMESH_DEGENERATE_ANGLE).any(axis=-1)] = 0.0
    # One row per vertex, one column per face corner: the product sums each vertex's corners for all frames at once.
    corners_of: csr_matrix = csr_matrix((np.ones(faces.size), (faces.ravel(), np.arange(faces.size))), shape=(vertices.shape[1], faces.size))
    weighted: Float64[ndarray, "n f 3 3"] = face_normals[:, :, None, :] * angles[:, :, :, None]
    summed: Float64[ndarray, "v m"] = corners_of @ weighted.transpose(1, 2, 0, 3).reshape(faces.size, -1)
    return unit_vectors(summed.reshape(vertices.shape[1], -1, 3).swapaxes(0, 1)).astype(np.float32)


def unit_vectors(vectors: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 3"]:
    """Unit vectors; a vector at trimesh's zero tolerance becomes zero."""
    norms: Float64[ndarray, "... 1"] = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > TRIMESH_ZERO_NORM)


def log_mesh(recording: rr.RecordingStream, fit: LayerFit, upper_body_only: bool) -> None:
    """Log the surface, with newly computed normals per frame; the upper-body crop is stored for later stages."""
    ids: Int64[ndarray, "c"] = np.arange(fit.vertices.shape[1], dtype=np.int64)
    if upper_body_only:
        if fit.upper_body_vertices is None:
            raise ValueError("fit is missing its rest-pose upper-body vertex selection")
        require_full_rate_mesh(fit.mesh_stamps, np.rint(fit.times * 1e9).astype(np.int64))
        ids = fit.upper_body_vertices
        recording.log(SOMA_CROP, rr.AnyValues(vertex_ids=ids), static=True)
    remap: Int64[ndarray, "v"] = np.full(fit.vertices.shape[1], -1, dtype=np.int64)
    remap[ids] = np.arange(len(ids))
    faces: Int32[ndarray, "e 3"] = remap[fit.faces[(remap[fit.faces] >= 0).all(1)]].astype(np.int32)
    recording.log(SOMA_MESH, rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[0.25, 0.65, 0.8, 0.5]), static=True)
    for start in range(0, len(fit.mesh_stamps), NORMAL_FRAMES):
        vertices: Float32[ndarray, "k c 3"] = fit.vertices[start : start + NORMAL_FRAMES][:, ids]
        recording.send_columns(
            SOMA_MESH,
            indexes=[rr.TimeColumn("video_time", duration=fit.mesh_stamps[start : start + NORMAL_FRAMES].astype("timedelta64[ns]"))],
            columns=rr.Mesh3D.columns(
                vertex_positions=vertices.reshape(-1, 3), vertex_normals=angle_weighted_normals(vertices, faces).reshape(-1, 3)
            ).partition([len(ids)] * len(vertices)),
        )


def main(config: Config) -> None:
    """Write a layer from a legacy fit recording."""
    write_layer(config, read_fit(config.source))


def write_layer(config: Config, fit: LayerFit) -> None:
    """Write one segment's layer to config.output; the caller registers it."""
    segment: str = config.segment
    times: Float64[ndarray, "n"] = fit.times
    points: Float32[ndarray, "n 133 3"] = fit.points.copy()
    quest: Float32[ndarray, "n 133 3"] = fit.quest
    reference: Float64[ndarray, "n 133 3"] = fit.exo
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
    recording = rr.RecordingStream("dataforge", recording_id=segment)
    recording.save(config.output)
    world_joints: Float32[ndarray, "n j 3"] = fit.joints.copy()
    if config.upper_body_only:
        if fit.visible_joints is None:
            raise ValueError("upper-body output requires a fit with a rest-pose display selection")
        hidden: Bool[ndarray, "j"] = np.ones(world_joints.shape[1], dtype=bool)
        hidden[fit.visible_joints] = False
        world_joints[:, hidden] = np.nan
    log_joint_style(recording, SOMA_JOINTS, fit.edges, world_joints.shape[1])
    recording.send_columns(
        SOMA_JOINTS,
        indexes=[video_time_column(times)],
        columns=rr.Points3D.columns(positions=world_joints.reshape(-1, 3)).partition([world_joints.shape[1]] * len(times)),
    )
    log_mesh(recording, fit, config.upper_body_only)
    if fit.ears is not None:
        recording.log(SOMA_EARS, rr.Points3D.from_fields(colors=[0, 255, 255], radii=0.008), static=True)
        recording.send_columns(
            SOMA_EARS, indexes=[video_time_column(times)], columns=rr.Points3D.columns(positions=fit.ears.reshape(-1, 3)).partition([2] * len(times))
        )
    if config.native_exo:
        for path in (SOMA_JOINTS, SOMA_EARS):
            recording.log(path, rr.Points3D.from_fields(radii=rr.Radius.ui_points(2.5)), static=True)
    log_quality(recording, fit.quality)
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
