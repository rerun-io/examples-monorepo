"""Write a layer aligned with SelfCap's original recording and camera tree, from a fit in memory."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dataforge.oak_mount import StaticCamera, read_static_cameras
from dataforge.schema import cam_path, pinhole_path
from dataforge.selfcap_schema import SOMA_EARS, SOMA_JOINTS, SOMA_MESH, SOMA_PARAMS
from dataforge.triangulation import EgoCamera, project, project_ego
from dataforge.writing import atomic_recording
from jaxtyping import Bool, Float32, Float64, Int32, Int64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetSelection
from rerun.chunk import RrdReader
from scipy.sparse import csr_matrix

from selfcap_soma.assets import SOMA_X_REPO_ID, SOMA_X_REVISION
from selfcap_soma.catalog import ExoSource, Observations, SegmentCatalog, open_segment
from selfcap_soma.evaluation_io import nearest_indices, read_egohand_rgb, read_views
from selfcap_soma.fitting import FitResult
from selfcap_soma.observations import require_full_rate_mesh, video_time_column
from selfcap_soma.output import joint_edges, log_joint_style, mapped
from selfcap_soma.overlay_style import observed_links, observed_points, residual_tolerance
from selfcap_soma.quality import SomaQuality, assess_quality, log_quality

SOMA_CROP: str = "/__soma/upper_body_vertices"
"""Static low-LOD vertex ids of the stored crop, component ``vertex_ids``: outside /world and without a visualizer, so no view shows it."""
NORMAL_FRAMES: int = 128
"""Frames per normals batch and per mesh chunk; bounds the float64 working arrays."""
TRIMESH_ZERO_NORM: float = 1e-13
"""trimesh.util.TOL_ZERO: a shorter vector has no direction."""
TRIMESH_DEGENERATE_ANGLE: float = 1e-8
"""trimesh.constants.tol.merge: a triangle with a smaller corner angle contributes nothing."""


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
    ears: Float32[ndarray, "n 2 3"]
    """Fitted ear landmarks."""
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
    upper_body_vertices: Int64[ndarray, "c"]
    """Rest-pose vertex selection for the upper-body surface."""
    visible_joints: Int64[ndarray, "k"]
    """Joint indices shown in upper-body displays."""
    poses: Float32[ndarray, "n 77 3"]
    """The fit's joint rotations (SOMA ``pose`` input), one row per fitted frame."""
    translations: Float32[ndarray, "n 3"]
    """The fit's world root translations in metres."""
    identity: Float32[ndarray, "1 128"]
    """The fit's native SOMA identity coefficients."""
    scale: float
    """The fit's global scale."""
    bone_scales: Float32[ndarray, "1 60"]
    """The fit's local bone-length ratios."""
    display_twists: bool
    """Whether the surface used procedural twist distribution."""


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
        result.poses,
        result.translations,
        result.identity,
        result.scale,
        result.bone_scales,
        result.display_twists,
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


POSE_ENCODING: str = "axis-angle, SOMA pose input with absolute_pose=False (relative to the rest joint orient), 77 joints, one row per joint"


def log_params(recording: rr.RecordingStream, fit: LayerFit) -> None:
    """Log the fit's parameters (decision 12: every new fit): identity, scales and settings once, rotations and translation
    per fitted frame. ``SOMALayer(identity_model_type="soma", lod="low", enable_procedural_transforms=display_twists)``,
    ``prepare_identity(identity, scale_params=bone_scales, global_scale=global_scale)`` and ``pose(rotations,
    transl=root_translation, apply_correctives=False)`` give the posed SOMA surface; the logged mesh adds only the
    waist-rim display offsets of ``fitting.mesh_selection`` (upper-body crop)."""
    recording.log(
        SOMA_PARAMS,
        rr.AnyValues(
            identity_model_type="soma",
            lod="low",
            assets_repo=SOMA_X_REPO_ID,
            assets_revision=SOMA_X_REVISION,
            rotation_encoding=POSE_ENCODING,
            joint_names=pa.array(fit.names),
            identity_coeffs=pa.FixedSizeListArray.from_arrays(pa.array(fit.identity.reshape(-1), type=pa.float32()), fit.identity.shape[1]),
            bone_scales=pa.FixedSizeListArray.from_arrays(pa.array(fit.bone_scales.reshape(-1), type=pa.float32()), fit.bone_scales.shape[1]),
            global_scale=fit.scale,
            apply_correctives=False,
            display_twists=fit.display_twists,
        ),
        static=True,
    )
    rotations: Float32[ndarray, "n c"] = fit.poses.reshape(len(fit.poses), -1)
    recording.send_columns(
        SOMA_PARAMS,
        indexes=[video_time_column(fit.times)],
        columns=rr.AnyValues.columns(
            rotations=pa.FixedSizeListArray.from_arrays(pa.array(rotations.reshape(-1), type=pa.float32()), rotations.shape[1]),
            root_translation=pa.FixedSizeListArray.from_arrays(pa.array(fit.translations.reshape(-1), type=pa.float32()), 3),
        ),
    )


def oak_stereo_cameras(view: DatasetSelection, segment: str, rgb: EgoCamera) -> list[EgoCamera]:
    """The OAK's left and right cameras beside its rgb one; a camera with incomplete calibration is reported and skipped."""
    paths: list[str] = [cam_path(rgb.rig, cam) for cam in (1, 2)]
    cameras: dict[str, StaticCamera] = read_static_cameras(view, view, paths)  # one dataset holds both layers here
    stereo: list[EgoCamera] = []
    for cam, path in enumerate(paths, start=1):
        found: StaticCamera = cameras[path]
        if found.rig_T_cam is None or found.K is None or found.resolution is None or found.distortion is None:
            print(f"SKIP_CAMERA {segment} {path}: missing calibration {found}", flush=True)
            continue
        stereo.append(replace(rgb, cam=cam, K=found.K, distortion=found.distortion, size=found.resolution, rig_T_cam=found.rig_T_cam))
    return stereo


def write_layer(segment: str, dataset: str, catalog_url: str, exo_source: ExoSource, output: Path, fit: LayerFit, *, catalog: SegmentCatalog | None = None) -> None:
    """Write one upper-body layer atomically; the caller registers it."""
    times: Float64[ndarray, "n"] = fit.times
    points: Float32[ndarray, "n 133 3"] = fit.points.copy()
    quest: Float32[ndarray, "n 133 3"] = fit.quest
    reference: Float64[ndarray, "n 133 3"] = fit.exo
    points[:, 13:91] = np.nan
    if catalog is None:
        catalog = open_segment(CatalogClient(catalog_url).get_dataset(dataset), segment)
    phones, rgb = read_views(catalog, segment, times, exo_source)
    rgb_evidence = np.full((len(times), 133, 2), np.nan)
    egos = []
    if rgb is not None:
        for track in read_egohand_rgb(catalog, times):
            if not len(track.times):
                continue
            indices = nearest_indices(track.times, times)
            valid = abs(track.times[indices] - times) <= 0.05
            base = 91 if track.name.endswith("/left") else 112
            rgb_evidence[valid, base : base + 21] = track.points[indices[valid]]
        egos += [rgb, *oak_stereo_cameras(catalog.view, segment, rgb)]
    with atomic_recording(output, recording_id=segment, send_properties=False) as recording:
        world_joints: Float32[ndarray, "n j 3"] = fit.joints.copy()
        hidden: Bool[ndarray, "j"] = np.ones(world_joints.shape[1], dtype=bool)
        hidden[fit.visible_joints] = False
        world_joints[:, hidden] = np.nan
        log_joint_style(recording, SOMA_JOINTS, fit.edges, world_joints.shape[1])
        recording.send_columns(
            SOMA_JOINTS,
            indexes=[video_time_column(times)],
            columns=rr.Points3D.columns(positions=world_joints.reshape(-1, 3)).partition([world_joints.shape[1]] * len(times)),
        )
        log_mesh(recording, fit, True)
        log_params(recording, fit)
        recording.log(SOMA_EARS, rr.Points3D.from_fields(colors=[0, 255, 255], radii=0.008), static=True)
        recording.send_columns(
            SOMA_EARS, indexes=[video_time_column(times)], columns=rr.Points3D.columns(positions=fit.ears.reshape(-1, 3)).partition([2] * len(times))
        )
        log_quality(recording, fit.quality)
        projections = []
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
            projections.append((pinhole_path(ego.rig, ego.cam), pixels, seen))
        for pinhole, pixels, seen in projections:
            path = f"{pinhole}/pred/soma_x/joints_uv"
            log_joint_overlay(recording, path, times, pixels, seen)
            print(path, "observed:", int(seen.sum()), "finite joints:", int(np.isfinite(pixels).all(-1).sum()))
    print(output)
