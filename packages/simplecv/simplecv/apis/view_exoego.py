"""Rerun visualization helpers for combined exo- and ego-centric datasets."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Literal, NamedTuple, cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from einops import rearrange
from jaxtyping import Float, Float32, Int, UInt8, UInt16
from numpy import ndarray
from tqdm import tqdm

from simplecv.camera_parameters import Fisheye62Parameters, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence, ManoStack
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, EnvironmentMesh, ExoEgoLabels, ExoEgoSample
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.rerun_custom_types import (
    PinholeWithDistortion,
    Points2DWithConfidence,
    Points3DWithConfidence,
    confidence_scores_to_rgb,
)
from simplecv.rerun_log_utils import (
    RerunTyroConfig,
    log_pinhole,
    log_video,
)
from simplecv.sensors.camera.brown_conrady import (
    project_brown_conrady_diagonal,
    project_brown_conrady_grid,
)
from simplecv.sensors.camera.fisheye62 import project_kannala_brandt_diagonal

# Improve console readability when inspecting numeric debugging output.
np.set_printoptions(suppress=True)


@dataclass
class VisualizeConfig:
    """Structured configuration for running the exo/ego visualization CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""

    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing an annotated ``BaseExoEgoSequence``."""

    max_exo_videos_to_log: Literal[4, 8] = 8
    """Upper bound on the number of exo video panels rendered in the blueprint."""

    log_exo: bool = True
    """Enable logging of exo-camera imagery, intrinsics, and projections."""

    log_ego: bool = True
    """Enable logging of ego-camera imagery, intrinsics, and projections."""

    log_mano: bool = True
    """Enable streaming of MANO meshes and keypoints derived from the dataset."""

    log_mano_vertex_normals: bool = False
    """Compute and log dynamic MANO mesh vertex normals. Disabled by default for faster RRD ingest."""

    log_labels: bool = True
    """Control whether 2D/3D keypoint annotations are logged alongside videos."""

    log_env_mesh: bool = True
    """Relog the static environment mesh under ``/world/gt/env_mesh`` when available."""

    log_depths: bool = False
    """Enable logging of per-camera depth maps when available."""

    skip_camera_names: str = ""
    """Comma-separated camera names to exclude from 2D video panel tabs (e.g. 'quest3_right,rgb')."""


def set_annotation_context() -> None:
    """Register COCO-133 semantic metadata so subsequent logs show names/edges."""
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Coco Wholebody", color=(0, 0, 255)),
                    keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()],
                    keypoint_connections=COCO_133_LINKS,
                ),
            ]
        ),
        static=True,
    )


def log_environment_mesh(exoego_sequence: BaseExoEgoSequence, parent_log_path: Path) -> None:
    """Relog the static environment mesh from the dataset into the viewer."""
    environment_mesh: EnvironmentMesh | None = exoego_sequence.environment_mesh
    if environment_mesh is None:
        return

    vertex_positions: Float32[np.ndarray, "num_vertices 3"] = np.ascontiguousarray(
        environment_mesh.vertex_positions,
        dtype=np.float32,
    )
    triangle_indices: Int[np.ndarray, "num_faces 3"] = np.ascontiguousarray(
        environment_mesh.triangle_indices,
        dtype=np.uint32,
    )
    vertex_normals: Float32[np.ndarray, "num_vertices 3"] | None = (
        None if environment_mesh.vertex_normals is None else np.ascontiguousarray(environment_mesh.vertex_normals, dtype=np.float32)
    )
    vertex_colors: UInt8[np.ndarray, "num_vertices 4"] | None = (
        None if environment_mesh.vertex_colors is None else np.ascontiguousarray(environment_mesh.vertex_colors, dtype=np.uint8)
    )

    env_mesh_path: Path = parent_log_path / "gt" / "env_mesh"
    rr.log(
        str(env_mesh_path),
        rr.Mesh3D(
            vertex_positions=vertex_positions,
            triangle_indices=triangle_indices,
            vertex_normals=vertex_normals,
            vertex_colors=vertex_colors,
        ),
        static=True,
    )


def log_depths(
    exoego_sequence: BaseExoEgoSequence,
    parent_log_path: Path,
    timeline: str,
) -> None:
    """Log depth maps via column API using uint16 buffers (millimetres)."""

    for idx in tqdm(range(len(exoego_sequence))):
        sample: ExoEgoSample = exoego_sequence[idx]
        rr.set_time(timeline, duration=sample.canonical_timestamp_ns * 1e-9)
        exo_cam_params_list: list[PinholeParameters | Fisheye62Parameters | None] | None = sample.exo_cam_params_list
        exo_depth_list: list[UInt16[ndarray, "H W"]] | None = sample.exo_depth_list
        assert exo_cam_params_list is not None and exo_depth_list is not None
        for idx, exo_cam_param in enumerate(exo_cam_params_list):
            if exo_cam_param is None:
                continue
            cam_log_path: Path = parent_log_path / "exo" / exo_cam_param.name
            pinhole_log_path: Path = cam_log_path / "pinhole"
            depth_log_path: Path = pinhole_log_path / "depth"

            depth_uint16: UInt16[ndarray, "H W"] = exo_depth_list[idx]

            rr.log(f"{depth_log_path}", rr.DepthImage(depth_uint16, meter=1000))


def create_container(
    *,
    ego_video_log_paths: list[Path] | None = None,
    exo_video_log_paths: list[Path] | None = None,
    max_exo_videos_to_log: Literal[4, 8] = 8,
    skip_camera_names: frozenset[str] = frozenset(),
) -> rrb.ContainerLike:
    """Create a Rerun container for visualizing ego- and exo-centric streams.

    Args:
        ego_video_log_paths (list[Path] | None): Optional set of ego video entity
            roots; each path becomes a tabbed 2D view alongside the spatial view.
        exo_video_log_paths (list[Path] | None): Optional set of exo video entity
            roots; each path becomes a tabbed 2D view beneath the spatial view.
        max_exo_videos_to_log (Literal[4, 8]): Maximum number of exo video panels
            to materialize when ``exo_video_log_paths`` is provided.

    Returns:
        rrb.Blueprint: Assembled layout containing the configured views.
    """

    def _should_include(path: Path) -> bool:
        """Check if camera should be included based on skip list."""
        # Path structure: /world/{ego|exo}/{cam_name}/pinhole/video
        # path.parent = pinhole, path.parent.parent = cam_name
        cam_name: str = path.parent.parent.name
        return cam_name not in skip_camera_names

    # Build exclusion patterns for 3D view (exclude both ego and exo paths for skipped cameras)
    exclusion_patterns: list[str] = []
    for cam_name in skip_camera_names:
        exclusion_patterns.append(f"- /world/ego/{cam_name}/**")
        exclusion_patterns.append(f"- /world/exo/{cam_name}/**")

    # Include everything except excluded cameras
    contents_filter: str | list[str] = ["+ /**"] + exclusion_patterns if exclusion_patterns else "/**"

    main_view = rrb.Spatial3DView(
        origin="/",
        name="3D View",
        contents=contents_filter,
        spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
    )

    if ego_video_log_paths is not None:
        ego_video_log_paths = [p for p in ego_video_log_paths if _should_include(p)]
    if ego_video_log_paths:
        ego_view = rrb.Vertical(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in ego_video_log_paths
            ]
        )
        main_view = rrb.Horizontal(
            contents=[main_view, ego_view],
            column_shares=[4, 1],
        )

    if exo_video_log_paths is not None:
        exo_video_log_paths = [p for p in exo_video_log_paths if _should_include(p)]
    if exo_video_log_paths:
        exo_view = rrb.Horizontal(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}"),
                )
                for video_log_path in exo_video_log_paths[:max_exo_videos_to_log]
            ]
        )
        main_view = rrb.Vertical(
            contents=[main_view, exo_view],
            row_shares=[4, 1],
        )

    contents: rrb.ContainerLike = main_view

    return contents


def compute_vertex_normals_batch(
    verts: Float32[ndarray, "n_frames n_verts 3"],
    faces: Int[ndarray, "n_faces 3"],
    eps: float = 1e-12,
) -> Float32[ndarray, "n_frames n_verts 3"]:
    """Compute per-vertex normals for a batch of meshes sharing topology.

    Args:
        verts (Float32[np.ndarray, "n_frames n_verts 3"]): Batched vertex
            positions where each frame shares the same triangulation.
        faces (Int[np.ndarray, "n_faces 3"]): Triangle indices defining the
            mesh topology that ``verts`` adheres to.
        eps (float): Small epsilon that prevents division by zero when
            normalizing degenerate vertices.

    Returns:
        Float32[np.ndarray, "n_frames n_verts 3"]: Unit-length vertex normals
            for each frame, zeroed where the norm would be numerically unstable.
    """
    n_frames: int = int(verts.shape[0])
    n_verts: int = int(verts.shape[1])
    n_faces: int = int(faces.shape[0])

    faces_i: Int[ndarray, "n_faces 3"] = faces.astype(np.int64)
    v0: Float32[ndarray, "n_frames n_faces 3"] = verts[:, faces_i[:, 0], :]
    v1: Float32[ndarray, "n_frames n_faces 3"] = verts[:, faces_i[:, 1], :]
    v2: Float32[ndarray, "n_frames n_faces 3"] = verts[:, faces_i[:, 2], :]

    e1: Float32[ndarray, "n_frames n_faces 3"] = v1 - v0
    e2: Float32[ndarray, "n_frames n_faces 3"] = v2 - v0
    face_normals: Float32[ndarray, "n_frames n_faces 3"] = np.cross(e1, e2)

    vertex_normals: Float32[ndarray, "n_frames n_verts 3"] = np.zeros((n_frames, n_verts, 3), dtype=np.float32)
    for k in range(n_faces):
        i0: int = int(faces_i[k, 0])
        i1: int = int(faces_i[k, 1])
        i2: int = int(faces_i[k, 2])
        fn_k: Float32[ndarray, "n_frames 3"] = face_normals[:, k, :]
        vertex_normals[:, i0, :] = vertex_normals[:, i0, :] + fn_k
        vertex_normals[:, i1, :] = vertex_normals[:, i1, :] + fn_k
        vertex_normals[:, i2, :] = vertex_normals[:, i2, :] + fn_k

    norms: Float32[ndarray, "n_frames n_verts 1"] = np.linalg.norm(vertex_normals, axis=-1, keepdims=True).astype(np.float32)
    denom: Float32[ndarray, "n_frames n_verts 1"] = np.maximum(norms, np.float32(eps))
    vn_unit: Float32[ndarray, "n_frames n_verts 3"] = (vertex_normals / denom).astype(np.float32)
    mask: ndarray = norms > eps
    vn_unit = np.where(mask, vn_unit, np.float32(0.0))
    return vn_unit


def log_mano_batch(
    exoego_sequence: BaseExoEgoSequence,
    mano_parent_log_path: Path,
    timeline: str,
    timestamps_ns: Int[ndarray, "n_frames"],
    log_mano: bool,
    log_mano_vertex_normals: bool = False,
) -> None:
    """Stream MANO meshes and derived COCO joints to Rerun for both hands.

    Args:
        exoego_sequence (BaseExoEgoSequence): Sequence that may contain MANO
            pose parameters inside ``exoego_labels``.
        parent_log_path (Path): Root Rerun entity under which MANO data is
            organized.
        timeline (str): Logical timeline label shared across logged modalities.
        timestamps_ns (Int[np.ndarray, "n_frames"]): Nanosecond timestamps
            aligned with the MANO parameter stream; typically the label
            timeline from the recording.
        log_mano (bool): Gate controlling whether any MANO data is emitted.
        log_mano_vertex_normals (bool): Compute and emit per-frame MANO mesh
            vertex normals. This is disabled by default because it substantially
            increases EPFL Smart Kitchen RRD ingest time and output size.

    Returns:
        None: Data is emitted via ``rr.log`` and ``rr.send_columns`` side
            effects.
    """
    exoego_labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    if exoego_labels is None:
        return

    mano_mesh_color_rgba_map: dict[Literal["left", "right"], tuple[int, int, int, int]] = {
        "right": (255, 0, 0, 90),
        "left": (0, 0, 255, 90),
    }

    mano_stack: ManoStack | None = exoego_labels.mano_stack
    if mano_stack is not None and log_mano:
        from simplecv.ops.mano.mano_np import MANOLayerNP

        mano_root_path: Path = mano_parent_log_path / "mano"
        mano_layers = [
            # previous version only returned one shape for both hands. This is backwards compatible and works for 2 hands
            MANOLayerNP(side="right", betas=mano_stack.betas_for(0), use_pca=mano_stack.use_pca),
            MANOLayerNP(side="left", betas=mano_stack.betas_for(1), use_pca=mano_stack.use_pca),
        ]
        mano_so3: Float32[ndarray, "n_frames n_hands=2 48"] = mano_stack.so3
        mano_trans: Float32[ndarray, "n_frames n_hands=2 3"] = mano_stack.trans
        so3_per_hand: Float32[ndarray, "n_hands=2 n_frames 48"] = rearrange(mano_so3, "n_frames n_hands pose -> n_hands n_frames pose")
        trans_per_hand: Float32[ndarray, "n_hands=2 n_frames 3"] = rearrange(mano_trans, "n_frames n_hands dim -> n_hands n_frames dim")
        # Prepare a single COCO-133 buffer (both hands combined)
        n_frames_mano_total: int = min(so3_per_hand.shape[1], len(timestamps_ns))
        xyz_coco_mano: Float32[ndarray, "n_frames n_joints_coco=133 3"] = np.full((n_frames_mano_total, 133, 3), np.nan, dtype=np.float32)
        conf_coco_mano: Float32[ndarray, "n_frames n_joints_coco=133"] = np.zeros((n_frames_mano_total, 133), dtype=np.float32)
        for poses, translations, mano_layer in zip(so3_per_hand, trans_per_hand, mano_layers, strict=True):
            mano_outputs: tuple[
                Float32[ndarray, "n_frames n_verts=778 3"],
                Float32[ndarray, "n_frames n_joints=21 3"],
            ] = mano_layer(poses, translations)
            verts: Float32[ndarray, "n_frames n_verts=778 3"] = mano_outputs[0]
            xyz_mano: Float32[ndarray, "n_frames n_joints=21 3"] = mano_outputs[1]

            # Aggregate MANO joints (21) → into single COCO-133 buffer
            xyz_mano_np: Float32[ndarray, "n_frames n_joints=21 3"] = xyz_mano
            hand_idx: ndarray = RIGHT_HAND_IDX if mano_layer.side == "right" else LEFT_HAND_IDX
            xyz_coco_mano[:, hand_idx, :] = xyz_mano_np[0:n_frames_mano_total]
            conf_coco_mano[:, hand_idx] = 1.0

            hand_root: Path = mano_root_path / mano_layer.side
            mesh_entity_path: Path = hand_root / "mesh"
            rr.log(
                f"{mesh_entity_path}",
                rr.Mesh3D.from_fields(
                    triangle_indices=mano_layer.f.astype(np.int32),
                    albedo_factor=mano_mesh_color_rgba_map[mano_layer.side],
                ),
                static=True,
            )

            # Log MANO mesh: static faces from the MANO layer, dynamic per-frame vertices.
            verts_np: Float32[ndarray, "n_frames n_verts=778 3"] = verts
            n_frames_mesh: int = min(len(verts_np), len(timestamps_ns))
            vertex_positions_flat: Float32[ndarray, "n_total 3"] = rearrange(
                verts_np[0:n_frames_mesh],
                "n v d -> (n v) d",
            )
            mesh_columns: rr.ComponentColumnList
            if log_mano_vertex_normals:
                faces_np: Int[ndarray, "n_faces=1538 3"] = mano_layer.f.astype(np.int32)
                vertex_normals: Float32[ndarray, "n_frames n_verts=778 3"] = compute_vertex_normals_batch(
                    verts_np[0:n_frames_mesh],
                    faces_np,
                )
                vertex_normals_flat: Float32[ndarray, "n_total 3"] = rearrange(
                    vertex_normals,
                    "n v d -> (n v) d",
                )
                mesh_columns = rr.Mesh3D.columns(
                    vertex_positions=vertex_positions_flat,
                    vertex_normals=vertex_normals_flat,
                )
            else:
                mesh_columns = rr.Mesh3D.columns(vertex_positions=vertex_positions_flat)
            rr.send_columns(
                f"{mesh_entity_path}",
                indexes=[rr.TimeColumn(timeline, duration=1e-9 * timestamps_ns[0:n_frames_mesh])],
                columns=[*mesh_columns.partition(lengths=[verts_np.shape[1]] * n_frames_mesh)],
            )

        if n_frames_mano_total > 0:
            colors_coco: UInt8[ndarray, "n_frames 133 3"] = confidence_scores_to_rgb(confidence_scores=conf_coco_mano[..., np.newaxis])
            positions_flat: Float32[ndarray, "n_total 3"] = rearrange(
                xyz_coco_mano,
                "n_frames kpts dim -> (n_frames kpts) dim",
            ).astype(np.float32)
            colors_flat: UInt8[ndarray, "n_total 3"] = rearrange(
                colors_coco,
                "n_frames kpts dim -> (n_frames kpts) dim",
            )
            confidences_flat: Float32[ndarray, "n_total"] = rearrange(
                conf_coco_mano,
                "n_frames kpts -> (n_frames kpts)",
            ).astype(np.float32)
            n_keypoints: int = len(COCO_133_IDS)
            keypoint_lengths: Int[ndarray, "n_frames"] = np.full(n_frames_mano_total, n_keypoints, dtype=np.int32)

            rr.log(
                f"{mano_root_path}/coco133_xyz",
                Points3DWithConfidence.from_fields(
                    class_ids=0,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                ),
                static=True,
            )
            rr.send_columns(
                f"{mano_root_path}/coco133_xyz",
                indexes=[
                    rr.TimeColumn(
                        timeline,
                        duration=1e-9 * timestamps_ns[0:n_frames_mano_total],
                    )
                ],
                columns=[
                    *Points3DWithConfidence.columns(
                        positions=positions_flat,
                        colors=colors_flat,
                        confidences=confidences_flat,
                    ).partition(keypoint_lengths),
                ],
            )


def log_exoego_batch(
    exoego_sequence: BaseExoEgoSequence,
    parent_log_path: Path,
    timeline: str,
    shortest_timestamp: Int[ndarray, "n_frames"],
    log_ego: bool = True,
    log_exo: bool = True,
    log_mano: bool = False,
    log_mano_vertex_normals: bool = False,
) -> None:
    """Bulk-log 3D labels plus their ego/exo projections using columnar APIs.

    Args:
        exoego_sequence (BaseExoEgoSequence): Sequence containing videos,
            annotations, and camera parameters for ego and exo sensors.
        parent_log_path (Path): Root entity under which all logged data is
            organized.
        timeline (str): Logical timeline name shared between all logged tracks.
        shortest_timestamp (Int[np.ndarray, "n_frames"]): Fallback timestamps
            derived from the video streams; used when label timestamps are not
            present in the dataset.
        log_ego (bool): Enable logging of ego camera projections.
        log_exo (bool): Enable logging of exo camera projections.
        log_mano (bool): Enable logging of MANO-derived meshes and keypoints.
        log_mano_vertex_normals (bool): Compute and log dynamic MANO mesh
            vertex normals when MANO mesh logging is enabled.

    Returns:
        None: Data is emitted via ``rr.log`` and ``rr.send_columns`` side
            effects.
    """
    exoego_labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    if exoego_labels is None:
        return
    gt_root_path: Path = parent_log_path / "gt"

    ##########################
    # batch send all 3D data #
    ##########################
    xyzc_stack_all: Float[ndarray, "n_frames 133 4"] = exoego_labels.xyzc_stack
    label_timestamps_ns: Int[ndarray, "n_frames"] = exoego_labels.timestamps_ns if exoego_labels.timestamps_ns is not None else shortest_timestamp
    n_frames_labels: int = len(xyzc_stack_all)
    n_frames_timestamps: int = len(label_timestamps_ns)
    n_frames_total: int = min(n_frames_labels, n_frames_timestamps)

    xyzc_stack: Float[ndarray, "n_frames 133 4"] = xyzc_stack_all[0:n_frames_total]
    label_timestamps_trim: Int[ndarray, "n_frames"] = label_timestamps_ns[0:n_frames_total]
    xyz_stack: Float[ndarray, "n_frames 133 3"] = xyzc_stack[:, :, :3]
    conf_stack: Float[ndarray, "n_frames 133"] = xyzc_stack[:, :, 3]
    colors: UInt8[ndarray, "n_frames 133 3"] = confidence_scores_to_rgb(confidence_scores=conf_stack[..., np.newaxis])
    if n_frames_total > 0:
        positions_flat: Float[ndarray, "n_total 3"] = rearrange(
            xyz_stack,
            "n_frames kpts dim -> (n_frames kpts) dim",
        ).astype(np.float32)
        colors_flat: UInt8[ndarray, "n_total 3"] = rearrange(
            colors,
            "n_frames kpts dim -> (n_frames kpts) dim",
        )
        confidences_flat: Float32[ndarray, "n_total"] = rearrange(
            conf_stack,
            "n_frames kpts -> (n_frames kpts)",
        ).astype(np.float32)
        n_keypoints: int = len(COCO_133_IDS)
        keypoint_lengths: Int[ndarray, "n_frames"] = np.full(n_frames_total, n_keypoints, dtype=np.int32)

        rr.log(
            f"{gt_root_path}/coco133_xyz",
            Points3DWithConfidence.from_fields(
                class_ids=0,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
            ),
            static=True,
        )
        rr.send_columns(
            f"{gt_root_path}/coco133_xyz",
            indexes=[
                rr.TimeColumn(
                    timeline,
                    duration=1e-9 * label_timestamps_trim,
                )
            ],
            columns=[
                *Points3DWithConfidence.columns(
                    positions=positions_flat,
                    colors=colors_flat,
                    confidences=confidences_flat,
                ).partition(keypoint_lengths),
            ],
        )

        ############################
        # batch send all MANO data #
        ############################
        log_mano_batch(
            exoego_sequence=exoego_sequence,
            mano_parent_log_path=gt_root_path,
            timeline=timeline,
            timestamps_ns=label_timestamps_trim,
            log_mano=log_mano,
            log_mano_vertex_normals=log_mano_vertex_normals,
        )

    ###########################
    # batch send all exo cams #
    ###########################
    if exoego_sequence.exo_sequence is not None and log_exo:
        exo_cam_param_list: list[PinholeParameters] = [c for c in exoego_sequence.exo_sequence.exo_cam_list if c is not None]
        if not exo_cam_param_list:
            warnings.warn(
                "Skipping exo camera projections; no exo camera metadata available.",
                stacklevel=2,
            )
        else:
            if isinstance(exo_cam_param_list[0], PinholeParameters):
                uv_raw_stack: Float[ndarray, "n_frames n_views 133 2"] = project_brown_conrady_grid(
                    xyz_stack_world=xyz_stack, pinholes_per_view=exo_cam_param_list
                )
            else:
                raise NotImplementedError(f"Exo camera parameters of type '{type(exo_cam_param_list[0])}' are not supported.")
            for exo_cam_idx, exo_cam in enumerate(exo_cam_param_list):
                exo_cam_path: Path = parent_log_path / "exo" / exo_cam.name
                exo_pinhole_path: Path = exo_cam_path / "pinhole"
                uv_exo: Float[ndarray, "n_frames 133 2"] = uv_raw_stack[:, exo_cam_idx, :, :].copy()
                n_frames_cam: int = len(uv_exo)
                if n_frames_cam == 0:
                    continue
                positions_flat_2d: Float[ndarray, "n_total 2"] = rearrange(
                    uv_exo,
                    "n_frames kpts dim -> (n_frames kpts) dim",
                ).astype(np.float32)
                colors_cam: UInt8[ndarray, "n_frames kpts 3"] = colors[0:n_frames_cam].copy()
                conf_cam: Float[ndarray, "n_frames kpts"] = conf_stack[0:n_frames_cam].copy()
                colors_flat_2d: UInt8[ndarray, "n_total 3"] = rearrange(
                    colors_cam,
                    "n_frames kpts dim -> (n_frames kpts) dim",
                )
                confidences_flat_2d: Float32[ndarray, "n_total"] = rearrange(
                    conf_cam,
                    "n_frames kpts -> (n_frames kpts)",
                ).astype(np.float32)
                n_keypoints: int = len(COCO_133_IDS)
                keypoint_lengths_cam: Int[ndarray, "n_frames"] = np.full(n_frames_cam, n_keypoints, dtype=np.int32)

                rr.log(
                    f"{exo_pinhole_path}/coco133_uv",
                    Points2DWithConfidence.from_fields(
                        class_ids=0,
                        keypoint_ids=COCO_133_IDS,
                        show_labels=False,
                    ),
                    static=True,
                )
                rr.send_columns(
                    f"{exo_pinhole_path}/coco133_uv",
                    indexes=[
                        rr.TimeColumn(
                            timeline,
                            duration=1e-9 * label_timestamps_trim[0:n_frames_cam],
                        )
                    ],
                    columns=[
                        *Points2DWithConfidence.columns(
                            positions=positions_flat_2d,
                            colors=colors_flat_2d,
                            confidences=confidences_flat_2d,
                        ).partition(keypoint_lengths_cam),
                    ],
                )

    ###########################
    # batch send all ego cams #
    ###########################
    if exoego_sequence.ego_sequence is not None and log_ego:
        for cam_name, ego_cam_param_list in exoego_sequence.ego_sequence.ego_cam_dict.items():
            if not ego_cam_param_list:
                warnings.warn(
                    f"Skipping ego camera '{cam_name}' projections; no ego camera metadata available.",
                    stacklevel=2,
                )
                continue
            # We assume that all cameras have the intrinsics
            cam_log_path: Path = parent_log_path / "ego" / cam_name
            pinhole_log_path: Path = cam_log_path / "pinhole"

            # Align coordinate, confidence, and camera-parameter buffers when their lengths differ.
            n_frames_total: int = min(len(xyz_stack), len(ego_cam_param_list))
            xyz_trim: Float[ndarray, "n_frames 133 3"] = xyz_stack[:n_frames_total]
            conf_trim: Float[ndarray, "n_frames 133"] = conf_stack[:n_frames_total]
            color_trim: UInt8[ndarray, "n_frames 133 3"] = colors[:n_frames_total]

            if n_frames_total == 0:
                continue

            if isinstance(ego_cam_param_list[0], PinholeParameters):
                # Time-aligned fast path: one call over the full trimmed sequence
                pinhole_slice_full: list[PinholeParameters] = cast(list[PinholeParameters], ego_cam_param_list[:n_frames_total])
                uv_ego_stack: Float[ndarray, "n_frames 133 2"] = project_brown_conrady_diagonal(
                    xyz_stack_world=xyz_trim[:n_frames_total],
                    pinholes_per_frame=pinhole_slice_full,
                    filter_invalid=True,
                )

            elif isinstance(ego_cam_param_list[0], Fisheye62Parameters):
                # Time-aligned fisheye fast path: one pose per frame, no outer-product grid
                fisheye_slice_full: list[Fisheye62Parameters] = cast(list[Fisheye62Parameters], ego_cam_param_list[:n_frames_total])
                uv_ego_stack: Float[ndarray, "n_frames 133 2"] = project_kannala_brandt_diagonal(
                    xyz_stack_world=xyz_trim[:n_frames_total],
                    pinholes_per_frame=fisheye_slice_full,
                    filter_invalid=True,
                )
            else:
                raise NotImplementedError(f"Ego camera parameters of type '{type(ego_cam_param_list[0])}' are not supported.")

            n_frames_cam: int = len(uv_ego_stack)
            if n_frames_cam > 0:
                positions_flat_ego: Float[ndarray, "n_total 2"] = rearrange(
                    uv_ego_stack,
                    "n_frames kpts dim -> (n_frames kpts) dim",
                ).astype(np.float32)
                colors_ego: UInt8[ndarray, "n_frames kpts 3"] = color_trim[0:n_frames_cam].copy()
                conf_ego: Float[ndarray, "n_frames kpts"] = conf_trim[0:n_frames_cam].copy()
                colors_flat_ego: UInt8[ndarray, "n_total 3"] = rearrange(
                    colors_ego,
                    "n_frames kpts dim -> (n_frames kpts) dim",
                )
                confidences_flat_ego: Float32[ndarray, "n_total"] = rearrange(
                    conf_ego,
                    "n_frames kpts -> (n_frames kpts)",
                ).astype(np.float32)
                n_keypoints: int = len(COCO_133_IDS)
                keypoint_lengths_ego: Int[ndarray, "n_frames"] = np.full(n_frames_cam, n_keypoints, dtype=np.int32)

                # Same helper makes the ego path symmetrical with the exo cameras.
                rr.log(
                    f"{pinhole_log_path}/coco133_uv",
                    Points2DWithConfidence.from_fields(
                        class_ids=0,
                        keypoint_ids=COCO_133_IDS,
                        show_labels=False,
                    ),
                    static=True,
                )
                rr.send_columns(
                    f"{pinhole_log_path}/coco133_uv",
                    indexes=[
                        rr.TimeColumn(
                            timeline,
                            duration=1e-9 * label_timestamps_trim[0:n_frames_cam],
                        )
                    ],
                    columns=[
                        *Points2DWithConfidence.columns(
                            positions=positions_flat_ego,
                            colors=colors_flat_ego,
                            confidences=confidences_flat_ego,
                        ).partition(keypoint_lengths_ego),
                    ],
                )


class LogPaths(NamedTuple):
    """Collection of optional Rerun entity roots for video playback.

    Attributes:
        exo_video_log_paths (list[Path] | None): List of exo video entities to
            display as 2D panels, if available.
        ego_video_log_paths (list[Path] | None): List of ego video entities to
            display as 2D panels, if available.
    """

    exo_video_log_paths: list[Path] | None
    ego_video_log_paths: list[Path] | None


class SceneSetupResult(NamedTuple):
    """Combined result of scene bootstrapping and shared timing metadata.

    Attributes:
        log_paths (LogPaths): Optional entity roots for ego/exo videos.
        shortest_timestamp (Int[np.ndarray, "n_frames"]): Aligned timestamps
            shared across all logged modalities.
    """

    log_paths: LogPaths
    shortest_timestamp: Int[ndarray, "n_frames"]


def _choose_shortest_timeline_by_duration(
    timelines: list[Int[ndarray, "n_frames"]],
) -> Int[ndarray, "n_frames"]:
    """Select the timeline with the smallest actual duration, not frame count."""

    assert timelines, "No timelines provided to select from."
    durations_ns: list[int] = [int(ts[-1] - ts[0]) for ts in timelines]
    min_idx: int = int(np.argmin(durations_ns))
    return timelines[min_idx]


def _video_stream_timestamps_for_logging(
    exoego_sequence: BaseExoEgoSequence,
    *,
    log_ego: bool,
    log_exo: bool,
) -> list[Int[ndarray, "n_frames"]]:
    """Return only ego/exo video stream timestamps, excluding label-only streams."""
    timestamp_list: list[Int[ndarray, "n_frames"]] = []
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    if log_ego and ego_sequence is not None:
        for stream_name in ego_sequence.ego_video_names:
            stream_timestamps: Int[ndarray, "n_frames"] | None = exoego_sequence.stream_timestamps_ns.get(f"ego/{stream_name}")
            if stream_timestamps is not None:
                timestamp_list.append(stream_timestamps)

    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence
    if log_exo and exo_sequence is not None:
        for stream_name in exo_sequence.exo_video_names:
            stream_timestamps = exoego_sequence.stream_timestamps_ns.get(f"exo/{stream_name}")
            if stream_timestamps is not None:
                timestamp_list.append(stream_timestamps)

    return timestamp_list


def setup_scene(
    exoego_sequence: BaseExoEgoSequence,
    *,
    parent_log_path: Path,
    timeline: str,
    log_ego: bool,
    log_exo: bool,
    recording: rr.RecordingStream | None = None,
) -> SceneSetupResult:
    """Log static assets, videos, and transforms; derive the shared timeline.

    Args:
        exoego_sequence (BaseExoEgoSequence): Combined ego/exo dataset ready for
            visualization.
        parent_log_path (Path): Root entity under which all scene elements are
            recorded.
        timeline (str): Logical timeline label for video frame timestamps.

    Returns:
        SceneSetupResult: Bundle with optional video log paths and the
            synchronized timestamp vector ``Int[np.ndarray, "n_frames"]``.
    """
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    exo_sequence: BaseExoSequence | None = exoego_sequence.exo_sequence

    exo_timestamp_list: list[Int[ndarray, "n_frames"]] = []
    exo_video_log_paths: list[Path] | None = None
    if exo_sequence is not None and log_exo:
        exo_video_names: list[str] = exo_sequence.exo_video_names
        # Get video blobs if available (RRD sequences), otherwise use paths
        exo_video_blobs: dict[str, bytes] | None = getattr(exo_sequence, "_video_blobs", None)
        exo_video_files: list[Path] = exo_sequence.exo_video_paths
        assert len(exo_video_files) == len(exo_video_names), (
            f"Mismatched exo video assets ({len(exo_video_files)}) and names ({len(exo_video_names)})."
        )
        # Build name→cam dict using video names as keys (handles None cam params for uncalibrated cameras)
        exo_cam_by_name: dict[str, PinholeParameters | None] = dict(zip(exo_sequence.exo_video_names, exo_sequence.exo_cam_list, strict=True))
        exo_video_log_path_list: list[Path] = []
        logged_exo_cameras: set[str] = set()
        for stream_name, video_file in zip(exo_video_names, exo_video_files, strict=True):
            cam_log_path: Path = parent_log_path / "exo" / stream_name
            exo_cam: PinholeParameters | None = exo_cam_by_name.get(stream_name)
            if exo_cam is not None and stream_name not in logged_exo_cameras:
                log_pinhole(
                    camera=exo_cam,
                    cam_log_path=cam_log_path,
                    image_plane_distance=exo_sequence.image_plane_distance,
                    static=True,
                    recording=recording,
                    include_distortion=True,
                )
                logged_exo_cameras.add(stream_name)

            video_log_path: Path = cam_log_path / "pinhole" / "video"
            exo_video_log_path_list.append(video_log_path)
            # Use blob if available, otherwise use file path
            video_source: bytes | Path = exo_video_blobs[stream_name] if exo_video_blobs and stream_name in exo_video_blobs else video_file
            if isinstance(video_source, Path):
                assert video_source.suffix == ".mp4", f"Video file {video_source} is not an mp4."
            # Log video asset which is referred to by frame references.
            exo_timestamps_ns: Int[ndarray, "n_frames"] = log_video(video_source, video_log_path, timeline=timeline, recording=recording)
            exo_timestamp_list.append(exo_timestamps_ns)
        exo_video_log_paths = exo_video_log_path_list

    ego_timestamp_list: list[Int[ndarray, "n_frames"]] = []
    ego_video_log_paths: list[Path] | None = None
    if ego_sequence is not None and log_ego:
        ego_video_names: list[str] = ego_sequence.ego_video_names
        # Get video blobs if available (RRD sequences), otherwise use paths
        ego_video_blobs: dict[str, bytes] | None = getattr(ego_sequence, "_video_blobs", None)
        ego_video_files: list[Path] = ego_sequence.ego_video_paths
        assert len(ego_video_files) == len(ego_video_names), (
            f"Mismatched ego video assets ({len(ego_video_files)}) and names ({len(ego_video_names)})."
        )
        ego_video_log_path_list: list[Path] = []

        def _log_ego_cameras(shortest_ego_timestamp: Int[ndarray, "n_frames"]) -> None:
            ego_cam_dict: dict[str, list[PinholeParameters | Fisheye62Parameters]] = cast(
                dict[str, list[PinholeParameters | Fisheye62Parameters]], ego_sequence.ego_cam_dict
            )
            for cam_name, ego_cam_param_list in ego_cam_dict.items():
                if not ego_cam_param_list:
                    continue
                n_frames_cam: int = min(len(ego_cam_param_list), len(shortest_ego_timestamp))
                if n_frames_cam <= 0:
                    continue
                trimmed_cam_params: list[PinholeParameters | Fisheye62Parameters] = ego_cam_param_list[:n_frames_cam]
                # We assume that all cameras share intrinsics across frames
                first_cam: PinholeParameters | Fisheye62Parameters = trimmed_cam_params[0]
                cam_log_path: Path = parent_log_path / "ego" / str(cam_name)
                pinhole_log_path: Path = cam_log_path / "pinhole"
                rr.log(
                    f"{pinhole_log_path}",
                    PinholeWithDistortion.from_camera(
                        first_cam,
                        image_plane_distance=ego_sequence.image_plane_distance,
                        include_distortion=True,
                    ),
                    static=True,
                    recording=recording,
                )
                batch_world_t_cam: Float[ndarray, "n_frames 3"] = np.array(
                    [ego_cam_param.extrinsics.world_t_cam for ego_cam_param in trimmed_cam_params]
                )
                batch_world_R_cam: Float[ndarray, "n_frames 3 3"] = np.array(
                    [ego_cam_param.extrinsics.world_R_cam for ego_cam_param in trimmed_cam_params]
                )
                # camera extrinsics, there's no from_parent=True so need to send as world_x_cam
                rr.send_columns(
                    f"{cam_log_path}",
                    indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_ego_timestamp[0 : len(batch_world_t_cam)])],
                    columns=[
                        *rr.Transform3D.columns(
                            translation=rearrange(batch_world_t_cam, "f d -> (f) d"),
                            mat3x3=rearrange(batch_world_R_cam, "f r c -> (f) r c"),
                        ),
                    ],
                    recording=recording,
                )

        for stream_name, video_file in zip(ego_video_names, ego_video_files, strict=True):
            cam_log_path: Path = parent_log_path / "ego" / stream_name
            ego_video_log_path: Path = cam_log_path / "pinhole" / "video"
            ego_video_log_path_list.append(ego_video_log_path)
            # Use blob if available, otherwise use file path
            video_source: bytes | Path = ego_video_blobs[stream_name] if ego_video_blobs and stream_name in ego_video_blobs else video_file
            if isinstance(video_source, Path):
                assert video_source.suffix == ".mp4", f"Video file {video_source} is not an mp4."
            # Log video asset which is referred to by frame references.
            ego_timestamps_ns: Int[ndarray, "n_frames"] = log_video(video_source, ego_video_log_path, timeline=timeline, recording=recording)
            ego_timestamp_list.append(ego_timestamps_ns)
        ego_video_log_paths = ego_video_log_path_list

        # Camera trajectories are logged after video ingestion so their time
        # range is clipped to the frames the demuxer actually accepted.
        if ego_timestamp_list:
            _log_ego_cameras(_choose_shortest_timeline_by_duration(ego_timestamp_list))

    shortest_timestamp: Int[ndarray, "n_frames"] = _choose_shortest_timeline_by_duration(exo_timestamp_list + ego_timestamp_list)

    return SceneSetupResult(
        log_paths=LogPaths(exo_video_log_paths=exo_video_log_paths, ego_video_log_paths=ego_video_log_paths),
        shortest_timestamp=shortest_timestamp,
    )


def visualize_exo_ego(exoego_sequence: BaseExoEgoSequence, config: VisualizeConfig):
    """Entry-point used by ``tools/view_exoego.py`` to drive the visualization.

    Args:
        exoego_sequence (BaseExoEgoSequence): The exo-ego sequence to visualize.
        config (VisualizeConfig): Run configuration describing dataset,
            logging toggles, and viewer options.

    Returns:
        None: Side-effectful logging call sequence that feeds the Rerun viewer.
    """
    start_time: float = timer()

    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    set_annotation_context()

    parent_log_path = Path("world")
    timeline: str = "video_time"

    if config.log_labels:
        video_timestamp_list: list[Int[ndarray, "n_frames"]] = _video_stream_timestamps_for_logging(
            exoego_sequence,
            log_ego=config.log_ego,
            log_exo=config.log_exo,
        )
        label_fallback_timestamp: Int[ndarray, "n_frames"] = _choose_shortest_timeline_by_duration(video_timestamp_list)
        log_exoego_batch(
            exoego_sequence,
            parent_log_path=parent_log_path,
            timeline=timeline,
            shortest_timestamp=label_fallback_timestamp,
            log_ego=config.log_ego,
            log_exo=config.log_exo,
            log_mano=config.log_mano,
            log_mano_vertex_normals=config.log_mano_vertex_normals,
        )

    scene_setup_result: SceneSetupResult = setup_scene(
        exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        log_ego=config.log_ego,
        log_exo=config.log_exo,
    )
    log_paths: LogPaths = scene_setup_result.log_paths

    if config.log_env_mesh:
        log_environment_mesh(exoego_sequence, parent_log_path)

    if config.log_depths:
        log_depths(
            exoego_sequence=exoego_sequence,
            parent_log_path=parent_log_path,
            timeline=timeline,
        )

    skip_set: frozenset[str] = (
        frozenset(name.strip() for name in config.skip_camera_names.split(",") if name.strip()) if config.skip_camera_names else frozenset()
    )
    container: rrb.ContainerLike = create_container(
        exo_video_log_paths=log_paths.exo_video_log_paths,
        ego_video_log_paths=log_paths.ego_video_log_paths,
        skip_camera_names=skip_set,
    )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=[container],
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    print(f"Total time taken: {timer() - start_time:.2f} seconds")


def main(config: VisualizeConfig) -> None:
    """
    Entry-point used by ``tools/view_exoego.py`` to drive the visualization.
    Seperated out so that we can use visualize_exo_ego in other contexts.

    Args:
        config (VisualizeConfig): Run configuration describing dataset,
            logging toggles, and viewer options.
    """
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()
    visualize_exo_ego(exoego_sequence, config)


def entrypoint() -> None:
    """CLI entrypoint for viewing ExoEgo data with Rerun."""
    import tyro

    tyro.extras.set_accent_color("bright_cyan")
    config: VisualizeConfig = tyro.cli(
        VisualizeConfig,
        description="Visualize ExoEgo dataset sequences with Rerun.",
    )
    main(config=config)


if __name__ == "__main__":
    entrypoint()
