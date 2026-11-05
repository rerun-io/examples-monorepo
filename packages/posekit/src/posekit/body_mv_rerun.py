"""Rerun layout and logging of the multiview body demo.

Static: the blueprint (3D body view, prompt, one 2D tab per exo camera), the MHR-70 / segmentation / debug
annotation context, the exo pinholes and each exo video as an asset. Per frame: each camera's segmentation, boxes and
2D keypoints, then either the fused body (mesh, triangulated and MHR keypoints, reprojection metrics, per-view debug
points) or, with fewer than two views, the first view's mesh.
"""

import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Int, UInt8
from numpy import ndarray
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.rerun_log_utils import log_pinhole, log_video

from posekit.body_mv_fusion import FusedFrame, to_world
from posekit.body_mv_inference import CameraSegmentation
from posekit.body_mv_sequence import exo_video_sources
from posekit.body_rerun import BOX_PALETTE, SEG_CLASS_OFFSET, SEG_OVERLAY_ALPHA, compute_vertex_normals
from sam3d_body.metadata.mhr70 import MHR70_ID2NAME, MHR70_IDS, MHR70_LINKS


def make_blueprint(exo_cam_names: list[str], max_exo_views: Literal[4, 8] = 8) -> rrb.Blueprint:
    """
    Create a viewer layout with 3D body view and per-camera 2D tabs.

    Args:
        exo_cam_names: List of exocentric camera names.
        max_exo_views: Maximum number of camera views to show.

    Returns:
        Blueprint configured to show 3D view, text prompt, and camera tabs.
    """
    main_view: rrb.ContainerLike = rrb.Spatial3DView(
        origin="/",
        name="3D Body View",
        eye_controls=rrb.EyeControls3D(
            kind="Orbital", position=[0.0, 2.0, 2.0], look_target=[0, 0, 0.5], spin_speed=0.5
        ),
        line_grid=rrb.LineGrid3D(visible=False),
    )

    if exo_cam_names:
        exo_tabs_row: list[rrb.ContainerLike] = []
        for name in exo_cam_names[:max_exo_views]:
            origin_base: str = f"/world/exo/{name}/pinhole"
            exo_tabs_row.append(
                rrb.Tabs(
                    contents=[
                        rrb.Spatial2DView(
                            origin=origin_base,
                            name=f"exo/{name}",
                            contents=[
                                f"{origin_base}/video",
                                f"{origin_base}/boxes",
                                f"{origin_base}/keypoints_2d",
                                f"{origin_base}/segmentation_ids",
                            ],
                        )
                    ]
                )
            )

        prompt_view: rrb.ContainerLike = rrb.TextDocumentView()
        exo_view: rrb.ContainerLike = rrb.Horizontal(contents=exo_tabs_row)
        main_view = rrb.Horizontal(contents=[prompt_view, main_view], column_shares=[1, 10])
        main_view = rrb.Vertical(contents=[main_view, exo_view], row_shares=[4, 1])

    return rrb.Blueprint(
        rrb.Horizontal(contents=[main_view], column_shares=[4, 1]),
        collapse_panels=True,
    )


def log_annotation_context() -> None:
    """Register annotation context with MHR70 skeleton and segmentation colors."""
    # Person class for keypoints
    person_class = rr.ClassDescription(
        info=rr.AnnotationInfo(id=0, label="Person", color=(0, 0, 255)),
        keypoint_annotations=[rr.AnnotationInfo(id=idx, label=name) for idx, name in MHR70_ID2NAME.items()],
        keypoint_connections=MHR70_LINKS,
    )

    # Segmentation classes
    seg_classes: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET, label="Background", color=(64, 64, 64, 0))),
    ]
    num_colors: int = BOX_PALETTE.shape[0]
    for idx in range(1, num_colors + 1):
        rgb: UInt8[ndarray, "3"] = BOX_PALETTE[(idx - 1) % num_colors, :3]
        color_rgba: tuple[int, int, int, int] = (int(rgb[0]), int(rgb[1]), int(rgb[2]), SEG_OVERLAY_ALPHA)
        seg_classes.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=SEG_CLASS_OFFSET + idx, label=f"Person-{idx}", color=color_rgba))
        )

    # Debug keypoint classes for validation visualization
    debug_classes: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=1000, label="detected_kpts", color=(0, 255, 0))),  # Green
        rr.ClassDescription(info=rr.AnnotationInfo(id=1001, label="projected_triangulated", color=(255, 128, 0))),  # Orange
        rr.ClassDescription(info=rr.AnnotationInfo(id=1002, label="projected_mhr", color=(0, 128, 255))),  # Blue
    ]

    rr.log("/", rr.AnnotationContext([person_class, *seg_classes, *debug_classes]), static=True)


def log_cameras(exoego_sequence: BaseExoEgoSequence) -> None:
    """Log camera intrinsics/extrinsics following the schema."""
    if exoego_sequence.exo_sequence is not None:
        for cam in exoego_sequence.exo_sequence.exo_cam_list:
            if cam is None:
                continue
            cam_log_path: Path = Path("/world") / "exo" / cam.name
            log_pinhole(
                cam,
                cam_log_path=cam_log_path,
                image_plane_distance=exoego_sequence.exo_sequence.image_plane_distance,
                static=True,
            )


def log_exo_videos(sequence: BaseExoEgoSequence, parent_log_path: Path, timeline: str = "video_time") -> dict[str, Int[ndarray, "n_frames"]]:
    """Log each exo video once as an asset (much cheaper than per-frame images); returns each stream's frame timestamps."""
    # Pattern from simplecv/apis/view_exoego.py
    video_timestamps: dict[str, Int[ndarray, "n_frames"]] = {}
    for stream_name, video_source in exo_video_sources(sequence):
        video_log_path: Path = parent_log_path / "exo" / stream_name / "pinhole" / "video"
        # log_video (rr.AssetVideo / Mp4Reader) accepts only file paths —
        # spill RRD blobs to a temp file first.
        if isinstance(video_source, bytes):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as blob_file:
                blob_file.write(video_source)
            video_path: Path = Path(blob_file.name)
        else:
            if not video_source.exists():
                continue
            video_path = video_source
        timestamps_ns: Int[ndarray, "n_frames"] = log_video(video_path, video_log_path, timeline=timeline)
        if isinstance(video_source, bytes):
            video_path.unlink()
        video_timestamps[stream_name] = timestamps_ns
    return video_timestamps


def log_static_scene(sequence: BaseExoEgoSequence, prompt: str, exo_cam_names: list[str]) -> None:
    """Prompt text, blueprint, world coordinate system, annotation context and exo pinholes."""
    rr.log("text-prompt", rr.TextDocument(f"## Prompt\n\n{prompt}", media_type="text/markdown"), static=True)
    rr.send_blueprint(make_blueprint(exo_cam_names))
    rr.log("/", sequence.world_coordinate_system, static=True)
    log_annotation_context()
    log_cameras(sequence)


def log_camera_outputs(
    *,
    pinhole_path: str,
    frame_idx: int,
    segmentation: CameraSegmentation,
    timestamp_ns: int | None = None,
) -> None:
    """Log per-camera segmentation and boxes to Rerun (video is logged as static asset)."""
    # Sync with video_time timeline if timestamp available
    if timestamp_ns is not None:
        rr.set_time("video_time", duration=1e-9 * timestamp_ns)
    rr.set_time("frame_idx", sequence=frame_idx)
    # Note: Video is logged once as AssetVideo, no per-frame image logging needed
    rr.log(f"{pinhole_path}/segmentation_ids", rr.SegmentationImage(segmentation.seg_map))

    boxes: Float32[ndarray, "n 4"] | None = segmentation.boxes
    if boxes is None:
        return

    class_ids: Int[ndarray, "n"] = np.arange(1, boxes.shape[0] + 1, dtype=np.int32)
    labels: list[str] | None = None
    if segmentation.scores is not None:
        labels = [f"{float(score):.2f}" for score in segmentation.scores.tolist()]

    colors: UInt8[ndarray, "n 3"] = np.asarray(
        [BOX_PALETTE[idx % BOX_PALETTE.shape[0], :3] for idx in range(boxes.shape[0])],
        dtype=np.uint8,
    )
    rr.log(
        f"{pinhole_path}/boxes",
        rr.Boxes2D(
            array=boxes,
            array_format=rr.Box2DFormat.XYXY,
            class_ids=class_ids,
            labels=labels,
            colors=colors,
            show_labels=True,
        ),
    )


def log_keypoints_2d(pinhole_path: Path, frame_idx: int, kpts_2d: Float32[ndarray, "n_kpts 2"]) -> None:
    """One camera's MHR-70 2D keypoints."""
    rr.set_time("frame_idx", sequence=frame_idx)
    rr.log(
        f"{pinhole_path}/keypoints_2d",
        rr.Points2D(
            positions=kpts_2d,
            keypoint_ids=MHR70_IDS,
            class_ids=0,
            colors=(0, 255, 0),
        ),
    )


def _log_mesh(entity_path: str, verts_world: Float32[ndarray, "n_verts 3"], faces: Int[ndarray, "n_faces 3"], albedo: tuple[float, float, float, float]) -> None:
    faces_int: Int[ndarray, "n_faces 3"] = np.ascontiguousarray(faces, dtype=np.int32)
    vertex_normals: Float32[ndarray, "n_verts 3"] = compute_vertex_normals(verts_world, faces_int)
    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=verts_world,
            triangle_indices=faces_int,
            vertex_normals=vertex_normals,
            albedo_factor=albedo,
        ),
    )


def log_fused_frame(frame_idx: int, fused: FusedFrame, faces: Int[ndarray, "n_faces 3"], cam_names: list[str]) -> None:
    """The fused mesh and keypoints, the reprojection metrics, and each view's detected and projected keypoints."""
    rr.set_time("frame_idx", sequence=frame_idx)
    _log_mesh("/world/pred/fused/mesh", fused.verts_world, faces, (0.2, 0.6, 1.0, 0.8))  # Blue for fused mesh
    rr.log(
        "/world/pred/fused/kpts3d",
        rr.Points3D(
            positions=fused.triangulated_kpts,
            keypoint_ids=MHR70_IDS,
            class_ids=0,
            colors=(255, 128, 0),  # Orange for triangulated
            radii=0.015,
        ),
    )
    rr.log("/metrics/reprojection/triangulated/mean_error", rr.Scalars(fused.val_triangulated.mean_error_px))
    rr.log("/metrics/reprojection/triangulated/p90_error", rr.Scalars(fused.val_triangulated.p90_error_px))
    rr.log("/metrics/reprojection/mhr_perview/mean_error", rr.Scalars(fused.val_mhr.mean_error_px))
    rr.log("/metrics/reprojection/mhr_world/mean_error", rr.Scalars(fused.val_mhr_world.mean_error_px))
    rr.log("/metrics/3d_distance/mhr_vs_triangulated", rr.Scalars(fused.mean_3d_dist))

    # Debug visualization: MHR 3D keypoints from mhr_forward (cyan, to compare with triangulated orange)
    rr.log(
        "/world/pred/fused/mhr_kpts3d",
        rr.Points3D(
            positions=fused.mhr_world,
            keypoint_ids=MHR70_IDS,
            class_ids=0,
            colors=(0, 255, 255),  # Cyan for MHR FK
            radii=0.012,
        ),
    )
    for view_idx, cam_name in enumerate(cam_names):
        debug_path: str = f"/world/exo/{cam_name}/pinhole/debug"
        # Detected 2D keypoints (green, class_id=1000)
        rr.log(f"{debug_path}/detected", rr.Points2D(positions=fused.target_2d[view_idx], class_ids=1000, radii=4))
        # Projected triangulated (orange, class_id=1001)
        rr.log(f"{debug_path}/triangulated", rr.Points2D(positions=fused.projected_triangulated[view_idx], class_ids=1001, radii=3))
        # Projected MHR per-view (blue, class_id=1002)
        rr.log(f"{debug_path}/mhr_perview", rr.Points2D(positions=fused.projected_mhr[view_idx], class_ids=1002, radii=3))
        # Projected MHR world-optimized (cyan, class_id=1003)
        rr.log(f"{debug_path}/mhr_world", rr.Points2D(positions=fused.projected_mhr_world[view_idx], class_ids=1003, radii=3))


def log_fallback_mesh(
    frame_idx: int,
    verts_cam: Float32[ndarray, "n_verts 3"],
    cam_t: Float32[ndarray, "3"],
    world_T_cam: Float32[ndarray, "4 4"],
    faces: Int[ndarray, "n_faces 3"],
) -> None:
    """With fewer than two views: one view's mesh, in world coordinates."""
    rr.set_time("frame_idx", sequence=frame_idx)
    _log_mesh("/world/pred/fallback/mesh", to_world(verts_cam + cam_t, world_T_cam), faces, (0.8, 0.4, 0.2, 0.6))  # Orange for fallback
