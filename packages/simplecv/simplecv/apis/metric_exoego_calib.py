from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.apis.view_exoego import LogPaths, SceneSetupResult, _choose_shortest_timeline_by_duration, log_exoego_batch
from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Fisheye62Parameters, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence, ExoEgoLabels
from simplecv.data.skeleton.coco133_layers import (
    COCO133_LAYER_COLORS,
    COCO133_LAYER_LABELS,
    COCO133_ROI_COLORS,
    COCO133_ROI_LABELS,
    Coco133AnnotationLayer,
    Coco133RoiLayer,
)
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.ops.triangulate import batch_triangulate
from simplecv.rerun_custom_types import PinholeWithDistortion, Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole, log_video
from simplecv.rig import rebuild_camera_with_extrinsics
from simplecv.sensors.camera.brown_conrady import (
    project_brown_conrady_diagonal,
    project_brown_conrady_grid,
    undistort_brown_conrady_batch,
)
from simplecv.video_io import MultiVideoReader

np.set_printoptions(suppress=True)

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
device = "cuda" if torch.cuda.is_available() else "cpu"


CameraParam = PinholeParameters | Fisheye62Parameters


@dataclass(slots=True)
class OakQuestAlignmentConfig:
    """Camera labels used to anchor the Oak rig to the quest rig."""

    quest_reference_cam: str = "quest3_left"
    oak_reference_cam: str = "left"


def _clone_with_extrinsics(camera: CameraParam, world_T_cam: Float[ndarray, "4 4"]) -> CameraParam:
    """Return a fresh camera parameter object sharing intrinsics but new extrinsics."""
    new_extrinsics = Extrinsics(world_R_cam=world_T_cam[:3, :3], world_t_cam=world_T_cam[:3, 3])
    return cast(CameraParam, rebuild_camera_with_extrinsics(camera, new_extrinsics))


@dataclass(slots=True)
class UmeyamaResult:
    """Similarity transform mapping Oak (source) points to the target/new Oak world (destination)."""

    dst_T_src: Float32[ndarray, "4 4"]
    n_matches: int
    rms_error_m: float


def create_view_container(
    *,
    ego_video_log_paths: list[Path] | None = None,
    exo_video_log_paths: list[Path] | None = None,
    max_exo_videos_to_log: int = 8,
) -> rrb.ContainerLike:
    """Create a Rerun blueprint for visualizing ego- and exo-centric streams.

    Args:
        ego_video_log_paths (list[Path] | None): Optional set of ego video entity
            roots; each path becomes a tabbed 2D view alongside the spatial view.
        exo_video_log_paths (list[Path] | None): Optional set of exo video entity
            roots; each path becomes a tabbed 2D view beneath the spatial view.
        max_exo_videos_to_log (Literal[4, 8]): Maximum number of exo video panels
            to materialize when ``exo_video_log_paths`` is provided.
        max_ego_videos_to_log (Literal[4, 8]): Maximum number of ego video panels
            to materialize when ``ego_video_log_paths`` is provided.

    Returns:
        rrb.Blueprint: Assembled layout containing the configured views.
    """
    contents_3d: list[str] = [
        "+ $origin/**",
        "- /world/gt/env_pointcloud",  # hide raw point clouds by default
    ]
    video_log_paths: list[Path] = []
    if ego_video_log_paths is not None:
        video_log_paths += ego_video_log_paths
    if exo_video_log_paths is not None:
        video_log_paths += exo_video_log_paths[:max_exo_videos_to_log]

    contents_3d += [f"- {video_log_path.parent}/image/**" for video_log_path in video_log_paths]
    main_view = rrb.Spatial3DView(
        origin="/",
        contents=contents_3d,
        line_grid=rrb.archetypes.LineGrid3D(visible=True),
        spatial_information=rrb.SpatialInformation.from_fields(show_axes=True),
    )

    if ego_video_log_paths is not None:
        # TODO sort such that the quest cameras appear last in the 2d views
        ego_view = rrb.Vertical(
            contents=[
                rrb.Tabs(
                    rrb.Spatial2DView(
                        origin=f"{video_log_path.parent}", contents=["+ $origin/**", "- $origin/image/**"]
                    ),
                    rrb.Spatial2DView(origin=f"{video_log_path.parent}/image", contents=["+ $origin/**"]),
                    active_tab=0,
                )
                for video_log_path in ego_video_log_paths
            ]
        )
        main_view = rrb.Horizontal(
            contents=[main_view, ego_view],
            column_shares=[4, 1],
        )

    if exo_video_log_paths is not None:
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

    final_view = rrb.Horizontal(
        contents=[main_view],
        column_shares=[4, 1],
    )

    return final_view


def set_annotation_context(*, recording: rr.RecordingStream | None) -> None:
    keypoint_infos: list[rr.AnnotationInfo] = [
        rr.AnnotationInfo(id=id, label=name) for id, name in COCO_133_ID2NAME.items()
    ]
    class_descriptions: list[rr.ClassDescription] = []
    for layer in Coco133AnnotationLayer:
        label: str = COCO133_LAYER_LABELS[layer]
        color: tuple[int, int, int] = COCO133_LAYER_COLORS[layer]
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=int(layer), label=label, color=color),
                keypoint_annotations=keypoint_infos,
                keypoint_connections=COCO_133_LINKS,
            )
        )

    for roi_layer in Coco133RoiLayer:
        roi_label: str = COCO133_ROI_LABELS[roi_layer]
        roi_color: tuple[int, int, int] = COCO133_ROI_COLORS[roi_layer]
        class_descriptions.append(
            rr.ClassDescription(info=rr.AnnotationInfo(id=int(roi_layer), label=roi_label, color=roi_color))
        )
    rr.log(
        "/",
        rr.AnnotationContext(class_descriptions),
        static=True,
        recording=recording,
    )


def timestamp_to_frame_index(time_ns: int, frame_timestamps_ns: Int[ndarray, "num_frames"]) -> int:
    """Map a timestamp (ns) to the closest frame idx at-or-before that time.

    The mapping mirrors how VideoFrameReference columns are generated from
    AssetVideo.read_frame_timestamps_nanos().
    """

    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, len(frame_timestamps_ns) - 1))


def frame_index_to_timestamp(frame_timestamps_ns: Int[ndarray, "num_frames"], frame_index: int) -> int:
    """Return the nanosecond timestamp associated with a frame index."""
    if frame_index < 0 or frame_index >= int(frame_timestamps_ns.shape[0]):
        msg = f"frame_index {frame_index} is outside the valid range [0, {frame_timestamps_ns.shape[0] - 1}]"
        raise IndexError(msg)
    timestamp_ns: int = int(frame_timestamps_ns[frame_index])
    return timestamp_ns


def single_frame_mv_hands(
    ego_sequence: BaseEgoSequence,
    gt_xyzc_stack_all: Float[ndarray, "n_frames 133 4"],
    parent_log_path: Path,
    timeline: str,
    shortest_timestamp: Int[ndarray, "num_frames"],
    ts_nano: int,
    debug: bool = False,
) -> UmeyamaResult | None:
    """
    Take a single frame from the ego sequence and run hand detection and keypoint estimation
    """
    from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig
    from wilor_nano.hand_keypoints import (
        FinalWilorPred,
        HandKeypointDetectorConfig,
        KeypointResults,
        WilorHandKeypointDetector,
    )

    hand_detector = HandDetector(HandDetectorConfig(hf_wilor_repo_id="pablovela5620/wilor-nano", verbose=False))
    hand_keypoint_model = WilorHandKeypointDetector(
        HandKeypointDetectorConfig(hf_wilor_repo_id="pablovela5620/wilor-nano", verbose=False)
    )
    ego_video_readers: MultiVideoReader = ego_sequence.ego_video_readers
    rr.set_time(timeline=timeline, duration=1e-9 * ts_nano)
    ts_idx: int = timestamp_to_frame_index(time_ns=ts_nano, frame_timestamps_ns=shortest_timestamp)
    bgr_list: list[UInt8[ndarray, "H W 3"]] = ego_video_readers[ts_idx]
    rgb_list: list[UInt8[ndarray, "H W 3"]] = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in bgr_list]

    uvc_coco_list: list[Float32[ndarray, "133 3"]] = []
    uvc_cam_names: list[str] = []
    for ego_cam_name, rgb_hw3 in tqdm(
        list(zip(ego_sequence.ego_cam_dict.keys(), rgb_list, strict=True)),
        desc="Hand detect/keypoints (calib frame)",
        leave=False,
    ):
        # don't run on the quest cameras
        if "quest" in ego_cam_name.lower() or "rgb" in ego_cam_name.lower():
            continue
        det_results: DetectionResult = hand_detector(rgb_hw3, hand_conf=0.3)
        cam_log_path: Path = parent_log_path / "ego" / ego_cam_name
        pinhole_log_path: Path = cam_log_path / "pinhole"
        rr.log(f"{pinhole_log_path}/image", rr.Image(rgb_hw3).compress(jpeg_quality=90))
        uvc_coco: Float32[ndarray, "133 3"] = np.full((133, 3), np.nan, dtype=np.float32)
        # log detected hands and keypoints
        if det_results.left_xyxy is not None:
            wilor_preds: FinalWilorPred | KeypointResults = hand_keypoint_model(
                rgb_hw3=rgb_hw3, xyxy=det_results.left_xyxy, handedness="left"
            )
            assert isinstance(wilor_preds, KeypointResults)
            if debug:
                rr.log(
                    f"{pinhole_log_path}/image/{COCO133_ROI_LABELS[Coco133RoiLayer.LEFT_HAND]}",
                    rr.Boxes2D(
                        array=det_results.left_xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=int(Coco133RoiLayer.LEFT_HAND),
                        show_labels=False,
                    ),
                )

            uvc_coco[LEFT_HAND_IDX, :2] = wilor_preds.keypoints_2d[0]
            uvc_coco[LEFT_HAND_IDX, 2] = wilor_preds.scores[0]

        if det_results.right_xyxy is not None:
            wilor_preds: FinalWilorPred | KeypointResults = hand_keypoint_model(
                rgb_hw3=rgb_hw3, xyxy=det_results.right_xyxy, handedness="right"
            )
            assert isinstance(wilor_preds, KeypointResults)
            if debug:
                rr.log(
                    f"{pinhole_log_path}/image/{COCO133_ROI_LABELS[Coco133RoiLayer.RIGHT_HAND]}",
                    rr.Boxes2D(
                        array=det_results.right_xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=int(Coco133RoiLayer.RIGHT_HAND),
                        show_labels=False,
                    ),
                )

            uvc_coco[RIGHT_HAND_IDX, :2] = wilor_preds.keypoints_2d[0]
            uvc_coco[RIGHT_HAND_IDX, 2] = wilor_preds.scores[0]

        uvc_coco_list.append(uvc_coco)
        uvc_cam_names.append(ego_cam_name)
        confidence_rgb_view_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
            uvc_coco[:, 2][np.newaxis, :, np.newaxis]
        )
        confidence_rgb_view: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_view_stack[0]
        # log all keypoints
        if debug:
            rr.log(
                f"{pinhole_log_path}/image/left_hand/keypoints_2d",
                Points2DWithConfidence(
                    positions=uvc_coco[:, :2],
                    confidences=uvc_coco[:, 2],
                    class_ids=int(Coco133AnnotationLayer.RAW_2D),
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=confidence_rgb_view,
                ),
            )
    # triangulate 3d keypoints from all ego views (undistort before solving)
    uvc_coco_stack: Float[ndarray, "n_views n_kpts=133 3"] = np.stack(uvc_coco_list, axis=0)
    pinhole_param_list: list[PinholeParameters] = []
    for ego_cam_name in uvc_cam_names:
        pinhole_list: list[Fisheye62Parameters | PinholeParameters] = ego_sequence.ego_cam_dict[ego_cam_name]
        chosen_pinhole: Fisheye62Parameters | PinholeParameters = pinhole_list[ts_idx]
        assert isinstance(chosen_pinhole, PinholeParameters), f"Expected PinholeParameters, got {type(chosen_pinhole)}"
        pinhole_param_list.append(chosen_pinhole)

    k_matrices: list[Float[ndarray, "3 3"]] = []
    distortions: list[BrownConradyDistortion | None] = []
    for cam in pinhole_param_list:
        k_matrix: Float[ndarray, "3 3"] | None = cam.intrinsics.k_matrix
        assert k_matrix is not None, "All pinhole cameras must include intrinsics.k_matrix for triangulation"
        k_matrices.append(k_matrix)
        distortions.append(cam.distortion)
    K_stack: Float[ndarray, "n_views 3 3"] = np.stack(k_matrices, axis=0).astype(np.float64)

    if any(distortion is not None for distortion in distortions):
        uv_batched: Float[ndarray, "1 n_views n_kpts 2"] = uvc_coco_stack[None, :, :, :2]
        uv_undistorted_batched: Float[ndarray, "1 n_views n_kpts 2"] = undistort_brown_conrady_batch(
            uv_distorted=uv_batched,
            intrinsics_stack=K_stack,
            distortions=distortions,
        )
        uvc_coco_stack[:, :, :2] = uv_undistorted_batched[0].astype(np.float32, copy=False)
    Pall: Float32[ndarray, "n_views 3 4"] = np.stack(
        [cam.projection_matrix for cam in pinhole_param_list],
        axis=0,
    ).astype(np.float32)
    xyzc_coco: Float[ndarray, "n_kpts=133 4"] = batch_triangulate(
        keypoints_2d=uvc_coco_stack,
        projection_matrices=Pall,
    )
    tri_conf_rgb_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
        xyzc_coco[:, 3][np.newaxis, :, np.newaxis]
    )
    tri_conf_rgb: UInt8[ndarray, "n_kpts 3"] = tri_conf_rgb_stack[0]

    positions_for_log: Float32[ndarray, "n_kpts 3"] = xyzc_coco[:, :3].astype(np.float32, copy=True)
    confidences_for_log: Float32[ndarray, "n_kpts"] = xyzc_coco[:, 3].astype(np.float32, copy=True)
    # Points with zero/invalid confidence correspond to missing detections; log them as NaN
    # so Rerun does not visualize them at the origin.
    invalid_mask: Bool[ndarray, "n_kpts"] = (~np.isfinite(confidences_for_log)) | (confidences_for_log <= 0.0)
    positions_for_log[invalid_mask, :] = np.nan

    if debug:
        rr.log(
            f"{parent_log_path}/triangulated_hands/coco133_xyz",
            Points3DWithConfidence(
                positions=positions_for_log,
                confidences=confidences_for_log,
                class_ids=int(Coco133AnnotationLayer.TRIANGULATED_3D),
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=tri_conf_rgb,
            ),
            static=True,
        )

    ## now do umeyama to align the triangulated hands to the gt hands
    gt_xyzc: Float[ndarray, "133 4"] = gt_xyzc_stack_all[ts_idx]
    valid_mask: Bool[ndarray, "133"] = (
        np.isfinite(xyzc_coco[:, :3]).all(axis=1)
        & np.isfinite(gt_xyzc[:, :3]).all(axis=1)
        & (xyzc_coco[:, 3] > 0.0)
        & (gt_xyzc[:, 3] > 0.0)
    )
    num_valid: int = int(np.count_nonzero(valid_mask))

    if num_valid < 3:
        print(f"[umeyama] not enough valid correspondences ({num_valid}) to estimate transform")
        return None

    # src_points live in the Oak rig/world implied by the current Oak camera extrinsics
    # because batch_triangulate solved using those projection matrices.
    src_points: Float[ndarray, "n_valid 3"] = xyzc_coco[valid_mask, :3].astype(np.float64, copy=False)
    # dst_points are GT joints expressed in the Quest/world frame coming from the labels.
    dst_points: Float[ndarray, "n_valid 3"] = gt_xyzc[valid_mask, :3].astype(np.float64, copy=False)

    # Similarity that maps Oak→new-Oak-world (destination ← source).
    new_oak_world_T_oak_world: Float32[ndarray, "4 4"] = umeyama_transform(
        src_points=src_points,
        dst_points=dst_points,
        allow_scaling=False,
    )
    src_points_h: Float[ndarray, "n_valid 4"] = np.concatenate(
        (src_points, np.ones((num_valid, 1), dtype=np.float64)),
        axis=1,
    )
    aligned_points: Float[ndarray, "n_valid 3"] = (new_oak_world_T_oak_world @ src_points_h.T).T[:, :3]
    # Project the aligned 3D points back into each ego camera for visual inspection.
    aligned_xyz_full: Float32[ndarray, "1 133 3"] = np.full((1, 133, 3), np.nan, dtype=np.float32)
    aligned_conf_full: Float32[ndarray, "1 133"] = np.zeros((1, 133), dtype=np.float32)
    aligned_xyz_full[0, valid_mask, :] = aligned_points.astype(np.float32, copy=False)
    aligned_conf_full[0, valid_mask] = xyzc_coco[valid_mask, 3].astype(np.float32, copy=False)
    uv_aligned: Float32[ndarray, "1 n_views 133 2"] = project_brown_conrady_grid(
        xyz_stack_world=aligned_xyz_full,
        pinholes_per_view=pinhole_param_list,
        filter_invalid=True,
    )
    aligned_conf_rgb: UInt8[ndarray, "1 133 3"] = confidence_scores_to_rgb(aligned_conf_full[..., np.newaxis])
    for view_idx, cam_name in enumerate(uvc_cam_names):
        if debug:
            pinhole_log_path: Path = parent_log_path / "ego" / cam_name / "pinhole"
            rr.log(
                f"{pinhole_log_path}/image/aligned_keypoints_2d",
                Points2DWithConfidence(
                    positions=uv_aligned[0, view_idx],
                    confidences=aligned_conf_full[0],
                    class_ids=int(Coco133AnnotationLayer.TRIANGULATED_3D),
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=aligned_conf_rgb[0],
                ),
            )
    residuals: Float[ndarray, "n_valid"] = np.linalg.norm(aligned_points - dst_points, axis=1)
    rms_error: float = float(np.sqrt(np.mean(residuals**2)))

    return UmeyamaResult(
        dst_T_src=new_oak_world_T_oak_world,
        n_matches=num_valid,
        rms_error_m=rms_error,
    )


def relog_ego_cams(
    ego_sequence: BaseEgoSequence,
    shortest_ego_timestamp: Int[ndarray, "n_frames"],
    parent_log_path: Path,
    timeline: str,
    recording: rr.RecordingStream | None,
    oak_alignment: OakQuestAlignmentConfig | None = None,
    debug: bool = False,
) -> None:
    """Relog ego camera intrinsics/extrinsics and log the quest↔oak baseline."""
    if oak_alignment is None:
        oak_alignment = OakQuestAlignmentConfig()
    ego_cam_dict: dict[str, list[Fisheye62Parameters | PinholeParameters]] = ego_sequence.ego_cam_dict

    quest_ref_cam_list: list[Fisheye62Parameters | PinholeParameters] | None = ego_cam_dict.get(
        oak_alignment.quest_reference_cam
    )
    oak_ref_cam_list: list[Fisheye62Parameters | PinholeParameters] | None = ego_cam_dict.get(
        oak_alignment.oak_reference_cam
    )
    if quest_ref_cam_list is None or oak_ref_cam_list is None:
        print(
            "[relog_ego_cams] missing reference cameras for baseline check: "
            f"quest='{oak_alignment.quest_reference_cam}' oak='{oak_alignment.oak_reference_cam}'"
        )
    else:
        n_frames_dist: int = min(len(quest_ref_cam_list), len(oak_ref_cam_list), len(shortest_ego_timestamp))
        if n_frames_dist > 0:
            quest_world_t: Float32[ndarray, "n_frames_dist 3"] = np.array(
                [quest_ref_cam_list[idx].extrinsics.world_t_cam for idx in range(n_frames_dist)],
                dtype=np.float32,
            )
            oak_world_t: Float32[ndarray, "n_frames_dist 3"] = np.array(
                [oak_ref_cam_list[idx].extrinsics.world_t_cam for idx in range(n_frames_dist)],
                dtype=np.float32,
            )
            line_strips: Float32[ndarray, "n_frames_dist 2 3"] = np.stack(
                (quest_world_t, oak_world_t),
                axis=1,
            )
            distances_m: Float32[ndarray, "n_frames_dist"] = np.linalg.norm(
                oak_world_t - quest_world_t,
                axis=1,
            ).astype(np.float32, copy=False)
            distance_labels: list[str] = [f"{float(dist):.3f} m" for dist in distances_m]
            durations_dist: Float[ndarray, "n_frames_dist"] = 1e-9 * shortest_ego_timestamp[:n_frames_dist]
            if debug:
                distance_log_path: Path = parent_log_path / "ego" / "oak_quest_distance"
                show_labels: list[bool] = [True] * n_frames_dist
                rr.send_columns(
                    f"{distance_log_path}",
                    indexes=[rr.TimeColumn(timeline, duration=durations_dist)],
                    columns=[
                        *rr.LineStrips3D.columns(
                            strips=line_strips,
                            labels=distance_labels,
                            show_labels=show_labels,
                        ),
                    ],
                    recording=recording,
                )
    for cam_name, ego_cam_param_list in tqdm(
        ego_cam_dict.items(),
        desc="Logging ego cameras",
        leave=False,
    ):
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
            rr.Pinhole(
                image_from_camera=first_cam.intrinsics.k_matrix,
                height=first_cam.intrinsics.height,
                width=first_cam.intrinsics.width,
                camera_xyz=getattr(
                    rr.ViewCoordinates,
                    first_cam.intrinsics.camera_conventions,
                ),
                image_plane_distance=ego_sequence.image_plane_distance,
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
        durations: Float[ndarray, "n_frames"] = 1e-9 * shortest_ego_timestamp[0 : len(batch_world_t_cam)]
        rr.send_columns(
            f"{cam_log_path}",
            indexes=[rr.TimeColumn(timeline, duration=durations)],
            columns=[
                *rr.Transform3D.columns(
                    translation=rearrange(batch_world_t_cam, "f d -> (f) d"),
                    mat3x3=rearrange(batch_world_R_cam, "f r c -> (f) r c"),
                ),
            ],
            recording=recording,
        )


def apply_umeyama_to_oak_cams(
    *,
    ego_sequence: BaseEgoSequence,
    new_oak_world_T_oak_world: Float32[ndarray, "4 4"],
    calib_frame_idx: int,
    oak_cam_names: tuple[str, ...] = ("left", "rgb", "right"),
    oak_ref_cam: str = "left",
    quest_ref_cam: str = "quest3_left",
) -> None:
    """Lock the Oak rig to the Quest reference camera using a single Umeyama delta.

    Steps:
    1) At the calibration frame, compute the fixed intra-rig offsets oak_ref→cam.
    2) Compute quest_ref→oak_ref at the calibration frame using the Umeyama result.
    3) For every frame, place the Oak rig under the current Quest ref pose with that
       fixed quest_ref→oak_ref offset, then reapply the rigid intra-rig offsets.
    """
    ego_cams_dict: dict[str, list[Fisheye62Parameters | PinholeParameters]] = ego_sequence.ego_cam_dict
    quest_ref_list: list[Fisheye62Parameters | PinholeParameters] | None = ego_cams_dict.get(quest_ref_cam)
    oak_ref_list: list[Fisheye62Parameters | PinholeParameters] | None = ego_cams_dict.get(oak_ref_cam)
    if quest_ref_list is None or oak_ref_list is None:
        raise ValueError(
            f"[apply_umeyama_to_oak_cams] missing reference cams: quest='{quest_ref_cam}' oak='{oak_ref_cam}'"
        )

    calib_frame_idx = int(calib_frame_idx)
    if calib_frame_idx < 0 or calib_frame_idx >= len(quest_ref_list):
        raise IndexError(
            f"[apply_umeyama_to_oak_cams] calib_frame_idx {calib_frame_idx} out of range (quest frames={len(quest_ref_list)})"
        )

    # 1) Fixed intra-rig offsets at calibration frame (oak_ref -> cam)
    oak_ref_cam_calib: Fisheye62Parameters | PinholeParameters = oak_ref_list[calib_frame_idx]
    world_T_oak_ref_calib: Float[ndarray, "4 4"] = oak_ref_cam_calib.extrinsics.world_T_cam  # world ← oak_ref
    oak_ref_T_cam_const: dict[str, Float32[ndarray, "4 4"]] = {}
    for cam_name in oak_cam_names:
        cam_list = ego_cams_dict.get(cam_name)
        if not cam_list:
            continue
        cam_param_calib = cam_list[min(calib_frame_idx, len(cam_list) - 1)]
        world_T_cam_calib: Float[ndarray, "4 4"] = cam_param_calib.extrinsics.world_T_cam
        oak_ref_T_cam: Float[ndarray, "4 4"] = np.linalg.inv(world_T_oak_ref_calib) @ world_T_cam_calib
        oak_ref_T_cam_const[cam_name] = oak_ref_T_cam.astype(np.float32, copy=False)

    # 2) quest_ref -> oak_ref offset at calibration (after Umeyama)
    quest_ref_cam_calib: Fisheye62Parameters | PinholeParameters = quest_ref_list[calib_frame_idx]
    world_T_quest_ref_calib: Float[ndarray, "4 4"] = quest_ref_cam_calib.extrinsics.world_T_cam  # world ← quest_ref
    quest_world_T_oak_ref_calib: Float32[ndarray, "4 4"] = (new_oak_world_T_oak_world @ world_T_oak_ref_calib).astype(
        np.float32, copy=False
    )  # quest/world ← oak_ref
    quest_ref_T_oak_ref_calib: Float32[ndarray, "4 4"] = (
        np.linalg.inv(world_T_quest_ref_calib) @ quest_world_T_oak_ref_calib
    ).astype(np.float32, copy=False)  # quest_ref ← oak_ref
    # 3) Rebuild every frame: world ← oak_ref = world←quest_ref @ quest_ref←oak_ref_calib
    n_frames: int = min(
        len(quest_ref_list), *(len(ego_cams_dict.get(name, [])) for name in oak_cam_names if name in ego_cams_dict)
    )
    for frame_idx in tqdm(range(n_frames), desc="Aligning Oak rig to Quest", leave=False):
        world_T_quest_ref: Float[ndarray, "4 4"] = quest_ref_list[frame_idx].extrinsics.world_T_cam
        world_T_oak_ref_aligned: Float[ndarray, "4 4"] = world_T_quest_ref @ quest_ref_T_oak_ref_calib

        for cam_name, oak_ref_T_cam in oak_ref_T_cam_const.items():
            cam_list = ego_cams_dict.get(cam_name, [])
            if frame_idx >= len(cam_list):
                continue
            cam_param = cam_list[frame_idx]
            world_T_cam_aligned: Float[ndarray, "4 4"] = world_T_oak_ref_aligned @ oak_ref_T_cam
            cam_list[frame_idx] = _clone_with_extrinsics(cam_param, world_T_cam_aligned)


def log_reprojected_gt_uv(
    *,
    ego_sequence: BaseEgoSequence,
    xyzc_stack: Float[ndarray, "n_frames 133 4"],
    parent_log_path: Path,
    timeline: str,
    timestamps_ns: Int[ndarray, "n_frames"],
    cam_names: tuple[str, ...] = ("left", "rgb", "right"),
) -> None:
    """Project GT 3D keypoints into ego cameras using current (aligned) extrinsics."""

    xyz_stack: Float[ndarray, "n_frames 133 3"] = xyzc_stack[:, :, :3]
    conf_stack: Float[ndarray, "n_frames 133"] = xyzc_stack[:, :, 3]
    colors: UInt8[ndarray, "n_frames 133 3"] = confidence_scores_to_rgb(confidence_scores=conf_stack[..., np.newaxis])

    for cam_name, ego_cam_param_list in tqdm(
        ego_sequence.ego_cam_dict.items(),
        desc="Reproject GT per ego cam",
        leave=False,
    ):
        if cam_name not in cam_names:
            continue
        if not ego_cam_param_list:
            continue

        pinhole_log_path: Path = parent_log_path / "ego" / cam_name / "pinhole"

        n_frames_total: int = min(len(xyz_stack), len(ego_cam_param_list), len(timestamps_ns))
        if n_frames_total == 0:
            continue

        xyz_trim: Float[ndarray, "n_frames 133 3"] = xyz_stack[:n_frames_total]
        conf_trim: Float[ndarray, "n_frames 133"] = conf_stack[:n_frames_total]
        color_trim: UInt8[ndarray, "n_frames 133 3"] = colors[:n_frames_total]

        if isinstance(ego_cam_param_list[0], PinholeParameters):
            pinhole_slice: list[PinholeParameters] = cast(list[PinholeParameters], ego_cam_param_list[:n_frames_total])
            uv_stack: Float[ndarray, "n_frames 133 2"] = project_brown_conrady_diagonal(
                xyz_stack_world=xyz_trim,
                pinholes_per_frame=pinhole_slice,
                filter_invalid=True,
            )
        else:
            raise NotImplementedError(
                f"Ego camera parameters of type '{type(ego_cam_param_list[0])}' are not supported."
            )

        positions_flat: Float[ndarray, "n_total 2"] = rearrange(
            uv_stack,
            "n_frames kpts dim -> (n_frames kpts) dim",
        ).astype(np.float32)
        colors_flat: UInt8[ndarray, "n_total 3"] = rearrange(
            color_trim,
            "n_frames kpts dim -> (n_frames kpts) dim",
        )
        confidences_flat: Float32[ndarray, "n_total"] = rearrange(
            conf_trim,
            "n_frames kpts -> (n_frames kpts)",
        ).astype(np.float32)
        n_keypoints: int = len(COCO_133_IDS)
        keypoint_lengths: Int[ndarray, "n_frames"] = np.full(n_frames_total, n_keypoints, dtype=np.int32)

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
                    duration=1e-9 * timestamps_ns[:n_frames_total],
                )
            ],
            columns=[
                *Points2DWithConfidence.columns(
                    positions=positions_flat,
                    colors=colors_flat,
                    confidences=confidences_flat,
                ).partition(keypoint_lengths),
            ],
        )


def umeyama_transform(
    *,
    src_points: Float[ndarray, "n 3"],
    dst_points: Float[ndarray, "n 3"],
    allow_scaling: bool = False,
    eps: float = 1e-9,
) -> Float32[ndarray, "4 4"]:
    """Compute the similarity transform aligning ``src_points`` to ``dst_points``.

    The returned transform follows the right-to-left convention::

        dst_points ≈ (dst_T_src[:3, :3] @ src_points.T).T + dst_T_src[:3, 3]

    Args:
        src_points: Source XYZ coordinates (e.g., triangulated Oak keypoints).
        dst_points: Target XYZ coordinates (e.g., quest ground-truth keypoints).
        allow_scaling: If True, estimate an isotropic scale factor; otherwise fix scale=1.
        eps: Small value guarding against degenerate variance.

    Returns:
        dst_R_src: Rotation matrix mapping source → destination.
        dst_t_src: Translation vector (destination frame) mapping source → destination.
        scale: Isotropic scale factor.
        dst_T_src: 4x4 homogeneous transform representing the similarity.

    Raises:
        ValueError: If fewer than 3 correspondences are provided or the source variance is degenerate.
    """
    if src_points.shape != dst_points.shape:
        msg = f"Source and destination shapes must match; got {src_points.shape} vs {dst_points.shape}"
        raise ValueError(msg)

    n_points: int = int(src_points.shape[0])
    if n_points < 3:
        msg = f"Umeyama transform requires at least 3 correspondences; received {n_points}"
        raise ValueError(msg)

    src_f64: Float[ndarray, "n 3"] = src_points.astype(np.float64, copy=False)
    dst_f64: Float[ndarray, "n 3"] = dst_points.astype(np.float64, copy=False)

    src_mean: Float[ndarray, "3"] = np.mean(src_f64, axis=0)
    dst_mean: Float[ndarray, "3"] = np.mean(dst_f64, axis=0)

    src_centered: Float[ndarray, "n 3"] = src_f64 - src_mean
    dst_centered: Float[ndarray, "n 3"] = dst_f64 - dst_mean

    covariance: Float[ndarray, "3 3"] = (dst_centered.T @ src_centered) / float(n_points)

    u: Float[ndarray, "3 3"]
    singular_vals: Float[ndarray, "3"]
    vt: Float[ndarray, "3 3"]
    u, singular_vals, vt = np.linalg.svd(covariance)

    reflection: Float[ndarray, "3"] = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        reflection = np.array([1.0, 1.0, -1.0], dtype=np.float64)

    dst_R_src: Float[ndarray, "3 3"] = u @ np.diag(reflection) @ vt

    src_var: float = float(np.mean(np.sum(src_centered**2, axis=1)))
    if src_var <= eps:
        msg = f"Source variance too small for stable Umeyama estimation (variance={src_var})"
        raise ValueError(msg)

    scale: float = 1.0
    if allow_scaling:
        scale = float(np.sum(singular_vals * reflection) / src_var)

    dst_t_src: Float[ndarray, "3"] = dst_mean - scale * (dst_R_src @ src_mean)

    dst_T_src: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    dst_T_src[:3, :3] = scale * dst_R_src
    dst_T_src[:3, 3] = dst_t_src

    return dst_T_src.astype(np.float32)


@dataclass
class RRDPipelineConfig:
    rr_config: RerunTyroConfig
    """Configuration for rerun logging."""
    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing an annotated ``BaseExoEgoSequence``."""
    calib_ts_nano: int | None = None
    """Optional nanosecond timestamp used to select calibration frames for cameras and MANO."""
    align_oak_rig: bool = True
    """Enable re-anchoring the Oak rig to the quest reference camera before logging."""
    oak_alignment: OakQuestAlignmentConfig = field(default_factory=OakQuestAlignmentConfig)
    """Name configuration controlling which cameras define the quest and Oak references."""
    debug: bool = False
    """Enable debug logging and visualizations."""


def main(config: RRDPipelineConfig) -> None:
    """Preserve the Tyro CLI entry point while delegating to the reusable pipeline helper."""
    run_full_exoego_pipeline(config=config, recording=None)


def _log_flat_scene(
    exoego_sequence: BaseExoEgoSequence,
    *,
    parent_log_path: Path,
    timeline: str,
    recording: rr.RecordingStream | None,
) -> SceneSetupResult:
    """Flat-layout (pre-rig) video logging for the OAK<->Quest calibration tool.

    This tool aligns *two* physical devices (an OAK rig to a Quest rig) and re-logs
    per-camera world poses post-alignment (see :func:`relog_ego_cams`), which does
    not fit the single-device COLMAP rig model the exoego catalog writer uses. It
    therefore keeps the original flat ``/world/{exo,ego}/{name}`` layout, decoupled
    from :func:`simplecv.apis.view_exoego.setup_scene`. Exo cameras get pinhole +
    video; ego cameras get pinhole + video + their (unaligned) per-frame pose, so a
    frustum is present even if Umeyama alignment fails — ``relog_ego_cams`` later
    overwrites the ego pose with the aligned one when alignment succeeds.
    """
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    exo_sequence = exoego_sequence.exo_sequence

    exo_ts_list: list[Int[ndarray, "n_frames"]] = []
    exo_video_log_paths: list[Path] | None = None
    if exo_sequence is not None:
        exo_video_blobs: dict[str, bytes] | None = getattr(exo_sequence, "_video_blobs", None)
        exo_paths: list[Path] = []
        for name, cam, video_file in zip(exo_sequence.exo_video_names, exo_sequence.exo_cam_list, exo_sequence.exo_video_paths, strict=True):
            cam_path: Path = parent_log_path / "exo" / name
            if cam is not None:
                log_pinhole(camera=cam, cam_log_path=cam_path, image_plane_distance=exo_sequence.image_plane_distance, static=True, recording=recording)
            video_log_path: Path = cam_path / "pinhole" / "video"
            exo_paths.append(video_log_path)
            video_source: bytes | Path = exo_video_blobs[name] if exo_video_blobs and name in exo_video_blobs else video_file
            exo_ts_list.append(log_video(video_source, video_log_path, timeline=timeline, recording=recording))
        exo_video_log_paths = exo_paths

    ego_ts_list: list[Int[ndarray, "n_frames"]] = []
    ego_video_log_paths: list[Path] | None = None
    if ego_sequence is not None:
        ego_video_blobs: dict[str, bytes] | None = getattr(ego_sequence, "_video_blobs", None)
        ego_paths: list[Path] = []
        for name, video_file in zip(ego_sequence.ego_video_names, ego_sequence.ego_video_paths, strict=True):
            video_log_path = parent_log_path / "ego" / name / "pinhole" / "video"
            ego_paths.append(video_log_path)
            video_source = ego_video_blobs[name] if ego_video_blobs and name in ego_video_blobs else video_file
            ego_ts_list.append(log_video(video_source, video_log_path, timeline=timeline, recording=recording))
        ego_video_log_paths = ego_paths

        # Log each ego camera's pinhole + (unaligned) per-frame world_T_cam up front,
        # as the old setup_scene did unconditionally. relog_ego_cams overwrites these
        # with the aligned poses on Umeyama success; logging them here keeps the ego
        # frusta visible when alignment fails.
        if ego_ts_list:
            shortest_ego: Int[ndarray, "n_frames"] = _choose_shortest_timeline_by_duration(ego_ts_list)
            for cam_name, cam_param_list in ego_sequence.ego_cam_dict.items():
                n_frames_cam: int = min(len(cam_param_list), len(shortest_ego))
                if n_frames_cam <= 0:
                    continue
                trimmed: list[PinholeParameters | Fisheye62Parameters] = cam_param_list[:n_frames_cam]
                cam_log_path: Path = parent_log_path / "ego" / str(cam_name)
                rr.log(
                    f"{cam_log_path}/pinhole",
                    PinholeWithDistortion.from_camera(trimmed[0], image_plane_distance=ego_sequence.image_plane_distance, include_distortion=True),
                    static=True,
                    recording=recording,
                )
                world_t_cam: Float[ndarray, "n_frames 3"] = np.array([c.extrinsics.world_t_cam for c in trimmed])
                world_R_cam: Float[ndarray, "n_frames 3 3"] = np.array([c.extrinsics.world_R_cam for c in trimmed])
                rr.send_columns(
                    f"{cam_log_path}",
                    indexes=[rr.TimeColumn(timeline, duration=1e-9 * shortest_ego[:n_frames_cam])],
                    columns=[*rr.Transform3D.columns(translation=world_t_cam, mat3x3=world_R_cam)],
                    recording=recording,
                )

    shortest_timestamp: Int[ndarray, "n_frames"] = _choose_shortest_timeline_by_duration(exo_ts_list + ego_ts_list)
    return SceneSetupResult(
        log_paths=LogPaths(exo_video_log_paths=exo_video_log_paths, ego_video_log_paths=ego_video_log_paths),
        shortest_timestamp=shortest_timestamp,
    )


def run_full_exoego_pipeline(config: RRDPipelineConfig, recording: rr.RecordingStream | None = None) -> None:
    """Execute the Exo/Ego pipeline; split from main so UI backends can supply explicit Rerun recordings."""
    parent_log_path = Path("world")
    timeline = "video_time"

    ###################
    # 0. Parse inputs #
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()  # one-liner
    ego_sequence_obj: object | None = exoego_sequence.ego_sequence
    if ego_sequence_obj is None:
        raise ValueError("Dataset setup failed to provide an ego sequence.")
    ego_sequence: BaseEgoSequence = cast(BaseEgoSequence, ego_sequence_obj)

    rr.log("/", exoego_sequence.world_coordinate_system, static=True, recording=recording)
    set_annotation_context(recording=recording)

    parent_log_path = Path("world")
    timeline: str = "video_time"

    scene_setup_result: SceneSetupResult = _log_flat_scene(
        exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        recording=recording,
    )
    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    ego_video_log_paths_opt: list[Path] | None = log_paths.ego_video_log_paths
    if ego_video_log_paths_opt is None:
        raise ValueError("Scene setup must return ego video log paths.")

    ego_video_log_paths: list[Path] = list(ego_video_log_paths_opt)

    exo_video_log_paths: list[Path] | None = log_paths.exo_video_log_paths
    if exo_video_log_paths is None:
        exo_video_log_paths = []

    final_container: rrb.ContainerLike = create_view_container(
        exo_video_log_paths=exo_video_log_paths,
        ego_video_log_paths=ego_video_log_paths,
    )
    blueprint = rrb.Blueprint(
        final_container,
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint, recording=recording)
    # log the ground truth from exoego
    # Flat-layout cam paths matching _log_flat_scene / relog_ego_cams (this tool
    # keeps /world/{exo,ego}/{name}, not the rig layout — see _log_flat_scene).
    exo_cam_paths: dict[str, Path] = (
        {name: parent_log_path / "exo" / name for name in exoego_sequence.exo_sequence.exo_video_names}
        if exoego_sequence.exo_sequence is not None
        else {}
    )
    ego_cam_paths: dict[str, Path] = {name: parent_log_path / "ego" / name for name in ego_sequence.ego_video_names}
    log_exoego_batch(
        exoego_sequence=exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        shortest_timestamp=shortest_timestamp,
        exo_cam_paths=exo_cam_paths,
        ego_cam_paths=ego_cam_paths,
    )

    if shortest_timestamp.size == 0:
        raise ValueError("Scene setup returned no timestamps to derive calibration frame.")

    # using the calibration timestamp, do the single frame hand detection and triangulation
    labels: ExoEgoLabels | None = exoego_sequence.exoego_labels
    if labels is None:
        raise ValueError("Expected ExoEgoLabels to be available for calibration.")
    gt_xyzc_stack_all: Float[ndarray, "n_frames 133 4"] = labels.xyzc_stack.astype(np.float32)
    label_timestamps_ns: Int[ndarray, "n_frames"] = (
        labels.timestamps_ns if labels.timestamps_ns is not None else shortest_timestamp
    )
    calib_ts_nano: int = config.calib_ts_nano if config.calib_ts_nano is not None else int(shortest_timestamp[0])
    calib_frame_idx: int = timestamp_to_frame_index(time_ns=calib_ts_nano, frame_timestamps_ns=shortest_timestamp)

    umeyama_result: UmeyamaResult | None = single_frame_mv_hands(
        ego_sequence=ego_sequence,
        gt_xyzc_stack_all=gt_xyzc_stack_all,
        parent_log_path=parent_log_path,
        timeline=timeline,
        shortest_timestamp=shortest_timestamp,
        ts_nano=calib_ts_nano,
        debug=config.debug,
    )

    if umeyama_result is not None:
        apply_umeyama_to_oak_cams(
            ego_sequence=ego_sequence,
            new_oak_world_T_oak_world=umeyama_result.dst_T_src,
            calib_frame_idx=calib_frame_idx,
            oak_ref_cam=config.oak_alignment.oak_reference_cam,
            quest_ref_cam=config.oak_alignment.quest_reference_cam,
        )
        relog_ego_cams(
            ego_sequence=ego_sequence,
            shortest_ego_timestamp=shortest_timestamp,
            parent_log_path=parent_log_path,
            timeline=timeline,
            recording=recording,
            oak_alignment=config.oak_alignment,
            debug=config.debug,
        )
        log_reprojected_gt_uv(
            ego_sequence=ego_sequence,
            xyzc_stack=gt_xyzc_stack_all,
            parent_log_path=parent_log_path,
            timeline=timeline,
            timestamps_ns=label_timestamps_ns,
        )
