"""Wilor-based hand tracking for ego camera sequences.

Detect hands and extract 2D keypoints from ego cameras, outputting COCO-133
format annotations for visualization in Rerun. Optionally triangulate 3D
keypoints from multi-view detections.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from tqdm import tqdm

from simplecv.apis.view_exoego import create_container, setup_scene
from simplecv.camera_parameters import Fisheye62Parameters, KannalaBrandtDistortion, PinholeParameters
from simplecv.configs.exoego_dataset_configs import AnnotatedExoEgoDatasetUnion
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
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
from simplecv.rerun_custom_types import (
    Points2DWithConfidence,
    Points3DWithConfidence,
    confidence_scores_to_rgb,
)
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.video_io import MultiVideoReader

device: str = "cuda" if torch.cuda.is_available() else "cpu"
CamNameType = str


@dataclass
class WilorEgoTrackingConfig:
    """Structured configuration for running the wilor ego tracking CLI."""

    rr_config: RerunTyroConfig
    """Command-line options for spawning and configuring the Rerun viewer."""

    dataset: AnnotatedExoEgoDatasetUnion
    """Dataset factory capable of producing an annotated ``BaseExoEgoSequence``."""

    skip_camera_names: str = ""
    """Comma-separated camera names to skip from hand detection (e.g., 'left,right,left_front,right_front')."""

    hand_conf: float = 0.3
    """Detection confidence threshold for hand detection."""

    frame_start: int = 0
    """First frame index to process."""

    frame_end: int | None = None
    """Last frame index to process (None = all frames)."""

    max_triangulate_frames: int | None = None
    """Maximum frames to triangulate (None = all frames)."""

    debug: bool = False
    """Enable debug logging (bounding boxes, additional info)."""

    detection_only: bool = False
    """Only run hand detection (bounding boxes), skip keypoint extraction for faster processing."""

    triangulate: bool = True
    """Enable 3D triangulation of keypoints from multiple views."""


def set_annotation_context(*, recording: rr.RecordingStream | None = None) -> None:
    """Register COCO-133 semantic metadata so subsequent logs show names/edges."""
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



def triangulate_frame(
    uvc_coco_list: list[Float32[ndarray, "133 3"]],
    cam_params_list: list[Fisheye62Parameters | PinholeParameters],
    parent_log_path: Path,
) -> Float[ndarray, "133 4"]:
    """Triangulate 3D keypoints from multi-view 2D detections.
    
    Args:
        uvc_coco_list: List of 2D keypoints per view ``[n_views][133, 3]`` (u, v, conf).
        cam_params_list: Camera parameters for each view.
        parent_log_path: Base log path for Rerun.
        
    Returns:
        3D keypoints ``[133, 4]`` (x, y, z, conf).
    """
    n_views: int = len(uvc_coco_list)
    if n_views < 2:
        return np.full((133, 4), np.nan, dtype=np.float32)
    
    # Stack 2D keypoints: [n_views, 133, 3]
    uvc_coco_stack: Float[ndarray, "n_views 133 3"] = np.stack(uvc_coco_list, axis=0)
    
    # Build intrinsics and distortion lists
    k_matrices: list[Float[ndarray, "3 3"]] = []
    distortions: list[KannalaBrandtDistortion | None] = []
    
    for cam in cam_params_list:
        k_matrix: Float[ndarray, "3 3"] | None = cam.intrinsics.k_matrix
        assert k_matrix is not None, "All cameras must include intrinsics.k_matrix for triangulation"
        k_matrices.append(k_matrix)
        
        if isinstance(cam, Fisheye62Parameters):
            distortions.append(cam.distortion)
        else:
            distortions.append(None)
    
    # TODO: Undistort if any camera has distortion (K_stack and distortions will be used here)
    _ = distortions  # Suppress unused warning until undistortion is implemented
    
    # Build projection matrices
    Pall: Float32[ndarray, "n_views 3 4"] = np.stack(
        [cam.projection_matrix for cam in cam_params_list],
        axis=0,
    ).astype(np.float32)
    
    # Triangulate
    xyzc_coco: Float[ndarray, "133 4"] = batch_triangulate(
        keypoints_2d=uvc_coco_stack,
        projection_matrices=Pall,
    )
    
    return xyzc_coco


def run_wilor_ego_tracking(
    exoego_sequence: BaseExoEgoSequence,
    config: WilorEgoTrackingConfig,
) -> None:
    """Run wilor hand tracking on ego cameras and log results to Rerun."""
    from wilor_nano.hand_detection import DetectionResult, HandDetector, HandDetectorConfig
    from wilor_nano.hand_keypoints import (
        HandKeypointDetectorConfig,
        KeypointResults,
        WilorHandKeypointDetector,
    )

    # Initialize models
    print(f"Initializing models on device: {device}")
    hand_detector: HandDetector = HandDetector(
        HandDetectorConfig(hf_wilor_repo_id="pablovela5620/wilor-nano", verbose=False)
    )
    
    # Only initialize keypoint model if needed
    hand_keypoint_model: WilorHandKeypointDetector | None = None
    if not config.detection_only:
        hand_keypoint_model = WilorHandKeypointDetector(
            HandKeypointDetectorConfig(hf_wilor_repo_id="pablovela5620/wilor-nano", verbose=False)
        )
        print("Wilor keypoint model initialized")
    else:
        print("Detection-only mode: skipping keypoint model initialization")

    # Parse skip list (cameras to skip for detection, but still show frustums)
    skip_set: frozenset[str] = frozenset(
        name.strip() for name in config.skip_camera_names.split(",") if name.strip()
    )

    # Get ego sequence
    ego_sequence: BaseEgoSequence | None = exoego_sequence.ego_sequence
    if ego_sequence is None:
        print("Error: No ego sequence available in this dataset.")
        return

    ego_video_readers: MultiVideoReader = ego_sequence.ego_video_readers
    ego_cam_names: list[str] = ego_sequence.ego_video_names

    # Active cameras for detection (not skipped)
    active_cam_names: list[str] = [name for name in ego_cam_names if name not in skip_set]
    active_cam_indices: list[int] = [ego_cam_names.index(name) for name in active_cam_names]

    if not active_cam_names:
        print(f"No cameras to process after skipping: {skip_set}")
        return

    print(f"Processing cameras for detection: {active_cam_names}")
    print(f"Skipping detection on: {skip_set}")

    # Timeline and path setup
    timeline: str = "video_time"
    parent_log_path: Path = Path("/world")

    # Log world coordinate system (required for proper 3D view)
    rr.log("/", exoego_sequence.world_coordinate_system, static=True)
    
    # RerunTyroConfig auto-initializes via __post_init__
    set_annotation_context()

    # Use setup_scene from view_exoego for proper camera/video logging
    scene_result = setup_scene(
        exoego_sequence,
        parent_log_path=parent_log_path,
        timeline=timeline,
        log_ego=True,
        log_exo=False,  # No exo cameras for this tool
        recording=None,
    )
    
    # Use canonical timestamps from scene setup (not computed from FPS)
    timestamps_ns: Int[ndarray, "n_frames"] = scene_result.shortest_timestamp
    n_total_frames: int = len(timestamps_ns)
    
    # Get all ego video log paths from scene setup
    all_ego_video_log_paths: list[Path] = scene_result.log_paths.ego_video_log_paths or []

    # Send blueprint with ALL cameras (not just active ones)
    blueprint: rrb.ContainerLike = create_container(
        ego_video_log_paths=all_ego_video_log_paths,
    )
    rr.send_blueprint(rrb.Blueprint(blueprint, collapse_panels=True))

    # Determine frame range
    frame_start: int = max(0, config.frame_start)
    frame_end: int = config.frame_end if config.frame_end is not None else n_total_frames
    frame_end = min(frame_end, n_total_frames)
    
    # Triangulation range (None = all frames)
    if config.max_triangulate_frames is None:
        triangulate_end: int = frame_end
    else:
        triangulate_end = min(frame_start + config.max_triangulate_frames, frame_end)

    print(f"Processing frames {frame_start} to {frame_end} (total: {n_total_frames})")
    if config.triangulate and not config.detection_only:
        print(f"Triangulating frames {frame_start} to {triangulate_end}")

    # Get camera parameters for triangulation
    ego_cam_dict: dict[CamNameType, list[PinholeParameters | Fisheye62Parameters]] = ego_sequence.ego_cam_dict

    # Process frames
    for frame_idx in tqdm(range(frame_start, frame_end), desc="Processing frames"):
        bgr_list: list[UInt8[ndarray, "H W 3"]] = ego_video_readers[frame_idx]
        ts_nano: int = int(timestamps_ns[frame_idx])
        rr.set_time(timeline=timeline, duration=1e-9 * ts_nano)

        # Collect 2D keypoints for triangulation
        uvc_coco_list: list[Float32[ndarray, "133 3"]] = []
        cam_params_for_tri: list[Fisheye62Parameters | PinholeParameters] = []

        for cam_idx, cam_name in zip(active_cam_indices, active_cam_names, strict=True):
            bgr_frame: UInt8[ndarray, "H W 3"] = bgr_list[cam_idx]
            rgb_frame: UInt8[ndarray, "H W 3"] = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            cam_log_path: Path = parent_log_path / "ego" / cam_name
            pinhole_log_path: Path = cam_log_path / "pinhole"

            uvc_coco: Float32[ndarray, "133 3"] = np.full((133, 3), np.nan, dtype=np.float32)
            det_results: DetectionResult = hand_detector(rgb_frame, hand_conf=config.hand_conf)

            assert hand_keypoint_model is not None

            if det_results.left_xyxy is not None:
                rr.log(
                    f"{pinhole_log_path}/image/{COCO133_ROI_LABELS[Coco133RoiLayer.LEFT_HAND]}",
                    rr.Boxes2D(
                        array=det_results.left_xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=int(Coco133RoiLayer.LEFT_HAND),
                        show_labels=True,
                    ),
                )
                wilor_preds = hand_keypoint_model(
                    rgb_hw3=rgb_frame, xyxy=det_results.left_xyxy, handedness="left"
                )
                if isinstance(wilor_preds, KeypointResults):
                    uvc_coco[LEFT_HAND_IDX, :2] = wilor_preds.keypoints_2d[0]
                    uvc_coco[LEFT_HAND_IDX, 2] = wilor_preds.scores[0]

            if det_results.right_xyxy is not None:
                rr.log(
                    f"{pinhole_log_path}/image/{COCO133_ROI_LABELS[Coco133RoiLayer.RIGHT_HAND]}",
                    rr.Boxes2D(
                        array=det_results.right_xyxy,
                        array_format=rr.Box2DFormat.XYXY,
                        class_ids=int(Coco133RoiLayer.RIGHT_HAND),
                        show_labels=True,
                    ),
                )
                wilor_preds = hand_keypoint_model(
                    rgb_hw3=rgb_frame, xyxy=det_results.right_xyxy, handedness="right"
                )
                if isinstance(wilor_preds, KeypointResults):
                    uvc_coco[RIGHT_HAND_IDX, :2] = wilor_preds.keypoints_2d[0]
                    uvc_coco[RIGHT_HAND_IDX, 2] = wilor_preds.scores[0]

            # Log 2D keypoints
            confidence_rgb_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
                uvc_coco[:, 2][np.newaxis, :, np.newaxis]
            )
            confidence_rgb: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_stack[0]

            rr.log(
                f"{pinhole_log_path}/coco133_uv",
                Points2DWithConfidence(
                    positions=uvc_coco[:, :2],
                    confidences=uvc_coco[:, 2],
                    class_ids=int(Coco133AnnotationLayer.RAW_2D),
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=confidence_rgb,
                ),
            )
            # Collect for triangulation
            if config.triangulate and frame_idx < triangulate_end:
                uvc_coco_list.append(uvc_coco)
                cam_param_list: list[Fisheye62Parameters | PinholeParameters] = ego_cam_dict.get(cam_name, [])
                if cam_param_list:
                    # Use frame-specific params if available, otherwise use first (constant calibration)
                    param_idx: int = min(frame_idx, len(cam_param_list) - 1)
                    cam_params_for_tri.append(cam_param_list[param_idx])

        
        # Triangulate if we have enough views
        if (
            config.triangulate
            and not config.detection_only
            and frame_idx < triangulate_end
            and len(uvc_coco_list) >= 2
            and len(cam_params_for_tri) == len(uvc_coco_list)
        ):
            xyzc_coco: Float[ndarray, "133 4"] = triangulate_frame(
                uvc_coco_list=uvc_coco_list,
                cam_params_list=cam_params_for_tri,
                parent_log_path=parent_log_path,
            )
            
            # Log 3D keypoints
            positions_for_log: Float32[ndarray, "133 3"] = xyzc_coco[:, :3].astype(np.float32, copy=True)
            confidences_for_log: Float32[ndarray, "133"] = xyzc_coco[:, 3].astype(np.float32, copy=True)
            
            # Mask invalid points
            invalid_mask: Bool[ndarray, "133"] = (~np.isfinite(confidences_for_log)) | (confidences_for_log <= 0.0)
            positions_for_log[invalid_mask, :] = np.nan
            
            tri_conf_rgb_stack: UInt8[ndarray, "1 133 3"] = confidence_scores_to_rgb(
                xyzc_coco[:, 3][np.newaxis, :, np.newaxis]
            )
            tri_conf_rgb: UInt8[ndarray, "133 3"] = tri_conf_rgb_stack[0]
            
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
            )


def main(config: WilorEgoTrackingConfig) -> None:
    """Entry point for wilor ego tracking CLI."""
    exoego_sequence: BaseExoEgoSequence = config.dataset.setup()
    run_wilor_ego_tracking(exoego_sequence, config)