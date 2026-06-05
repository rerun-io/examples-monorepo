from dataclasses import dataclass, field
from pathlib import Path
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
import torch
import tyro
from jaxtyping import Bool, Float, Float32, Int, UInt8
from monopriors.apis.multiview_calibration import MultiViewCalibrator, MultiViewCalibratorConfig, MVCalibResults
from numpy import ndarray
from simplecv.apis.view_exoego import LogPaths, SceneSetupResult, create_container
from simplecv.camera_parameters import Intrinsics, PinholeParameters
from simplecv.configs.exoego_dataset_configs import dataset_defaults as simplecv_dataset_defaults
from simplecv.data.ego.base_ego import BaseEgoSequence
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.exoego_config import BaseExoEgoDatasetConfig
from simplecv.data.exoego.hocap import HocapConfig
from simplecv.data.skeleton.coco133_layers import (
    COCO133_LAYER_COLORS,
    COCO133_LAYER_LABELS,
    COCO133_PREDICTION_LAYER_TO_PATH,
    Coco133AnnotationLayer,
)
from simplecv.data.skeleton.coco_133 import (
    COCO_133_ID2NAME,
    COCO_133_IDS,
    COCO_133_LINKS,
    FACE_IDX,
    LEFT_HAND_IDX,
    RIGHT_HAND_IDX,
)
from simplecv.ops.pc_utils import estimate_voxel_size
from simplecv.ops.triangulate import proj_3d_vectorized
from simplecv.ops.tsdf_depth_fuser import Open3DScaleInvariantFuser
from simplecv.rerun_custom_types import Points2DWithConfidence, Points3DWithConfidence, confidence_scores_to_rgb
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from simplecv.video_io import MultiVideoReader, TorchCodecMultiVideoReader, rgb_chw_tensor_to_bgr_hwc
from tqdm import tqdm
from wilor_nano.hand_keypoints import FinalWilorPred, HandKeypointDetectorConfig, KeypointResults, WilorHandKeypointDetector

from mv_api.data.synced_videos import SyncedVideoExoEgoConfig, frame_timestamps_from_reader
from mv_api.multiview_pose_estimator import (
    MultiviewBodyTracker,
    MultiviewBodyTrackerConfig,
    MVHistory,
    compute_square_bbox_from_uv,
)

np.set_printoptions(suppress=True)

CALIB_TARGET_RESOLUTION: tuple[int, int] = (1280, 720)

mv_api_dataset_defaults: dict[str, BaseExoEgoDatasetConfig] = {
    **simplecv_dataset_defaults,
    "synced-videos": SyncedVideoExoEgoConfig(),
}

if TYPE_CHECKING:
    MVAPIDatasetUnion = BaseExoEgoDatasetConfig
else:
    MVAPIDatasetUnion = tyro.extras.subcommand_type_from_defaults(mv_api_dataset_defaults, prefix_names=False)

AnnotatedMVAPIDatasetUnion = tyro.conf.OmitSubcommandPrefixes[MVAPIDatasetUnion]

CameraSource = Literal["auto", "estimated", "dataset"]
ResolvedCameraSource = Literal["estimated", "dataset"]


class MaskedPoints2D(NamedTuple):
    """2D points and confidences after applying image-domain visibility masks."""

    uv: Float32[ndarray, "n_kpts 2"]
    """Pixel coordinates with invalid or out-of-image keypoints set to NaN."""
    confidences: Float32[ndarray, "n_kpts"]
    """Per-keypoint confidence values with invalid or out-of-image entries set to NaN."""


@dataclass(frozen=True, slots=True)
class PoseCameraSelection:
    """Camera parameters selected for pose tracking and ego reprojection."""

    source: ResolvedCameraSource
    """Resolved camera source after applying automatic dataset-camera fallback."""
    exo_pinhole_param_list: list[PinholeParameters]
    """Exocentric camera parameters ordered like the exo video log paths."""
    ego_pinhole_param_lists: list[list[PinholeParameters]]
    """Egocentric camera trajectories ordered like the filtered ego RGB video log paths."""
    log_estimated_pinholes: bool
    """Whether estimated calibration pinholes should be logged over the scene pinholes."""


@dataclass
class RRDPipelineConfig:
    """Configuration for running the full exo/ego pipeline."""

    rr_config: RerunTyroConfig
    """Rerun logging configuration."""
    dataset: AnnotatedMVAPIDatasetUnion = field(default_factory=HocapConfig)
    """Dataset factory capable of producing an annotated exo/ego sequence."""
    calib_config: MultiViewCalibratorConfig = field(default_factory=MultiViewCalibratorConfig)
    """Parameters forwarded to the multi-view calibrator."""
    tracker_config: MultiviewBodyTrackerConfig = field(default_factory=MultiviewBodyTrackerConfig)
    """Configuration for the multiview body tracker."""
    camera_source: CameraSource = "auto"
    """
    Camera parameters used for pose tracking and reprojection.
    auto uses dataset-provided cameras when both exo and ego cameras exist, otherwise estimated cameras.
    estimated always uses cameras from the multi-view calibrator.
    dataset requires dataset-provided cameras and fails when they are unavailable.
    """
    calib_ts_nano: int | None = None
    """Optional nanosecond timestamp used to select calibration frames for cameras and hand models."""
    max_frames: int | None = None
    """Maximum number of frames to process. If None, all frames are processed."""


def resize_images_to_common_resolution(
    images: list[UInt8[ndarray, "H W 3"]],
    target_size: tuple[int, int] | None = None,
) -> list[UInt8[ndarray, "H W 3"]]:
    """Resize all images to a common width and height.

    Args:
        images: BGR or RGB images with potentially different resolutions.
        target_size: Optional ``(width, height)``. Uses the first image shape when omitted.

    Returns:
        Images resized with OpenCV linear interpolation and preserved ``uint8`` dtype.
    """
    if not images:
        return images

    if target_size is None:
        height: int = int(images[0].shape[0])
        width: int = int(images[0].shape[1])
        target_size = (width, height)

    target_width: int = int(target_size[0])
    target_height: int = int(target_size[1])
    resized: list[UInt8[ndarray, "H W 3"]] = []
    for image in images:
        image_height: int = int(image.shape[0])
        image_width: int = int(image.shape[1])
        if image_width == target_width and image_height == target_height:
            resized.append(image)
            continue
        resized_image: UInt8[ndarray, "H W 3"] = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        resized.append(resized_image)
    return resized


def _scale_pinhole_to_image_shape(
    *,
    camera: PinholeParameters,
    image_shape: tuple[int, int],
    source_size: tuple[int, int],
) -> PinholeParameters:
    """Scale a pinhole calibrated on resized pixels back to an image's native pixels."""
    source_width: int = int(source_size[0])
    source_height: int = int(source_size[1])
    target_height: int = int(image_shape[0])
    target_width: int = int(image_shape[1])
    scale_x: float = float(target_width) / float(source_width)
    scale_y: float = float(target_height) / float(source_height)
    intrinsics: Intrinsics = camera.intrinsics
    fl_x: float = float(cast(float, intrinsics.fl_x))
    fl_y: float = float(cast(float, intrinsics.fl_y))
    cx: float = float(cast(float, intrinsics.cx))
    cy: float = float(cast(float, intrinsics.cy))
    scaled_intrinsics: Intrinsics = Intrinsics.from_focal_principal_point(
        camera_conventions=intrinsics.camera_conventions,
        fl_x=fl_x * scale_x,
        fl_y=fl_y * scale_y,
        cx=cx * scale_x,
        cy=cy * scale_y,
        height=target_height,
        width=target_width,
    )
    return PinholeParameters(
        name=camera.name,
        extrinsics=camera.extrinsics,
        intrinsics=scaled_intrinsics,
        distortion=camera.distortion,
    )


def _mask_points_outside_intrinsics(
    *,
    uv: Float32[ndarray, "n_kpts 2"],
    confidences: Float32[ndarray, "n_kpts"],
    intrinsics: Intrinsics,
    margin: float = 0.0,
) -> MaskedPoints2D:
    """Mask 2D keypoints outside the image bounds described by camera intrinsics."""
    masked_uv: Float32[ndarray, "n_kpts 2"] = uv.astype(np.float32, copy=True)
    masked_confidences: Float32[ndarray, "n_kpts"] = confidences.astype(np.float32, copy=True)
    min_x: float = -float(margin)
    min_y: float = -float(margin)
    max_x: float = float(intrinsics.width - 1) + float(margin)
    max_y: float = float(intrinsics.height - 1) + float(margin)
    finite_mask: Bool[ndarray, "n_kpts"] = np.asarray(np.isfinite(masked_uv).all(axis=1), dtype=bool)
    in_bounds_mask: Bool[ndarray, "n_kpts"] = np.asarray(
        finite_mask
        & np.asarray(masked_uv[:, 0] >= min_x, dtype=bool)
        & np.asarray(masked_uv[:, 0] <= max_x, dtype=bool)
        & np.asarray(masked_uv[:, 1] >= min_y, dtype=bool)
        & np.asarray(masked_uv[:, 1] <= max_y, dtype=bool),
        dtype=bool,
    )
    masked_uv[~in_bounds_mask, :] = np.nan
    masked_confidences[~in_bounds_mask] = np.nan
    return MaskedPoints2D(uv=masked_uv, confidences=masked_confidences)


def _dataset_exo_pinhole_params(
    *,
    exo_sequence: BaseExoSequence,
    exo_video_log_paths: list[Path],
) -> list[PinholeParameters] | None:
    """Return dataset-provided exo pinholes ordered like the logged exo videos, if available."""
    exo_cam_by_name: dict[str, PinholeParameters] = {}
    try:
        exo_cam_list: list[PinholeParameters | None] = exo_sequence.exo_cam_list
    except AttributeError:
        return None
    for camera in exo_cam_list:
        if isinstance(camera, PinholeParameters):
            exo_cam_by_name[camera.name] = camera

    ordered_cameras: list[PinholeParameters] = []
    for video_log_path in exo_video_log_paths:
        camera_name: str = video_log_path.parent.parent.name
        camera: PinholeParameters | None = exo_cam_by_name.get(camera_name)
        if camera is None:
            return None
        ordered_cameras.append(camera)
    return ordered_cameras


def _dataset_ego_pinhole_param_lists(
    *,
    ego_sequence: BaseEgoSequence,
    ego_video_log_paths: list[Path],
) -> list[list[PinholeParameters]] | None:
    """Return dataset-provided ego pinhole trajectories ordered like the logged ego videos, if available."""
    try:
        ego_cam_dict: dict[Any, list[Any]] = cast(dict[Any, list[Any]], ego_sequence.ego_cam_dict)
    except AttributeError:
        return None
    ordered_camera_lists: list[list[PinholeParameters]] = []
    for video_log_path in ego_video_log_paths:
        camera_name: str = video_log_path.parent.parent.name
        camera_list_raw: list[Any] | None = ego_cam_dict.get(camera_name)
        if not camera_list_raw:
            return None
        camera_list: list[PinholeParameters] = []
        for camera in camera_list_raw:
            if not isinstance(camera, PinholeParameters):
                return None
            camera_list.append(camera)
        ordered_camera_lists.append(camera_list)
    return ordered_camera_lists


def _select_pose_camera_params(
    *,
    camera_source: CameraSource,
    estimated_exo_pinhole_param_list: list[PinholeParameters],
    estimated_ego_pinhole_param_list: list[PinholeParameters],
    dataset_exo_pinhole_param_list: list[PinholeParameters] | None,
    dataset_ego_pinhole_param_lists: list[list[PinholeParameters]] | None,
) -> PoseCameraSelection:
    """Select dataset or estimated cameras for pose tracking and ego reprojection."""
    estimated_ego_pinhole_param_lists: list[list[PinholeParameters]] = [
        [camera] for camera in estimated_ego_pinhole_param_list
    ]
    dataset_cameras_available: bool = (
        dataset_exo_pinhole_param_list is not None and dataset_ego_pinhole_param_lists is not None
    )

    if camera_source == "estimated":
        return PoseCameraSelection(
            source="estimated",
            exo_pinhole_param_list=estimated_exo_pinhole_param_list,
            ego_pinhole_param_lists=estimated_ego_pinhole_param_lists,
            log_estimated_pinholes=True,
        )

    if camera_source == "dataset":
        if not dataset_cameras_available:
            msg: str = "camera_source='dataset' requires dataset-provided exo and ego cameras."
            raise ValueError(msg)
        assert dataset_exo_pinhole_param_list is not None
        assert dataset_ego_pinhole_param_lists is not None
        return PoseCameraSelection(
            source="dataset",
            exo_pinhole_param_list=dataset_exo_pinhole_param_list,
            ego_pinhole_param_lists=dataset_ego_pinhole_param_lists,
            log_estimated_pinholes=False,
        )

    if dataset_cameras_available:
        assert dataset_exo_pinhole_param_list is not None
        assert dataset_ego_pinhole_param_lists is not None
        return PoseCameraSelection(
            source="dataset",
            exo_pinhole_param_list=dataset_exo_pinhole_param_list,
            ego_pinhole_param_lists=dataset_ego_pinhole_param_lists,
            log_estimated_pinholes=False,
        )

    return PoseCameraSelection(
        source="estimated",
        exo_pinhole_param_list=estimated_exo_pinhole_param_list,
        ego_pinhole_param_lists=estimated_ego_pinhole_param_lists,
        log_estimated_pinholes=True,
    )


def compute_square_bbox(
    hand_uv: Float32[ndarray, "21 2"],
    *,
    intrinsics: Intrinsics,
    expansion_ratio: float,
) -> Float32[ndarray, "4"] | None:
    """Return an expanded square ``XYXY`` box clipped to camera bounds."""
    return compute_square_bbox_from_uv(
        hand_uv=hand_uv,
        image_shape=(int(intrinsics.height), int(intrinsics.width)),
        expansion_ratio=expansion_ratio,
    )


def set_annotation_context(*, recording: rr.RecordingStream | None) -> None:
    """Register COCO-133 semantic metadata for all logged prediction layers."""
    keypoint_infos: list[rr.AnnotationInfo] = [
        rr.AnnotationInfo(id=kpt_id, label=name) for kpt_id, name in COCO_133_ID2NAME.items()
    ]
    class_descriptions: list[rr.ClassDescription] = []
    for layer in (
        Coco133AnnotationLayer.GT,
        Coco133AnnotationLayer.RAW_2D,
        Coco133AnnotationLayer.TRACKED_2D,
        Coco133AnnotationLayer.PROJECTED_2D,
        Coco133AnnotationLayer.OPTIMIZED_2D,
        Coco133AnnotationLayer.TRIANGULATED_3D,
    ):
        label: str = COCO133_LAYER_LABELS[layer]
        color: tuple[int, int, int] = COCO133_LAYER_COLORS[layer]
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(id=int(layer), label=label, color=color),
                keypoint_annotations=keypoint_infos,
                keypoint_connections=COCO_133_LINKS,
            )
        )
    rr.log("/", rr.AnnotationContext(class_descriptions), static=True, recording=recording)


def send_scene_blueprint(*, log_paths: LogPaths, recording: rr.RecordingStream | None) -> None:
    container: rrb.ContainerLike = create_container(
        exo_video_log_paths=log_paths.exo_video_log_paths,
        ego_video_log_paths=log_paths.ego_video_log_paths,
    )
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            contents=[container],
            column_shares=[4, 1],
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint, recording=recording)


def timestamp_to_frame_index(time_ns: int, frame_timestamps_ns: Int[ndarray, "num_frames"]) -> int:
    """Map a nanosecond timestamp to the closest frame index at or before that time."""
    idx: int = int(np.searchsorted(frame_timestamps_ns, time_ns, side="right") - 1)
    return max(0, min(idx, int(frame_timestamps_ns.shape[0]) - 1))


def frame_index_to_timestamp(frame_timestamps_ns: Int[ndarray, "num_frames"], frame_index: int) -> int:
    """Return the nanosecond timestamp associated with a frame index."""
    if frame_index < 0 or frame_index >= int(frame_timestamps_ns.shape[0]):
        msg: str = f"frame_index {frame_index} is outside the valid range [0, {frame_timestamps_ns.shape[0] - 1}]"
        raise IndexError(msg)
    timestamp_ns: int = int(frame_timestamps_ns[frame_index])
    return timestamp_ns


def _read_bgr_frame(*, reader: Any, frame_idx: int, stream_name: str) -> UInt8[ndarray, "H W 3"]:
    frame_obj: object = reader[frame_idx]
    if frame_obj is None:
        raise ValueError(f"Missing frame {frame_idx} for {stream_name}.")
    if torch.is_tensor(frame_obj):
        rgb_chw: UInt8[torch.Tensor, "3 H W"] = frame_obj.detach()
        try:
            return rgb_chw_tensor_to_bgr_hwc(rgb_chw)
        except ValueError as exc:
            raise ValueError(f"{exc} for {stream_name}.") from exc
    frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame_obj, dtype=np.uint8)
    if frame_array.ndim != 3 or frame_array.shape[2] != 3:
        raise ValueError(f"Expected BGR frame with shape (*, *, 3), got {frame_array.shape} for {stream_name}.")
    return frame_array


def _bgr_to_rgb(bgr: UInt8[ndarray, "H W 3"]) -> UInt8[ndarray, "H W 3"]:
    rgb_frame: UInt8[ndarray, "H W 3"] = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame with shape (*, *, 3), got {rgb_frame.shape}.")
    return rgb_frame


def _extract_wilor_uv(wilor_pred: FinalWilorPred | KeypointResults) -> Float32[ndarray, "1 21 2"]:
    raw_uv: Float[ndarray, "1 21 2"] = (
        wilor_pred.keypoints_2d if isinstance(wilor_pred, KeypointResults) else wilor_pred.pred_keypoints_2d
    )
    pred_uv: Float32[ndarray, "1 21 2"] = np.asarray(raw_uv, dtype=np.float32)
    return pred_uv


def predict_kpts3d_from_calibrated_videos(
    *,
    exo_video_readers: MultiVideoReader | TorchCodecMultiVideoReader,
    exo_cam_list: list[PinholeParameters],
    pose_tracker: MultiviewBodyTracker,
    top_half_mask: Bool[ndarray, "n_kpts"],
    shortest_timestamp: Int[ndarray, "num_frames"],
    parent_log_path: Path,
    timeline: str,
    max_frames: int | None = None,
    recording: rr.RecordingStream | None = None,
) -> list[Float32[ndarray, "n_kpts 4"]]:
    """Predict and triangulate COCO-133 keypoints from calibrated exo videos."""
    projection_matrices: Float32[ndarray, "n_views 3 4"] = np.stack(
        [cam.projection_matrix for cam in exo_cam_list]
    ).astype(np.float32)
    exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
        frame_timestamps_from_reader(reader) for reader in exo_video_readers.video_readers
    ]

    total: int = len(shortest_timestamp) if max_frames is None else min(len(shortest_timestamp), max_frames)
    timestamps_to_process: Int[ndarray, "num_frames"] = shortest_timestamp[:total]
    pbar = tqdm(timestamps_to_process, total=total)
    conf_thresh: float = pose_tracker.config.keypoint_threshold
    num_keypoints: int = int(pose_tracker.num_keypoints)
    mv_output: MVHistory = MVHistory()
    xyzc_list: list[Float32[ndarray, "n_kpts 4"]] = []
    triangulated_class_id: int = int(Coco133AnnotationLayer.TRIANGULATED_3D)

    for ts_idx, timestamp in enumerate(pbar):
        if max_frames is not None and ts_idx >= max_frames:
            break

        timestamp_ns: int = int(timestamp)
        rr.set_time(timeline=timeline, duration=np.timedelta64(timestamp_ns, "ns"), recording=recording)
        frame_indices: list[int] = [
            timestamp_to_frame_index(time_ns=timestamp_ns, frame_timestamps_ns=frame_timestamps)
            for frame_timestamps in exo_frame_timestamps_list
        ]
        bgr_list: list[UInt8[ndarray, "H W 3"]] = [
            _read_bgr_frame(reader=video_reader, frame_idx=frame_idx, stream_name=f"exo/{frame_idx}")
            for video_reader, frame_idx in zip(exo_video_readers.video_readers, frame_indices, strict=True)
        ]

        mv_output = pose_tracker(
            bgr_list=bgr_list,
            pinhole_list=exo_cam_list,
            pred_state=mv_output,
            recording=recording,
        )
        xyzc_frame: Float32[ndarray, "n_kpts 4"] = (
            mv_output.xyzc_t if mv_output.xyzc_t is not None else np.full((num_keypoints, 4), np.nan, dtype=np.float32)
        )
        xyzc_list.append(xyzc_frame)
        if mv_output.xyzc_t is None:
            continue

        vis_xyz: Float32[ndarray, "n_kpts 3"] = mv_output.xyzc_t[:, :3].copy()
        vis_scores_3d: Float32[ndarray, "n_kpts"] = mv_output.xyzc_t[:, 3].copy()
        vis_xyz[~top_half_mask, :] = np.nan
        vis_scores_3d[~top_half_mask] = np.nan
        vis_xyz[vis_scores_3d < conf_thresh, :] = np.nan
        vis_scores_3d[vis_scores_3d < conf_thresh] = np.nan

        left_hand_scores: Float32[ndarray, "n_left_hand"] = vis_scores_3d[LEFT_HAND_IDX]
        right_hand_scores: Float32[ndarray, "n_right_hand"] = vis_scores_3d[RIGHT_HAND_IDX]
        left_hand_valid_scores: Float32[ndarray, "n_left_hand_valid"] = left_hand_scores[np.isfinite(left_hand_scores)]
        right_hand_valid_scores: Float32[ndarray, "n_right_hand_valid"] = right_hand_scores[np.isfinite(right_hand_scores)]
        # Gate each hand as a coherent group to avoid logging fragmented hand skeletons.
        left_hand_conf: float = float(np.mean(left_hand_valid_scores, dtype=np.float32)) if left_hand_valid_scores.size else float("nan")
        right_hand_conf: float = (
            float(np.mean(right_hand_valid_scores, dtype=np.float32)) if right_hand_valid_scores.size else float("nan")
        )
        if left_hand_conf < conf_thresh:
            vis_xyz[LEFT_HAND_IDX, :] = np.nan
            vis_scores_3d[LEFT_HAND_IDX] = np.nan
        if right_hand_conf < conf_thresh:
            vis_xyz[RIGHT_HAND_IDX, :] = np.nan
            vis_scores_3d[RIGHT_HAND_IDX] = np.nan

        confidence_rgb_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
            vis_scores_3d[np.newaxis, :, np.newaxis]
        )
        confidence_rgb: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_stack[0]
        rr.log(
            str(parent_log_path / "gt" / "coco133_xyz"),
            Points3DWithConfidence(
                positions=vis_xyz,
                confidences=vis_scores_3d,
                class_ids=triangulated_class_id,
                keypoint_ids=COCO_133_IDS,
                show_labels=False,
                colors=confidence_rgb,
            ),
            recording=recording,
        )

        xyz_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate(
            [vis_xyz, np.ones((vis_xyz.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        xyz_hom_stack: Float32[ndarray, "1 n_kpts 4"] = xyz_hom[None]
        uv_exo_stack: Float[ndarray, "1 n_views n_kpts 2"] = proj_3d_vectorized(
            xyz_hom=xyz_hom_stack,
            P=projection_matrices,
        )
        uv_exo: Float32[ndarray, "n_views n_kpts 2"] = uv_exo_stack[0].astype(np.float32, copy=False)
        for uv_view, exo_cam in zip(uv_exo, exo_cam_list, strict=True):
            uv: Float32[ndarray, "n_kpts 2"] = uv_view.astype(np.float32, copy=True)
            confidences_view: Float32[ndarray, "n_kpts"] = vis_scores_3d.astype(np.float32, copy=True)
            masked_points: MaskedPoints2D = _mask_points_outside_intrinsics(
                uv=uv,
                confidences=confidences_view,
                intrinsics=exo_cam.intrinsics,
            )
            uv = masked_points.uv
            confidences_view = masked_points.confidences
            pinhole_log_path: Path = parent_log_path / "exo" / exo_cam.name / "pinhole"
            confidence_rgb_view_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
                confidences_view[np.newaxis, :, np.newaxis]
            )
            confidence_rgb_view: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_view_stack[0]
            rr.log(
                str(pinhole_log_path / "gt" / "coco133_uv"),
                Points2DWithConfidence(
                    positions=uv,
                    confidences=confidences_view,
                    class_ids=triangulated_class_id,
                    keypoint_ids=COCO_133_IDS,
                    show_labels=False,
                    colors=confidence_rgb_view,
                ),
                recording=recording,
            )

        mv_output.xyzc_t[~top_half_mask, :] = np.nan

    return xyzc_list


def run_model_backed_pipeline(
    *,
    config: RRDPipelineConfig,
    exoego_sequence: BaseExoEgoSequence,
    scene_setup_result: SceneSetupResult,
    parent_log_path: Path,
    timeline: str,
    recording: rr.RecordingStream | None,
) -> None:
    """Run the model-backed calibration and prediction section."""
    start: float = timer()
    projected_variant: str = COCO133_PREDICTION_LAYER_TO_PATH[Coco133AnnotationLayer.PROJECTED_2D]
    projected_class_id: int = int(Coco133AnnotationLayer.PROJECTED_2D)

    log_paths: LogPaths = scene_setup_result.log_paths
    shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

    if log_paths.exo_video_log_paths is None:
        raise ValueError("Scene setup must return exo video log paths.")
    if log_paths.ego_video_log_paths is None:
        raise ValueError("Scene setup must return ego video log paths.")
    if shortest_timestamp.size == 0:
        raise ValueError("Scene setup returned no timestamps to derive calibration frame.")

    exo_video_log_paths: list[Path] = list(log_paths.exo_video_log_paths)
    ego_video_log_paths: list[Path] = list(log_paths.ego_video_log_paths)

    exo_sequence_obj: object = exoego_sequence.exo_sequence
    if exo_sequence_obj is None:
        raise ValueError("Dataset setup failed to provide an exo sequence.")
    ego_sequence_obj: object = exoego_sequence.ego_sequence
    if ego_sequence_obj is None:
        raise ValueError("Dataset setup failed to provide an ego sequence.")

    exo_sequence: BaseExoSequence = cast(BaseExoSequence, exo_sequence_obj)
    ego_sequence: BaseEgoSequence = cast(BaseEgoSequence, ego_sequence_obj)
    exo_mv_reader: MultiVideoReader | TorchCodecMultiVideoReader = exo_sequence.exo_video_readers
    ego_mv_reader: MultiVideoReader | TorchCodecMultiVideoReader = ego_sequence.ego_video_readers

    ego_camera_names: list[str] = [path.parent.parent.name for path in ego_video_log_paths]
    ego_rgb_indices: list[int] = [
        idx for idx, camera_name in enumerate(ego_camera_names) if "rgb" in camera_name.lower()
    ]
    if not ego_rgb_indices:
        ego_rgb_indices = list(range(len(ego_video_log_paths)))
    ego_rgb_index_set: set[int] = set(ego_rgb_indices)
    ego_rgb_video_log_paths: list[Path] = [ego_video_log_paths[idx] for idx in ego_rgb_indices]

    dataset_exo_pinhole_param_list: list[PinholeParameters] | None = _dataset_exo_pinhole_params(
        exo_sequence=exo_sequence,
        exo_video_log_paths=exo_video_log_paths,
    )
    dataset_ego_pinhole_param_lists: list[list[PinholeParameters]] | None = _dataset_ego_pinhole_param_lists(
        ego_sequence=ego_sequence,
        ego_video_log_paths=ego_rgb_video_log_paths,
    )
    dataset_cameras_available: bool = (
        dataset_exo_pinhole_param_list is not None and dataset_ego_pinhole_param_lists is not None
    )

    rr.set_time(timeline=timeline, duration=np.timedelta64(0, "ns"), recording=recording)
    hand_kpt_detector: WilorHandKeypointDetector = WilorHandKeypointDetector(
        cfg=HandKeypointDetectorConfig(verbose=False)
    )
    camera_selection: PoseCameraSelection | None = None
    if config.camera_source == "dataset" or (config.camera_source == "auto" and dataset_cameras_available):
        camera_selection = _select_pose_camera_params(
            camera_source="dataset",
            estimated_exo_pinhole_param_list=[],
            estimated_ego_pinhole_param_list=[],
            dataset_exo_pinhole_param_list=dataset_exo_pinhole_param_list,
            dataset_ego_pinhole_param_lists=dataset_ego_pinhole_param_lists,
        )

    if camera_selection is None:
        calib_timestamp_ns: int = int(shortest_timestamp[-1]) if config.calib_ts_nano is None else int(config.calib_ts_nano)
        exo_frame_timestamp_list: list[Int[ndarray, "num_frames"]] = [
            frame_timestamps_from_reader(reader) for reader in exo_mv_reader.video_readers
        ]
        exo_calib_indices: list[int] = [
            timestamp_to_frame_index(time_ns=calib_timestamp_ns, frame_timestamps_ns=frame_timestamps)
            for frame_timestamps in exo_frame_timestamp_list
        ]
        bgr_list_exo: list[UInt8[ndarray, "H W 3"]] = [
            _read_bgr_frame(reader=reader, frame_idx=frame_idx, stream_name=f"exo/{frame_idx}")
            for reader, frame_idx in zip(exo_mv_reader.video_readers, exo_calib_indices, strict=True)
        ]
        exo_frame_shapes: list[tuple[int, int]] = [
            (int(frame.shape[0]), int(frame.shape[1])) for frame in bgr_list_exo
        ]

        ego_frame_timestamp_list: list[Int[ndarray, "num_frames"]] = [
            frame_timestamps_from_reader(reader) for reader in ego_mv_reader.video_readers
        ]
        ego_calib_indices: list[int] = [
            timestamp_to_frame_index(time_ns=calib_timestamp_ns, frame_timestamps_ns=frame_timestamps)
            for frame_timestamps in ego_frame_timestamp_list
        ]
        bgr_list_ego: list[UInt8[ndarray, "H W 3"]] = []
        for camera_idx, reader in enumerate(ego_mv_reader.video_readers):
            if camera_idx not in ego_rgb_index_set:
                continue
            frame_idx: int = ego_calib_indices[camera_idx]
            bgr_list_ego.append(_read_bgr_frame(reader=reader, frame_idx=frame_idx, stream_name=f"ego/{camera_idx}"))
        ego_rgb_frame_shapes: list[tuple[int, int]] = [
            (int(frame.shape[0]), int(frame.shape[1])) for frame in bgr_list_ego
        ]

        rgb_list_exo: list[UInt8[ndarray, "H W 3"]] = [_bgr_to_rgb(bgr) for bgr in bgr_list_exo]
        rgb_list_ego: list[UInt8[ndarray, "H W 3"]] = [_bgr_to_rgb(bgr) for bgr in bgr_list_ego]
        rgb_list: list[UInt8[ndarray, "H W 3"]] = resize_images_to_common_resolution(
            images=[*rgb_list_exo, *rgb_list_ego],
            target_size=CALIB_TARGET_RESOLUTION,
        )
        input_log_paths: list[Path] = [*exo_video_log_paths, *ego_rgb_video_log_paths]

        mv_calibrator: MultiViewCalibrator = MultiViewCalibrator(parent_log_path=parent_log_path, config=config.calib_config)
        mv_calib_results: MVCalibResults = mv_calibrator(rgb_list=rgb_list)

        pinhole_param_list: list[PinholeParameters] = mv_calib_results.pinhole_param_list
        exo_pinhole_param_list: list[PinholeParameters] = pinhole_param_list[: len(rgb_list_exo)]
        ego_pinhole_param_list: list[PinholeParameters] = pinhole_param_list[len(rgb_list_exo) :]
        for cam, log_path in zip(exo_pinhole_param_list, exo_video_log_paths, strict=True):
            cam.name = log_path.parent.parent.name
        for cam, log_path in zip(ego_pinhole_param_list, ego_rgb_video_log_paths, strict=True):
            cam.name = log_path.parent.parent.name

        if len(exo_pinhole_param_list) != len(rgb_list_exo):
            raise ValueError("Calibrator did not return one pinhole per exo calibration frame.")
        if len(ego_pinhole_param_list) != len(rgb_list_ego):
            raise ValueError("Calibrator did not return one pinhole per ego RGB calibration frame.")

        exo_video_pinhole_param_list: list[PinholeParameters] = [
            _scale_pinhole_to_image_shape(
                camera=cam,
                image_shape=frame_shape,
                source_size=CALIB_TARGET_RESOLUTION,
            )
            for cam, frame_shape in zip(exo_pinhole_param_list, exo_frame_shapes, strict=True)
        ]
        ego_video_pinhole_param_list: list[PinholeParameters] = [
            _scale_pinhole_to_image_shape(
                camera=cam,
                image_shape=frame_shape,
                source_size=CALIB_TARGET_RESOLUTION,
            )
            for cam, frame_shape in zip(ego_pinhole_param_list, ego_rgb_frame_shapes, strict=True)
        ]
        camera_selection = _select_pose_camera_params(
            camera_source="estimated",
            estimated_exo_pinhole_param_list=exo_video_pinhole_param_list,
            estimated_ego_pinhole_param_list=ego_video_pinhole_param_list,
            dataset_exo_pinhole_param_list=dataset_exo_pinhole_param_list,
            dataset_ego_pinhole_param_lists=dataset_ego_pinhole_param_lists,
        )

        video_pinhole_param_list: list[PinholeParameters] = [
            *exo_video_pinhole_param_list,
            *ego_video_pinhole_param_list,
        ]
        for pinhole, input_log_path in zip(video_pinhole_param_list, input_log_paths, strict=True):
            cam_log_path: Path = input_log_path.parent.parent
            log_pinhole(
                camera=pinhole,
                cam_log_path=cam_log_path,
                image_plane_distance=0.1,
                static=True,
                recording=recording,
            )

        pcd: o3d.geometry.PointCloud = mv_calib_results.pcd
        voxel_size: float = estimate_voxel_size(np.asarray(pcd.points, dtype=np.float32), target_points=50_000)
        pcd_ds: o3d.geometry.PointCloud = pcd.voxel_down_sample(voxel_size)
        filtered_points: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.points, dtype=np.float32)
        filtered_colors: Float32[ndarray, "final_points 3"] = np.asarray(pcd_ds.colors, dtype=np.float32)
        rr.log(
            str(parent_log_path / "gt" / "env_pointcloud"),
            rr.Points3D(filtered_points, colors=filtered_colors),
            static=True,
            recording=recording,
        )

        if mv_calib_results.depth_list and mv_calib_results.pinhole_param_list:
            depth_fuser: Open3DScaleInvariantFuser = Open3DScaleInvariantFuser(grid_resolution=512)
            reference_points: Float32[ndarray, "num_points 3"] = np.asarray(pcd.points, dtype=np.float32)
            depth_fuser.initialise_from_points(reference_points)

            for depth_map, pinhole_param, rgb in zip(
                mv_calib_results.depth_list,
                mv_calib_results.pinhole_param_list,
                rgb_list,
                strict=True,
            ):
                depth_fuser.fuse_frame(depth_hw=depth_map, pinhole=pinhole_param, rgb_hw3=rgb)

            gt_mesh: o3d.geometry.TriangleMesh = depth_fuser.get_mesh()
            gt_mesh.compute_vertex_normals()
            vertex_positions: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertices, dtype=np.float32)
            triangle_indices: Int[ndarray, "num_faces 3"] = np.asarray(gt_mesh.triangles, dtype=np.int32)
            vertex_normals: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_normals, dtype=np.float32)
            vertex_colors: Float32[ndarray, "num_vertices 3"] = np.asarray(gt_mesh.vertex_colors, dtype=np.float32)
            rr.log(
                str(parent_log_path / "gt" / "env_mesh"),
                rr.Mesh3D(
                    vertex_positions=vertex_positions,
                    triangle_indices=triangle_indices,
                    vertex_normals=vertex_normals,
                    vertex_colors=vertex_colors,
                ),
                static=True,
                recording=recording,
            )

    assert camera_selection is not None
    pose_exo_pinhole_param_list: list[PinholeParameters] = camera_selection.exo_pinhole_param_list
    pose_ego_pinhole_param_lists: list[list[PinholeParameters]] = camera_selection.ego_pinhole_param_lists

    upper_body_filter_idx: Int[ndarray, "upper_body"] = np.array([5, 6, 7, 8, 9, 10], dtype=np.int64)
    wb_upper_body_filter_idx: Int[ndarray, "upper_body_plus"] = np.concatenate(
        [upper_body_filter_idx, FACE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX]
    )
    top_half_mask: Bool[ndarray, "n_kpts"] = np.isin(np.arange(len(COCO_133_IDS)), wb_upper_body_filter_idx)
    pose_tracker: MultiviewBodyTracker = MultiviewBodyTracker(
        config.tracker_config,
        filter_body_idxes=wb_upper_body_filter_idx,
    )
    xyzc_list: list[Float32[ndarray, "n_kpts 4"]] = predict_kpts3d_from_calibrated_videos(
        exo_video_readers=exo_mv_reader,
        exo_cam_list=pose_exo_pinhole_param_list,
        pose_tracker=pose_tracker,
        top_half_mask=top_half_mask,
        shortest_timestamp=shortest_timestamp,
        parent_log_path=parent_log_path,
        timeline=timeline,
        max_frames=config.max_frames,
        recording=recording,
    )

    bbox_expansion_percentage: float = 0.25
    keypoint_threshold: float = pose_tracker.config.keypoint_threshold
    if len(xyzc_list) > 0 and len(pose_ego_pinhole_param_lists) > 0:
        for idx, xyzc in enumerate(xyzc_list):
            ego_pose_pinhole_param_list: list[PinholeParameters] = [
                camera_list[min(idx, len(camera_list) - 1)] for camera_list in pose_ego_pinhole_param_lists
            ]
            projection_matrices_ego: Float32[ndarray, "n_views 3 4"] = np.stack(
                [cam.projection_matrix for cam in ego_pose_pinhole_param_list]
            ).astype(np.float32)
            rr.set_time(timeline=timeline, duration=np.timedelta64(int(shortest_timestamp[idx]), "ns"), recording=recording)
            bgr_list_ego_frame: list[UInt8[ndarray, "H W 3"]] = []
            for camera_idx, frame in enumerate(ego_mv_reader[idx]):
                if camera_idx not in ego_rgb_index_set:
                    continue
                if frame is None:
                    raise ValueError(f"Missing ego frame at batch index {camera_idx} for timestamp index {idx}.")
                frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame, dtype=np.uint8)
                if frame_array.ndim != 3 or frame_array.shape[2] != 3:
                    raise ValueError(
                        f"Expected BGR frame with shape (*, *, 3), got {frame_array.shape} at index {camera_idx}."
                    )
                bgr_list_ego_frame.append(frame_array)

            if len(bgr_list_ego_frame) != len(ego_pose_pinhole_param_list):
                msg: str = (
                    "Filtered ego frame count does not match calibrated ego cameras: "
                    f"{len(bgr_list_ego_frame)} vs {len(ego_pose_pinhole_param_list)}."
                )
                raise ValueError(msg)

            vis_xyz: Float32[ndarray, "n_kpts 3"] = xyzc[:, :3].copy()
            vis_scores_3d: Float32[ndarray, "n_kpts"] = xyzc[:, 3].copy()
            vis_xyz[~top_half_mask, :] = np.nan
            vis_scores_3d[~top_half_mask] = np.nan
            vis_xyz[vis_scores_3d < keypoint_threshold, :] = np.nan
            vis_scores_3d[vis_scores_3d < keypoint_threshold] = np.nan

            xyz_hom: Float32[ndarray, "n_kpts 4"] = np.concatenate(
                [vis_xyz, np.ones((vis_xyz.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            xyz_hom_stack: Float32[ndarray, "1 n_kpts 4"] = xyz_hom[None]
            uv_ego_stack: Float[ndarray, "1 n_views n_kpts 2"] = proj_3d_vectorized(
                xyz_hom=xyz_hom_stack,
                P=projection_matrices_ego,
            )
            uv_ego: Float32[ndarray, "n_views n_kpts 2"] = uv_ego_stack[0].astype(np.float32, copy=False)

            for view_idx, (uv_view, ego_cam) in enumerate(zip(uv_ego, ego_pose_pinhole_param_list, strict=True)):
                pinhole_log_path: Path = parent_log_path / "ego" / ego_cam.name / "pinhole"
                uv: Float32[ndarray, "n_kpts 2"] = uv_view.astype(np.float32, copy=True)
                projection_matrix: Float32[ndarray, "3 4"] = ego_cam.projection_matrix.astype(np.float32, copy=False)
                homog_cam: Float32[ndarray, "n_kpts 3"] = np.einsum("ij,nj->ni", projection_matrix, xyz_hom)
                depth_cam: Float32[ndarray, "n_kpts"] = homog_cam[:, 2]
                invalid_depth_mask: Bool[ndarray, "n_kpts"] = np.logical_or(~np.isfinite(depth_cam), depth_cam <= 0.0)
                uv[invalid_depth_mask, :] = np.nan
                invalid_uv_mask: Bool[ndarray, "n_kpts"] = np.any(~np.isfinite(uv), axis=1)

                confidences_view: Float32[ndarray, "n_kpts"] = vis_scores_3d.astype(np.float32, copy=True)
                confidences_view[invalid_uv_mask] = np.nan
                rgb_hw3: UInt8[ndarray, "H W 3"] = bgr_list_ego_frame[view_idx][..., ::-1]

                left_bbox_infer: Float32[ndarray, "4"] | None = compute_square_bbox(
                    uv[LEFT_HAND_IDX, :],
                    intrinsics=ego_cam.intrinsics,
                    expansion_ratio=bbox_expansion_percentage,
                )
                if left_bbox_infer is not None:
                    xyxy_left: Float32[ndarray, "1 4"] = left_bbox_infer[np.newaxis, :]
                    wilor_left: FinalWilorPred | KeypointResults = hand_kpt_detector(
                        rgb_hw3=rgb_hw3,
                        xyxy=xyxy_left,
                        handedness="left",
                    )
                    left_uv_pred: Float32[ndarray, "1 21 2"] = _extract_wilor_uv(wilor_left)
                    uv[LEFT_HAND_IDX, :] = left_uv_pred[0]

                right_bbox_infer: Float32[ndarray, "4"] | None = compute_square_bbox(
                    uv[RIGHT_HAND_IDX, :],
                    intrinsics=ego_cam.intrinsics,
                    expansion_ratio=bbox_expansion_percentage,
                )
                if right_bbox_infer is not None:
                    xyxy_right: Float32[ndarray, "1 4"] = right_bbox_infer[np.newaxis, :]
                    wilor_right: FinalWilorPred | KeypointResults = hand_kpt_detector(
                        rgb_hw3=rgb_hw3,
                        xyxy=xyxy_right,
                        handedness="right",
                    )
                    right_uv_pred: Float32[ndarray, "1 21 2"] = _extract_wilor_uv(wilor_right)
                    uv[RIGHT_HAND_IDX, :] = right_uv_pred[0]

                left_bbox_log: Float32[ndarray, "4"] | None = compute_square_bbox(
                    uv[LEFT_HAND_IDX, :],
                    intrinsics=ego_cam.intrinsics,
                    expansion_ratio=bbox_expansion_percentage,
                )
                if left_bbox_log is not None:
                    rr.log(
                        f"{pinhole_log_path}/video/left_hand_bbox",
                        rr.Boxes2D(array=left_bbox_log.astype(np.float32, copy=False), array_format=rr.Box2DFormat.XYXY),
                        recording=recording,
                    )
                else:
                    rr.log(f"{pinhole_log_path}/video/left_hand_bbox", rr.Clear(recursive=True), recording=recording)

                right_bbox_log: Float32[ndarray, "4"] | None = compute_square_bbox(
                    uv[RIGHT_HAND_IDX, :],
                    intrinsics=ego_cam.intrinsics,
                    expansion_ratio=bbox_expansion_percentage,
                )
                if right_bbox_log is not None:
                    rr.log(
                        f"{pinhole_log_path}/video/right_hand_bbox",
                        rr.Boxes2D(array=right_bbox_log.astype(np.float32, copy=False), array_format=rr.Box2DFormat.XYXY),
                        recording=recording,
                    )
                else:
                    rr.log(f"{pinhole_log_path}/video/right_hand_bbox", rr.Clear(recursive=True), recording=recording)

                masked_points = _mask_points_outside_intrinsics(
                    uv=uv,
                    confidences=confidences_view,
                    intrinsics=ego_cam.intrinsics,
                )
                uv = masked_points.uv
                confidences_view = masked_points.confidences
                confidence_rgb_view_stack: UInt8[ndarray, "1 n_kpts 3"] = confidence_scores_to_rgb(
                    confidences_view[np.newaxis, :, np.newaxis]
                )
                confidence_rgb_view: UInt8[ndarray, "n_kpts 3"] = confidence_rgb_view_stack[0]
                rr.log(
                    str(pinhole_log_path / "pred" / "coco133_uv" / projected_variant),
                    Points2DWithConfidence(
                        positions=uv,
                        confidences=confidences_view,
                        class_ids=projected_class_id,
                        keypoint_ids=COCO_133_IDS,
                        show_labels=False,
                        colors=confidence_rgb_view,
                    ),
                    recording=recording,
                )

    print(f"Inference completed in {timer() - start:.2f} seconds")
