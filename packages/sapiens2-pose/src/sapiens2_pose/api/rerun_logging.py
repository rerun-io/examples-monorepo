"""Rerun logging helpers shared by image, video, CLI, and Gradio paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Int, UInt8, UInt16
from numpy import ndarray
from simplecv.rerun_custom_types import Points2DWithConfidence

from sapiens2_pose.api.metadata import PoseSchema, log_annotation_context

SEGMENTATION_ALPHA: int = 120
TRACK_COLORS: UInt8[ndarray, "palette 3"] = np.asarray(
    [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ],
    dtype=np.uint8,
)
SEGMENTATION_TRACK_COLORS: UInt8[ndarray, "palette 3"] = np.asarray(
    [
        [8, 48, 107],
        [8, 81, 156],
        [33, 113, 181],
        [66, 146, 198],
        [107, 174, 214],
        [158, 202, 225],
        [0, 109, 170],
        [0, 144, 189],
        [0, 164, 214],
        [55, 196, 226],
    ],
    dtype=np.uint8,
)


def build_image_blueprint() -> rrb.Blueprint:
    """Create a Rerun blueprint for single-image pose visualization."""
    return rrb.Blueprint(
        rrb.Spatial2DView(name="Pose", origin="image"),
        collapse_panels=True,
    )


def build_video_blueprint() -> rrb.Blueprint:
    """Create a Rerun blueprint for video pose visualization."""
    return rrb.Blueprint(
        rrb.Spatial2DView(name="Video + Pose", origin="video"),
        collapse_panels=True,
    )


def _build_track_annotation_context(
    *,
    colors: UInt8[ndarray, "palette 3"],
    max_tracks: int,
) -> rr.AnnotationContext:
    """Build per-track annotation colors."""
    class_descriptions: list[rr.ClassDescription] = [
        rr.ClassDescription(info=rr.AnnotationInfo(id=0, label="Background", color=(0, 0, 0, 0)))
    ]
    for track_idx in range(max_tracks):
        color_rgb: UInt8[ndarray, "3"] = colors[track_idx % colors.shape[0]]
        class_descriptions.append(
            rr.ClassDescription(
                info=rr.AnnotationInfo(
                    id=track_idx + 1,
                    label=f"track_{track_idx}",
                    color=(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]), SEGMENTATION_ALPHA),
                )
            )
        )
    return rr.AnnotationContext(class_descriptions)


def log_video_segmentation_annotation_context(*, recording: rr.RecordingStream | None = None, max_tracks: int = 256) -> None:
    """Log annotation colors for tracked segmentation overlays."""
    rr.log(
        "video/segmentation",
        _build_track_annotation_context(colors=SEGMENTATION_TRACK_COLORS, max_tracks=max_tracks),
        static=True,
        recording=recording,
    )
    rr.log(
        "video/tracked_boxes",
        _build_track_annotation_context(colors=TRACK_COLORS, max_tracks=max_tracks),
        static=True,
        recording=recording,
    )
    rr.log(
        "video/detr_initial_boxes",
        _build_track_annotation_context(colors=TRACK_COLORS, max_tracks=max_tracks),
        static=True,
        recording=recording,
    )


def _prepare_keypoints_for_logging(
    keypoints: Float32[ndarray, "k 2"],
    scores: Float32[ndarray, "k"],
    kpt_thr: float,
) -> tuple[Float32[ndarray, "k 2"], Float32[ndarray, "k"]]:
    """Mask low-confidence keypoints with NaN and return the length-reconciled scores."""
    keypoints_arr: Float32[ndarray, "k 2"] = np.asarray(keypoints, dtype=np.float32).copy()
    scores_arr: Float32[ndarray, "k"] = np.asarray(scores, dtype=np.float32).reshape(-1)
    valid_count: int = min(keypoints_arr.shape[0], scores_arr.shape[0])
    if valid_count < keypoints_arr.shape[0]:
        keypoints_arr[valid_count:] = np.nan
    reconciled_scores: Float32[ndarray, "k"] = np.zeros((keypoints_arr.shape[0],), dtype=np.float32)
    reconciled_scores[:valid_count] = scores_arr[:valid_count]
    low_confidence_indices: Int[ndarray, "low_confidence"] = np.flatnonzero(scores_arr[:valid_count] < kpt_thr).astype(np.int32)
    keypoints_arr[low_confidence_indices] = np.nan
    return keypoints_arr, reconciled_scores


def log_pose_instances(
    *,
    entity_root: str,
    bboxes: Float32[ndarray, "n 4"],
    keypoints: list[Float32[ndarray, "k 2"]],
    scores: list[Float32[ndarray, "k"]],
    schema: PoseSchema,
    kpt_thr: float,
    track_ids: Int[ndarray, "n"] | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log per-person 2D boxes and keypoints below an entity root."""
    keypoint_ids: Int[ndarray, "k"] = np.asarray(schema.keypoint_ids, dtype=np.int32)
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if track_ids is None:
        track_ids_arr: Int[ndarray, "n"] = np.arange(bboxes_f32.shape[0], dtype=np.int32)
    else:
        track_ids_arr = np.asarray(track_ids, dtype=np.int32).reshape(-1)

    for idx, (bbox, kpts, scr) in enumerate(zip(bboxes_f32, keypoints, scores, strict=False)):
        track_id: int = int(track_ids_arr[idx]) if idx < track_ids_arr.shape[0] else idx
        person_path: str = f"{entity_root}/person_{track_id}"
        bbox_arr: Float32[ndarray, "1 4"] = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
        color_rgb: UInt8[ndarray, "3"] = TRACK_COLORS[track_id % TRACK_COLORS.shape[0]]
        rr.log(
            f"{person_path}/bbox",
            rr.Boxes2D(
                array=bbox_arr,
                array_format=rr.Box2DFormat.XYXY,
                class_ids=0,
                labels=f"person_{track_id}",
                colors=(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])),
                show_labels=True,
            ),
            recording=recording,
        )

        prepared: tuple[Float32[ndarray, "k 2"], Float32[ndarray, "k"]] = _prepare_keypoints_for_logging(kpts, scr, kpt_thr)
        keypoints_arr: Float32[ndarray, "k 2"] = prepared[0]
        scores_f32: Float32[ndarray, "k"] = prepared[1]
        rr.log(
            f"{person_path}/keypoints",
            Points2DWithConfidence(
                positions=keypoints_arr,
                confidences=scores_f32,
                class_ids=0,
                keypoint_ids=keypoint_ids,
            ),
            recording=recording,
        )


def log_image_pose_recording(
    *,
    image_rgb: UInt8[ndarray, "h w 3"],
    bboxes: Float32[ndarray, "n 4"],
    keypoints: list[Float32[ndarray, "k 2"]],
    scores: list[Float32[ndarray, "k"]],
    schema: PoseSchema,
    kpt_thr: float,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log one image and its pose overlays to Rerun."""
    rr.send_blueprint(build_image_blueprint(), recording=recording)
    rr.set_time("iteration", recording=recording, sequence=0)
    log_annotation_context(schema, recording=recording)
    rr.log("image", rr.Image(image_rgb, color_model=rr.ColorModel.RGB), recording=recording)
    log_pose_instances(
        entity_root="image",
        bboxes=bboxes,
        keypoints=keypoints,
        scores=scores,
        schema=schema,
        kpt_thr=kpt_thr,
        recording=recording,
    )


def log_tracked_segmentation(
    *,
    track_ids: Int[ndarray, "n"],
    bboxes: Float32[ndarray, "n 4"],
    masks: Bool[ndarray, "n h w"],
    previous_track_ids: set[int] | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log tracked SAM3 masks and their derived boxes for one video frame."""
    track_ids_arr: Int[ndarray, "n"] = np.asarray(track_ids, dtype=np.int32).reshape(-1)
    masks_bool: Bool[ndarray, "n h w"] = np.asarray(masks, dtype=bool)
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if masks_bool.ndim != 3:
        raise ValueError("masks must have shape [n, h, w].")

    height: int = int(masks_bool.shape[1])
    width: int = int(masks_bool.shape[2])
    visible_count: int = min(int(track_ids_arr.shape[0]), int(bboxes_f32.shape[0]), int(masks_bool.shape[0]))
    visible_track_ids: Int[ndarray, "visible"] = track_ids_arr[:visible_count]
    visible_bboxes: Float32[ndarray, "visible 4"] = bboxes_f32[:visible_count]
    visible_masks: Bool[ndarray, "visible h w"] = masks_bool[:visible_count]

    seg_map: UInt16[ndarray, "h w"] = np.zeros((height, width), dtype=np.uint16)
    for idx, track_id in enumerate(visible_track_ids.tolist()):
        class_id: int = int(track_id) + 1
        seg_map = np.where(visible_masks[idx], np.uint16(class_id), seg_map)

    rr.log("video/segmentation", rr.SegmentationImage(seg_map), recording=recording)

    current_track_ids: set[int] = {int(track_id) for track_id in visible_track_ids.tolist()}
    if previous_track_ids is not None:
        stale_track_ids: set[int] = previous_track_ids - current_track_ids
        for stale_track_id in sorted(stale_track_ids):
            rr.log(f"video/tracked_boxes/track_{stale_track_id}", rr.Clear(recursive=True), recording=recording)

    if visible_count == 0:
        if previous_track_ids is None:
            rr.log("video/tracked_boxes", rr.Clear(recursive=True), recording=recording)
        return

    for idx, track_id in enumerate(visible_track_ids.tolist()):
        track_id_int: int = int(track_id)
        color_rgb: UInt8[ndarray, "3"] = TRACK_COLORS[track_id_int % TRACK_COLORS.shape[0]]
        rr.log(
            f"video/tracked_boxes/track_{track_id_int}",
            rr.Boxes2D(
                array=visible_bboxes[idx].reshape(1, 4),
                array_format=rr.Box2DFormat.XYXY,
                class_ids=track_id_int + 1,
                labels=f"track_{track_id_int}",
                colors=(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])),
                show_labels=True,
                draw_order=6,
            ),
            recording=recording,
        )


def log_initial_detr_boxes(
    *,
    frame_idx: int,
    timestamp_ns: int | None,
    bboxes: Float32[ndarray, "n 4"],
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log the first-frame DETR boxes used to seed SAM3 tracking."""
    if timestamp_ns is not None:
        rr.set_time("video_time", recording=recording, duration=1e-9 * float(timestamp_ns))
    rr.set_time("frame_idx", recording=recording, sequence=frame_idx)

    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if bboxes_f32.shape[0] == 0:
        rr.log("video/detr_initial_boxes", rr.Clear(recursive=True), recording=recording)
        return

    labels: list[str] = [f"detr_seed_{idx}" for idx in range(bboxes_f32.shape[0])]
    colors: UInt8[ndarray, "n 3"] = np.tile(np.asarray([[255, 0, 0]], dtype=np.uint8), (bboxes_f32.shape[0], 1))
    rr.log(
        "video/detr_initial_boxes",
        rr.Boxes2D(
            array=bboxes_f32,
            array_format=rr.Box2DFormat.XYXY,
            labels=labels,
            colors=colors,
            show_labels=True,
            draw_order=8,
        ),
        recording=recording,
    )


def clear_initial_detr_boxes(
    *,
    frame_idx: int,
    timestamp_ns: int | None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Clear first-frame DETR seed boxes after the seed frame."""
    if timestamp_ns is not None:
        rr.set_time("video_time", recording=recording, duration=1e-9 * float(timestamp_ns))
    rr.set_time("frame_idx", recording=recording, sequence=frame_idx)
    rr.log("video/detr_initial_boxes", rr.Clear(recursive=True), recording=recording)


def log_video_pose_frame(
    *,
    frame_idx: int,
    timestamp_ns: int | None,
    bboxes: Float32[ndarray, "n 4"],
    keypoints: list[Float32[ndarray, "k 2"]],
    scores: list[Float32[ndarray, "k"]],
    schema: PoseSchema,
    kpt_thr: float,
    previous_instance_count: int = 0,
    track_ids: Int[ndarray, "n"] | None = None,
    previous_track_ids: set[int] | None = None,
    previous_segmentation_track_ids: set[int] | None = None,
    masks: Bool[ndarray, "n h w"] | None = None,
    recording: rr.RecordingStream | None = None,
) -> None:
    """Log pose overlays for one video frame."""
    if timestamp_ns is not None:
        rr.set_time("video_time", recording=recording, duration=1e-9 * float(timestamp_ns))
    rr.set_time("frame_idx", recording=recording, sequence=frame_idx)

    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if track_ids is None:
        track_ids_arr_full: Int[ndarray, "n"] = np.arange(bboxes_f32.shape[0], dtype=np.int32)
    else:
        track_ids_arr_full = np.asarray(track_ids, dtype=np.int32).reshape(-1)

    if masks is not None:
        masks_bool: Bool[ndarray, "n h w"] = np.asarray(masks, dtype=bool)
        segmentation_count: int = min(int(bboxes_f32.shape[0]), int(track_ids_arr_full.shape[0]), int(masks_bool.shape[0]))
        log_tracked_segmentation(
            track_ids=track_ids_arr_full[:segmentation_count],
            bboxes=bboxes_f32[:segmentation_count],
            masks=masks_bool[:segmentation_count],
            previous_track_ids=previous_segmentation_track_ids,
            recording=recording,
        )

    current_instance_count: int = min(int(bboxes_f32.shape[0]), len(keypoints), len(scores), int(track_ids_arr_full.shape[0]))
    pose_track_ids: Int[ndarray, "m"] = track_ids_arr_full[:current_instance_count]
    log_pose_instances(
        entity_root="video",
        bboxes=bboxes_f32[:current_instance_count],
        keypoints=keypoints[:current_instance_count],
        scores=scores[:current_instance_count],
        schema=schema,
        kpt_thr=kpt_thr,
        track_ids=pose_track_ids,
        recording=recording,
    )
    current_track_ids: set[int] = {int(track_id) for track_id in pose_track_ids.tolist()}
    if previous_track_ids is not None:
        stale_track_ids: set[int] = previous_track_ids - current_track_ids
        for stale_track_id in sorted(stale_track_ids):
            rr.log(f"video/person_{stale_track_id}", rr.Clear(recursive=True), recording=recording)
        return

    for stale_idx in range(current_instance_count, previous_instance_count):
        rr.log(f"video/person_{stale_idx}", rr.Clear(recursive=True), recording=recording)


def save_recording(recording: rr.RecordingStream, rrd_path: Path) -> Path:
    """Save a Rerun recording and return the output path."""
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording.save(str(rrd_path))
    return rrd_path
