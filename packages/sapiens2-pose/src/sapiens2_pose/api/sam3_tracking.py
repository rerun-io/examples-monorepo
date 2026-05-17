"""SAM3 box-prompt video tracking for Sapiens2 video pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int, UInt8
from numpy import ndarray
from transformers import Sam3Model, Sam3Processor, Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

Sam3DTypeChoice = Literal["bfloat16", "float16", "float32"]
DeviceChoice = Literal["auto", "cpu", "cuda"]

DEFAULT_SAM3_CHECKPOINT: str = "facebook/sam3"
DEFAULT_SAM3_MASK_THRESHOLD: float = 0.5
DEFAULT_SAM3_MIN_MASK_AREA_PX: int = 100
DEFAULT_SAM3_MEMORY_RETENTION_FRAMES: int = 64
MIN_SEED_BOX_IOU: float = 0.01
DISABLED_TEXT_DETECTION_THRESHOLD: float = 1.1
BOX_SEED_PROMPT: str = "box_seed"

_DTYPE_MAP: dict[Sam3DTypeChoice, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_device(device: DeviceChoice = "auto") -> str:
    """Resolve a user device choice into a concrete torch device string."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


@dataclass(frozen=True, slots=True)
class Sam3TrackerConfig:
    """Configuration for DETR-box-seeded SAM3 video tracking."""

    checkpoint: str = DEFAULT_SAM3_CHECKPOINT
    """SAM3 checkpoint passed to Transformers model and processor loaders."""
    mask_threshold: float = DEFAULT_SAM3_MASK_THRESHOLD
    """Probability threshold used to binarize first-frame SAM3 image-prompt masks."""
    min_mask_area_px: int = DEFAULT_SAM3_MIN_MASK_AREA_PX
    """Minimum visible mask area before a track is exposed to pose estimation."""
    memory_retention_frames: int = DEFAULT_SAM3_MEMORY_RETENTION_FRAMES
    """Number of recent non-conditioning tracker states retained for long videos."""
    dtype: Sam3DTypeChoice = "bfloat16"
    """Torch dtype for SAM3 inference; half precision falls back to float32 on CPU."""


@dataclass(frozen=True, slots=True)
class TrackedDetections:
    """SAM3 tracked person detections for one video frame."""

    track_ids: Int[ndarray, "n"]
    """Stable SAM3 object IDs for visible tracks."""
    bboxes: Float32[ndarray, "n 4"]
    """Bounding boxes derived from visible masks in XYXY image coordinates."""
    masks: Bool[ndarray, "n h w"]
    """Visible binary masks aligned with ``track_ids`` and ``bboxes``."""
    scores: Float32[ndarray, "n"]
    """Per-track confidence scores from SAM3 where available."""


def _safe_dtype(dtype_choice: Sam3DTypeChoice, device: torch.device) -> torch.dtype:
    """Return a valid dtype for a device."""
    dtype: torch.dtype = _DTYPE_MAP[dtype_choice]
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _empty_detections(height: int, width: int) -> TrackedDetections:
    """Return an empty tracked-detection record for a frame shape."""
    track_ids: Int[ndarray, "0"] = np.empty((0,), dtype=np.int32)
    bboxes: Float32[ndarray, "0 4"] = np.empty((0, 4), dtype=np.float32)
    masks: Bool[ndarray, "0 h w"] = np.zeros((0, height, width), dtype=bool)
    scores: Float32[ndarray, "0"] = np.empty((0,), dtype=np.float32)
    return TrackedDetections(track_ids=track_ids, bboxes=bboxes, masks=masks, scores=scores)


def boxes_to_rectangular_masks(
    bboxes: Float32[ndarray, "n 4"],
    *,
    height: int,
    width: int,
) -> Bool[ndarray, "n h w"]:
    """Rasterize XYXY boxes into binary rectangular masks."""
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    masks: Bool[ndarray, "n h w"] = np.zeros((bboxes_f32.shape[0], height, width), dtype=bool)
    for idx, bbox in enumerate(bboxes_f32):
        x0_float: float = float(np.floor(bbox[0]))
        y0_float: float = float(np.floor(bbox[1]))
        x1_float: float = float(np.ceil(bbox[2]))
        y1_float: float = float(np.ceil(bbox[3]))
        x0: int = max(0, min(width, int(x0_float)))
        y0: int = max(0, min(height, int(y0_float)))
        x1: int = max(0, min(width, int(x1_float)))
        y1: int = max(0, min(height, int(y1_float)))
        if x1 > x0 and y1 > y0:
            masks[idx, y0:y1, x0:x1] = True
    return masks


def box_iou_matrix(
    boxes_a: Float32[ndarray, "a 4"],
    boxes_b: Float32[ndarray, "b 4"],
) -> Float32[ndarray, "a b"]:
    """Compute pairwise IoU for XYXY boxes."""
    boxes_a_f32: Float32[ndarray, "a 4"] = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    boxes_b_f32: Float32[ndarray, "b 4"] = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if boxes_a_f32.shape[0] == 0 or boxes_b_f32.shape[0] == 0:
        return np.zeros((boxes_a_f32.shape[0], boxes_b_f32.shape[0]), dtype=np.float32)

    x0: Float32[ndarray, "a b"] = np.maximum(boxes_a_f32[:, None, 0], boxes_b_f32[None, :, 0])
    y0: Float32[ndarray, "a b"] = np.maximum(boxes_a_f32[:, None, 1], boxes_b_f32[None, :, 1])
    x1: Float32[ndarray, "a b"] = np.minimum(boxes_a_f32[:, None, 2], boxes_b_f32[None, :, 2])
    y1: Float32[ndarray, "a b"] = np.minimum(boxes_a_f32[:, None, 3], boxes_b_f32[None, :, 3])
    intersection_w: Float32[ndarray, "a b"] = np.maximum(0.0, x1 - x0)
    intersection_h: Float32[ndarray, "a b"] = np.maximum(0.0, y1 - y0)
    intersection: Float32[ndarray, "a b"] = intersection_w * intersection_h

    area_a: Float32[ndarray, "a"] = np.maximum(0.0, boxes_a_f32[:, 2] - boxes_a_f32[:, 0]) * np.maximum(
        0.0,
        boxes_a_f32[:, 3] - boxes_a_f32[:, 1],
    )
    area_b: Float32[ndarray, "b"] = np.maximum(0.0, boxes_b_f32[:, 2] - boxes_b_f32[:, 0]) * np.maximum(
        0.0,
        boxes_b_f32[:, 3] - boxes_b_f32[:, 1],
    )
    union: Float32[ndarray, "a b"] = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union > 0.0)


def clip_masks_to_boxes(
    masks: Bool[ndarray, "n h w"],
    bboxes: Float32[ndarray, "n 4"],
) -> Bool[ndarray, "n h w"]:
    """Constrain masks to their corresponding XYXY boxes."""
    masks_bool: Bool[ndarray, "n h w"] = np.asarray(masks, dtype=bool)
    bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if masks_bool.ndim != 3:
        raise ValueError("masks must have shape [n, h, w].")

    height: int = int(masks_bool.shape[1])
    width: int = int(masks_bool.shape[2])
    clipped_masks: Bool[ndarray, "n h w"] = np.zeros_like(masks_bool, dtype=bool)
    box_masks: Bool[ndarray, "n h w"] = boxes_to_rectangular_masks(bboxes_f32, height=height, width=width)
    valid_count: int = min(masks_bool.shape[0], box_masks.shape[0])
    clipped_masks[:valid_count] = masks_bool[:valid_count] & box_masks[:valid_count]
    return clipped_masks


def masks_to_boxes(
    masks: Bool[ndarray, "n h w"],
    *,
    min_area_px: int = DEFAULT_SAM3_MIN_MASK_AREA_PX,
) -> tuple[Int[ndarray, "m"], Float32[ndarray, "m 4"], Bool[ndarray, "m h w"]]:
    """Convert visible masks into XYXY boxes and keep indices."""
    masks_bool: Bool[ndarray, "n h w"] = np.asarray(masks, dtype=bool)
    if masks_bool.ndim != 3:
        raise ValueError("masks must have shape [n, h, w].")

    keep_indices_list: list[int] = []
    boxes_list: list[list[float]] = []
    kept_masks_list: list[Bool[ndarray, "h w"]] = []
    for idx, mask in enumerate(masks_bool):
        area_px: int = int(mask.sum())
        if area_px < min_area_px:
            continue
        ys: Int[ndarray, "area"]
        xs: Int[ndarray, "area"]
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        x0: float = float(xs.min())
        y0: float = float(ys.min())
        x1: float = float(xs.max() + 1)
        y1: float = float(ys.max() + 1)
        keep_indices_list.append(idx)
        boxes_list.append([x0, y0, x1, y1])
        kept_masks_list.append(mask)

    keep_indices: Int[ndarray, "m"] = np.asarray(keep_indices_list, dtype=np.int32)
    if len(boxes_list) == 0:
        height: int = int(masks_bool.shape[1])
        width: int = int(masks_bool.shape[2])
        boxes: Float32[ndarray, "0 4"] = np.empty((0, 4), dtype=np.float32)
        kept_masks: Bool[ndarray, "0 h w"] = np.zeros((0, height, width), dtype=bool)
        return keep_indices, boxes, kept_masks

    boxes = np.asarray(boxes_list, dtype=np.float32).reshape(-1, 4)
    kept_masks = np.stack(kept_masks_list, axis=0).astype(bool, copy=False)
    return keep_indices, boxes, kept_masks


class Sam3BoxPromptVideoTracker:
    """Track first-frame DETR boxes through a video with SAM3 mask propagation."""

    def __init__(self, config: Sam3TrackerConfig, *, device: DeviceChoice = "auto"):
        self.config: Sam3TrackerConfig = config
        self.device: torch.device = torch.device(resolve_device(device))
        self.dtype: torch.dtype = _safe_dtype(config.dtype, self.device)
        self.image_model: Any | None = None
        self.image_processor: Any | None = None
        self.video_model: Any | None = None
        self.video_processor: Any | None = None
        self.inference_session: Any | None = None
        self.prompt_id: int = 0
        self._next_internal_frame_idx: int = 0

    def _load_image_model(self) -> tuple[Any, Any]:
        """Load the SAM3 image model and processor lazily."""
        if self.image_model is None or self.image_processor is None:
            image_model: Any = Sam3Model.from_pretrained(self.config.checkpoint).eval().to(self.device)
            image_processor: Any = Sam3Processor.from_pretrained(self.config.checkpoint)
            self.image_model = image_model
            self.image_processor = image_processor
        return self.image_model, self.image_processor

    def _load_video_model(self) -> tuple[Any, Any]:
        """Load the SAM3 video model and processor lazily."""
        if self.video_model is None or self.video_processor is None:
            video_config: Any = Sam3VideoConfig.from_pretrained(
                self.config.checkpoint,
                score_threshold_detection=DISABLED_TEXT_DETECTION_THRESHOLD,
                new_det_thresh=DISABLED_TEXT_DETECTION_THRESHOLD,
            )
            video_model: Any = Sam3VideoModel.from_pretrained(self.config.checkpoint, config=video_config).eval().to(self.device)
            if self.dtype != torch.float32:
                video_model = video_model.to(dtype=self.dtype)
            video_processor: Any = Sam3VideoProcessor.from_pretrained(self.config.checkpoint)
            self.video_model = video_model
            self.video_processor = video_processor
        return self.video_model, self.video_processor

    def _box_prompt_masks(
        self,
        frame_rgb: UInt8[ndarray, "h w 3"],
        bboxes: Float32[ndarray, "n 4"],
    ) -> Bool[ndarray, "n h w"]:
        """Create first-frame SAM3 masks from DETR boxes using box prompts."""
        height: int = int(frame_rgb.shape[0])
        width: int = int(frame_rgb.shape[1])
        bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if bboxes_f32.shape[0] == 0:
            return np.zeros((0, height, width), dtype=bool)

        image_model: Any
        image_processor: Any
        image_model, image_processor = self._load_image_model()
        box_prompts: list[list[list[float]]] = [bboxes_f32.tolist()]
        box_labels: list[list[int]] = [[1 for _bbox in bboxes_f32]]
        inputs: Any = image_processor(
            images=frame_rgb,
            input_boxes=box_prompts,
            input_boxes_labels=box_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs: Any = image_model(**inputs)

        return self._select_seed_masks_from_outputs(
            outputs=outputs,
            bboxes=bboxes_f32,
            height=height,
            width=width,
        )

    def _select_seed_masks_from_outputs(
        self,
        *,
        outputs: Any,
        bboxes: Float32[ndarray, "n 4"],
        height: int,
        width: int,
    ) -> Bool[ndarray, "n h w"]:
        """Select one SAM3 candidate mask per DETR seed box."""
        bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        fallback_masks: Bool[ndarray, "n h w"] = boxes_to_rectangular_masks(bboxes_f32, height=height, width=width)
        if bboxes_f32.shape[0] == 0:
            return fallback_masks

        pred_masks: torch.Tensor = outputs.pred_masks[0]
        if pred_masks.shape[0] == 0:
            return fallback_masks

        scale: torch.Tensor = torch.tensor([width, height, width, height], dtype=outputs.pred_boxes.dtype, device=outputs.pred_boxes.device)
        candidate_boxes: Float32[ndarray, "q 4"] = (outputs.pred_boxes[0] * scale).detach().cpu().numpy().astype(np.float32, copy=False)
        box_ious: Float32[ndarray, "n q"] = box_iou_matrix(bboxes_f32, candidate_boxes)
        pred_scores: Float32[ndarray, "q"] = outputs.pred_logits[0].sigmoid().detach().cpu().numpy().astype(np.float32, copy=False)
        if outputs.presence_logits is not None:
            presence_score: float = float(outputs.presence_logits[0].sigmoid().detach().cpu().reshape(-1)[0])
            pred_scores = pred_scores * presence_score

        selected_query_indices: list[int] = []
        selected_seed_indices: list[int] = []
        used_query_indices: set[int] = set()
        for seed_idx in range(bboxes_f32.shape[0]):
            ranking: Float32[ndarray, "q"] = box_ious[seed_idx] + (0.001 * pred_scores)
            for query_idx in np.argsort(-ranking).tolist():
                if int(query_idx) in used_query_indices:
                    continue
                if float(box_ious[seed_idx, int(query_idx)]) < MIN_SEED_BOX_IOU:
                    break
                selected_query_indices.append(int(query_idx))
                selected_seed_indices.append(seed_idx)
                used_query_indices.add(int(query_idx))
                break

        if len(selected_query_indices) == 0:
            return fallback_masks

        selected_logits: torch.Tensor = pred_masks[selected_query_indices]
        selected_probs: torch.Tensor = torch.nn.functional.interpolate(
            selected_logits.sigmoid().unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        selected_masks_np: Bool[ndarray, "m h w"] = (selected_probs >= self.config.mask_threshold).detach().cpu().numpy().astype(bool, copy=False)
        selected_boxes: Float32[ndarray, "m 4"] = bboxes_f32[np.asarray(selected_seed_indices, dtype=np.int32)]
        selected_masks_np = clip_masks_to_boxes(selected_masks_np, selected_boxes)

        seed_masks: Bool[ndarray, "n h w"] = fallback_masks.copy()
        for selected_idx, seed_idx in enumerate(selected_seed_indices):
            mask_area: int = int(selected_masks_np[selected_idx].sum())
            if mask_area >= self.config.min_mask_area_px:
                seed_masks[seed_idx] = selected_masks_np[selected_idx]
        return seed_masks

    def start(
        self,
        *,
        frame_idx: int,
        frame_rgb: UInt8[ndarray, "h w 3"],
        initial_bboxes: Float32[ndarray, "n 4"],
    ) -> TrackedDetections:
        """Seed SAM3 video tracking from first-frame person boxes."""
        height: int = int(frame_rgb.shape[0])
        width: int = int(frame_rgb.shape[1])
        bboxes_f32: Float32[ndarray, "n 4"] = np.asarray(initial_bboxes, dtype=np.float32).reshape(-1, 4)
        if bboxes_f32.shape[0] == 0:
            return _empty_detections(height, width)

        seed_masks: Bool[ndarray, "n h w"] = self._box_prompt_masks(frame_rgb, bboxes_f32)
        visible_indices: Int[ndarray, "m"]
        visible_bboxes: Float32[ndarray, "m 4"]
        visible_masks: Bool[ndarray, "m h w"]
        visible_indices, visible_bboxes, visible_masks = masks_to_boxes(
            seed_masks,
            min_area_px=self.config.min_mask_area_px,
        )
        if visible_bboxes.shape[0] == 0:
            return _empty_detections(height, width)

        video_model: Any
        video_processor: Any
        video_model, video_processor = self._load_video_model()
        # SAM3's implicit streaming ids depend on retained frame count, so keep a separate contiguous index.
        internal_frame_idx: int = 0
        self._next_internal_frame_idx = internal_frame_idx
        self.inference_session = video_processor.init_video_session(
            video=None,
            inference_device=self.device,
            inference_state_device=torch.device("cpu"),
            processing_device=torch.device("cpu"),
            video_storage_device=torch.device("cpu"),
            max_vision_features_cache_size=1,
            dtype=self.dtype,
        )
        video_processor.add_text_prompt(inference_session=self.inference_session, text=BOX_SEED_PROMPT)
        self.prompt_id = int(next(iter(self.inference_session.prompts.keys())))

        inputs: Any = video_processor(images=frame_rgb, device=self.device, return_tensors="pt")
        self.inference_session.add_new_frame(inputs.pixel_values[0], frame_idx=internal_frame_idx)
        self._cache_tracker_features(video_model=video_model, frame_idx=internal_frame_idx)

        track_ids: Int[ndarray, "m"] = visible_indices.astype(np.int32, copy=False)
        seed_tensor: torch.Tensor = torch.from_numpy(visible_masks.astype(np.float32, copy=False)).to(
            self.device,
            dtype=self.dtype,
        )
        for track_id in track_ids.tolist():
            self.inference_session.obj_id_to_prompt_id[int(track_id)] = self.prompt_id
            self.inference_session.obj_id_to_score[int(track_id)] = 1.0
            self.inference_session.obj_first_frame_idx[int(track_id)] = internal_frame_idx
            self.inference_session.trk_keep_alive[int(track_id)] = getattr(video_model, "init_trk_keep_alive", 8)
        video_model._tracker_add_new_objects(
            inference_session=self.inference_session,
            frame_idx=internal_frame_idx,
            new_obj_ids=[int(track_id) for track_id in track_ids.tolist()],
            new_obj_masks=seed_tensor,
            reverse=False,
        )
        self.inference_session.max_obj_id = int(track_ids.max())
        self.inference_session.obj_id_to_tracker_score_frame_wise[internal_frame_idx].update(
            {int(track_id): 1.0 for track_id in track_ids.tolist()}
        )
        self._next_internal_frame_idx = internal_frame_idx + 1
        self._prune_session(current_frame_idx=internal_frame_idx)

        scores: Float32[ndarray, "m"] = np.ones((track_ids.shape[0],), dtype=np.float32)
        return TrackedDetections(track_ids=track_ids, bboxes=visible_bboxes, masks=visible_masks, scores=scores)

    def track(
        self,
        *,
        frame_idx: int,
        frame_rgb: UInt8[ndarray, "h w 3"],
    ) -> TrackedDetections:
        """Propagate SAM3 tracks to one streamed frame."""
        height: int = int(frame_rgb.shape[0])
        width: int = int(frame_rgb.shape[1])
        if self.inference_session is None or len(self.inference_session.obj_ids) == 0:
            return _empty_detections(height, width)

        video_model: Any
        video_processor: Any
        video_model, video_processor = self._load_video_model()
        inputs: Any = video_processor(images=frame_rgb, device=self.device, return_tensors="pt")
        internal_frame_idx: int = self._next_internal_frame_idx
        with torch.inference_mode():
            model_outputs: Any = video_model(
                inference_session=self.inference_session,
                frame_idx=internal_frame_idx,
                frame=inputs.pixel_values[0],
                reverse=False,
            )
            output_frame_idx: int = int(getattr(model_outputs, "frame_idx", internal_frame_idx))
            self._next_internal_frame_idx = output_frame_idx + 1
            processed_outputs: dict[str, Any] = video_processor.postprocess_outputs(
                self.inference_session,
                model_outputs,
                original_sizes=inputs.original_sizes,
            )

        raw_masks: Any = processed_outputs.get("masks")
        if raw_masks is None or len(raw_masks) == 0:
            self._prune_session(current_frame_idx=output_frame_idx)
            return _empty_detections(height, width)

        if isinstance(raw_masks, torch.Tensor):
            masks_np: Bool[ndarray, "n h w"] = raw_masks.detach().cpu().numpy().astype(bool, copy=False)
        else:
            masks_np = np.asarray(raw_masks, dtype=bool)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        keep_indices: Int[ndarray, "m"]
        bboxes: Float32[ndarray, "m 4"]
        masks: Bool[ndarray, "m h w"]
        keep_indices, bboxes, masks = masks_to_boxes(masks_np, min_area_px=self.config.min_mask_area_px)
        if keep_indices.shape[0] == 0:
            self._prune_session(current_frame_idx=output_frame_idx)
            return _empty_detections(height, width)

        raw_track_ids: Any = processed_outputs.get("object_ids")
        if isinstance(raw_track_ids, torch.Tensor):
            all_track_ids: Int[ndarray, "n"] = raw_track_ids.detach().cpu().numpy().astype(np.int32, copy=False).reshape(-1)
        else:
            all_track_ids = np.asarray(raw_track_ids, dtype=np.int32).reshape(-1)
        track_ids: Int[ndarray, "m"] = all_track_ids[keep_indices]

        raw_scores: Any = processed_outputs.get("scores")
        if isinstance(raw_scores, torch.Tensor):
            all_scores: Float32[ndarray, "n"] = raw_scores.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        else:
            all_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
        if all_scores.shape[0] == all_track_ids.shape[0]:
            scores: Float32[ndarray, "m"] = all_scores[keep_indices]
        else:
            scores = np.ones((track_ids.shape[0],), dtype=np.float32)

        self._prune_session(current_frame_idx=output_frame_idx)
        return TrackedDetections(track_ids=track_ids, bboxes=bboxes, masks=masks, scores=scores)

    def _cache_tracker_features(self, *, video_model: Any, frame_idx: int) -> None:
        """Cache SAM3 video features for a frame without running text detection."""
        if self.inference_session is None:
            raise RuntimeError("SAM3 inference session has not been initialized.")
        with torch.inference_mode():
            pixel_values: torch.Tensor = self.inference_session.get_frame(frame_idx).unsqueeze(0)
            vision_embeds: Any = video_model.detector_model.get_vision_features(pixel_values=pixel_values)
            vision_feats: list[torch.Tensor]
            vision_pos_embeds: list[torch.Tensor]
            vision_feats, vision_pos_embeds = video_model.get_vision_features_for_tracker(vision_embeds=vision_embeds)
            self.inference_session.cache.cache_vision_features(
                frame_idx,
                {"vision_feats": vision_feats, "vision_pos_embeds": vision_pos_embeds},
            )

    def _prune_session(self, *, current_frame_idx: int) -> None:
        """Drop old raw frames and non-conditioning tracker state for long videos."""
        if self.inference_session is None:
            return
        processed_frames: dict[int, torch.Tensor] | None = self.inference_session.processed_frames
        if processed_frames is not None:
            frame_retention_start: int = current_frame_idx - 1
            for frame_key in list(processed_frames.keys()):
                if frame_key < frame_retention_start:
                    processed_frames.pop(frame_key, None)

        retention_start: int = current_frame_idx - max(0, int(self.config.memory_retention_frames))
        for output_dict in self.inference_session.output_dict_per_obj.values():
            non_cond_outputs: dict[int, Any] = output_dict.get("non_cond_frame_outputs", {})
            for frame_key in list(non_cond_outputs.keys()):
                if frame_key < retention_start:
                    non_cond_outputs.pop(frame_key, None)

        for tracked_frames in self.inference_session.frames_tracked_per_obj.values():
            for frame_key in list(tracked_frames.keys()):
                if frame_key < retention_start:
                    tracked_frames.pop(frame_key, None)

        for frame_map_name in ("obj_id_to_tracker_score_frame_wise", "suppressed_obj_ids"):
            frame_map: dict[int, Any] = getattr(self.inference_session, frame_map_name, {})
            for frame_key in list(frame_map.keys()):
                if frame_key < retention_start:
                    frame_map.pop(frame_key, None)
