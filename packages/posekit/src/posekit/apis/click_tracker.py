"""Single-object, click-prompted video tracking over the SAM2-streaming fork (Kineo's interaction model).

One :class:`ClickTracker` owns one clip, one SAM2 memory state, and one object.
Points (positive/negative) may be placed on *any* frame; each prompted frame
becomes a conditional memory. Scrubbing to an unprompted frame previews the
memory-conditioned mask without writing memory; :meth:`track` then propagates
forward and backward from the first prompted frame.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import torch
from einops import rearrange
from jaxtyping import Bool, UInt8
from torch import Tensor
from torch.nn import functional as F

OBJ_ID: int = 0
"""The single tracked object."""
T = TypeVar("T")
_INFERENCE_LOCK = threading.Lock()  # no annotation: beartype rejects the builtin lock type hint
"""Serialize calls into the shared GPU predictor, one inference step at a time."""


@dataclass(frozen=True, slots=True)
class Point:
    """One click prompt in pixel coordinates."""

    frame_idx: int
    """Zero-based frame carrying the click."""
    x: float
    """Horizontal pixel coordinate."""
    y: float
    """Vertical pixel coordinate."""
    positive: bool
    """Whether the click includes rather than excludes the pixel."""


@dataclass(frozen=True, slots=True)
class MaskResult:
    """The object's mask and model confidence on one frame."""

    frame_idx: int
    """Zero-based frame carrying the result."""
    mask: Bool[Tensor, "h w"]
    """Binary object mask on the inference device."""
    score: float
    """SAM decoder predicted IoU for the chosen hypothesis."""
    object_score: float
    """Sigmoid object-presence score."""


class ClickTracker:
    """Stateful single-object tracker driven by clicks on arbitrary frames."""

    def __init__(
        self,
        video_path: Path,
        predictor,
        *,
        device: str = "cuda",
        memory_window_size: int = 10,
        remove_radius_px: float = 100.0,
    ) -> None:
        """Open the clip and start an empty memory state.

        Args:
            video_path: Input clip; decoded on ``device`` by torchcodec.
            predictor: A ``SAM2GenericVideoPredictor`` (e.g. ``Sam2VideoSegmenter(...).predictor``).
            device: CUDA device for decode + inference.
            memory_window_size: Non-conditional memories kept around the current frame.
            remove_radius_px: How close a click must be to an existing point to remove it.

        Raises:
            ValueError: If the clip has no decodable frames.
        """
        from torchcodec.decoders import VideoDecoder

        self.predictor = predictor
        self.device: str = device
        self.remove_radius_px: float = remove_radius_px
        # torchcodec's CUDA (NVDEC) decoder is thread-affine: any call from a thread
        # other than the one that created it fails ("Failed to create NVDEC decoder:
        # 201" / "Could not receive frame from decoder"), even when serialized. Web
        # callbacks run on pool threads, so the decoder lives on its own thread and
        # every decode is executed there.
        self._decode_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="click-tracker-decode")
        self._decoder = self._decode(lambda: VideoDecoder(video_path, dimension_order="NHWC", device=device))
        total: int | None = self._decoder.metadata.num_frames
        if total is None or total <= 0:
            raise ValueError(f"{video_path} has no decodable frames.")
        self.num_frames: int = int(total)
        first = self._decode(lambda: self._decoder.get_frame_at(0))
        self.frame_hw: tuple[int, int] = (int(first.data.shape[0]), int(first.data.shape[1]))
        self.points: list[Point] = []
        """Every live point, in insertion order (the undo order)."""
        self._memory_window_size: int = memory_window_size
        self._state = self._new_state()
        self._embedding_cache: tuple[int, tuple[list[Tensor], list[Tensor]]] | None = None
        self._closed: bool = False

    def _new_state(self):
        from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

        return SAM2GenericVideoPredictorState.create(
            video_hw=self.frame_hw, memory_bank=SAM2ForgetfulObjectMemoryBank(memory_window_size=self._memory_window_size)
        )

    # ── frames ────────────────────────────────────────────────────────────

    def _decode(self, fn: Callable[[], T]) -> T:
        """Run a decoder call on the decoder's own thread and return its result."""
        return self._decode_thread.submit(fn).result()

    def close(self) -> None:
        """Stop the thread-affine decoder executor. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._decode_thread.shutdown(wait=True, cancel_futures=True)

    def frame(self, frame_idx: int) -> UInt8[Tensor, "h w 3"]:
        """Decode one frame on the inference device."""
        return self._decode(lambda: self._decoder.get_frame_at(frame_idx)).data

    def _embeddings(self, frame_idx: int) -> tuple[list[Tensor], list[Tensor]]:
        """Image-encoder output for one frame; the last frame's is cached (Kineo's one-frame cache)."""
        if self._embedding_cache is None or self._embedding_cache[0] != frame_idx:
            frame_chw: UInt8[Tensor, "3 h w"] = rearrange(self.frame(frame_idx), "h w c -> c h w").contiguous()
            with _INFERENCE_LOCK:
                embeddings: tuple[list[Tensor], list[Tensor]] = self.predictor.encode_image(frame_chw)
            self._embedding_cache = (frame_idx, embeddings)
        return self._embedding_cache[1]

    # ── points ────────────────────────────────────────────────────────────

    def points_on(self, frame_idx: int) -> list[Point]:
        """The live points placed on one frame."""
        return [p for p in self.points if p.frame_idx == frame_idx]

    def prompted_frames(self) -> list[int]:
        """Sorted frame indices that carry at least one point."""
        return sorted({p.frame_idx for p in self.points})

    def add_point(self, frame_idx: int, x: float, y: float, *, positive: bool, resegment: bool = False) -> MaskResult:
        """Add a point on a frame and return that frame's re-prompted mask."""
        self.points.append(Point(frame_idx=frame_idx, x=float(x), y=float(y), positive=positive))
        result = self.refresh(frame_idx, resegment=resegment)
        assert result is not None
        return result

    def remove_point_near(self, frame_idx: int, x: float, y: float) -> tuple[Point | None, MaskResult | None]:
        """Remove the closest point on the frame within ``remove_radius_px`` (Kineo's Shift+Click)."""
        candidates: list[Point] = self.points_on(frame_idx)
        if not candidates:
            return None, None
        nearest: Point = min(candidates, key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)
        if (nearest.x - x) ** 2 + (nearest.y - y) ** 2 > self.remove_radius_px**2:
            return None, None
        self.points.remove(nearest)
        return nearest, self.refresh(frame_idx)

    def undo(self) -> tuple[Point | None, MaskResult | None]:
        """Remove the most recently added point."""
        if not self.points:
            return None, None
        last: Point = self.points.pop()
        return last, self.refresh(last.frame_idx)

    def clear(self) -> None:
        """Drop every point and every memory."""
        self.points = []
        self._state = self._new_state()

    # ── inference ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def refresh(self, frame_idx: int, *, resegment: bool = False) -> MaskResult | None:
        """Re-run the frame with its current points, writing (or clearing) its conditional memory.

        Returns:
            The prompted mask, or ``None`` when the frame has no points left.
        """
        from sam2.modeling.sam2_prompt import SAM2Prompt

        frame_points: list[Point] = self.points_on(frame_idx)
        if not frame_points:
            self._state.memory_bank.clear_conditional_memories_in_frame(frame_idx=frame_idx)
            self._state.memory_bank.clear_non_conditional_memories_in_frame(frame_idx=frame_idx)
            return None
        self._state.memory_bank.clear_conditional_memories_in_frame(frame_idx=frame_idx)
        self._state.memory_bank.clear_non_conditional_memories_in_frame(frame_idx=frame_idx)
        user_coords: Tensor = torch.tensor([[p.x, p.y] for p in frame_points], dtype=torch.float32, device=self.device)
        user_labels: Tensor = torch.tensor([1 if p.positive else 0 for p in frame_points], dtype=torch.long, device=self.device)
        embeddings: tuple[list[Tensor], list[Tensor]] = self._embeddings(frame_idx)
        img_embeddings: list[Tensor] = embeddings[0]
        img_pos_embeddings: list[Tensor] = embeddings[1]
        has_other_prompt: bool = any(point.frame_idx != frame_idx for point in self.points)
        if resegment or not has_other_prompt:
            prompt = SAM2Prompt(obj_id=OBJ_ID, points_coords=user_coords, points_labels=user_labels)
            with _INFERENCE_LOCK:
                results = self.predictor.forward_embeddings(
                    self._state,
                    frame_idx,
                    img_embeddings,
                    img_pos_embeddings,
                    prompts=[prompt],
                    multimask_output=True,
                    create_memory=True,
                )
            return self._result(frame_idx, results[OBJ_ID])

        with _INFERENCE_LOCK:
            propagated_results = self.predictor.forward_embeddings(
                self._state,
                frame_idx,
                img_embeddings,
                img_pos_embeddings,
                prompts=[],
                multimask_output=True,
                create_memory=False,
            )
        propagated = propagated_results.get(OBJ_ID)
        if propagated is None:
            prompt = SAM2Prompt(obj_id=OBJ_ID, points_coords=user_coords, points_labels=user_labels)
            with _INFERENCE_LOCK:
                results = self.predictor.forward_embeddings(
                    self._state,
                    frame_idx,
                    img_embeddings,
                    img_pos_embeddings,
                    prompts=[prompt],
                    multimask_output=True,
                    create_memory=True,
                )
            return self._result(frame_idx, results[OBJ_ID])

        propagated_mask: Bool[Tensor, "h w"] = propagated.best_mask_logits[0, 0] > 0.0
        anchors: Tensor = self._sample_anchors(propagated_mask, frame_points)
        coords: Tensor = torch.cat((user_coords, anchors), dim=0)
        labels: Tensor = torch.cat(
            (user_labels, torch.ones(anchors.shape[0], dtype=torch.long, device=self.device)),
            dim=0,
        )
        prompt = SAM2Prompt(obj_id=OBJ_ID, points_coords=coords, points_labels=labels)
        with _INFERENCE_LOCK:
            candidates = self.predictor.forward_embeddings(
                self._state,
                frame_idx,
                img_embeddings,
                img_pos_embeddings,
                prompts=[prompt],
                multimask_output=True,
                create_memory=False,
            )[OBJ_ID]
            selected = self._select_candidate(candidates, propagated_mask, frame_points)
            is_prompt: Tensor = torch.ones(1, dtype=torch.bool, device=selected.device)
            memory_embeddings, memory_pos_embeddings = self.predictor.encode_memory(
                img_embeddings=[embedding.expand((1, -1, -1, -1)) for embedding in img_embeddings],
                masks_logits=selected.best_mask_logits,
                obj_score_logits=selected.obj_score_logits,
                is_prompt=is_prompt,
            )
            self._state.memory_bank.try_add_memories(
                frame_idx=frame_idx,
                obj_ids=[OBJ_ID],
                memory_embeddings=memory_embeddings,
                memory_pos_embeddings=memory_pos_embeddings,
                results=selected,
                prompts=[prompt],
            )
            self._state.memory_bank.prune_memories(obj_ids=[OBJ_ID], current_frame_idx=frame_idx)
        return self._result(frame_idx, selected)

    def _sample_anchors(self, mask: Bool[Tensor, "h w"], frame_points: list[Point]) -> Tensor:
        """Sample four to six separated positives from an eroded mask interior."""
        area: int = int(mask.sum())
        erosion_radius: int = max(3, min(12, round(area**0.5 / 80.0)))
        exterior: Tensor = (~mask).to(dtype=torch.float32)[None, None]
        eroded: Bool[Tensor, "h w"] = F.max_pool2d(
            exterior,
            kernel_size=2 * erosion_radius + 1,
            stride=1,
            padding=erosion_radius,
        )[0, 0] < 0.5
        coords_yx: Tensor = torch.nonzero(eroded, as_tuple=False)
        if coords_yx.shape[0] < 4:
            coords_yx = torch.nonzero(mask, as_tuple=False)
        radius: float = max(20.0, min(80.0, area**0.5 / 8.0))
        negative_points: list[Point] = [point for point in frame_points if not point.positive]
        if negative_points and coords_yx.shape[0] > 0:
            negative_xy: Tensor = torch.tensor([[point.x, point.y] for point in negative_points], dtype=torch.float32, device=self.device)
            coords_xy: Tensor = coords_yx[:, [1, 0]].to(dtype=torch.float32)
            far_from_negative: Bool[Tensor, "n"] = ((coords_xy[:, None] - negative_xy[None]) ** 2).sum(dim=2).amin(dim=1) >= radius**2
            coords_yx = coords_yx[far_from_negative]
        if coords_yx.shape[0] == 0:
            return torch.empty((0, 2), dtype=torch.float32, device=self.device)

        coords_xy = coords_yx[:, [1, 0]].to(dtype=torch.float32)
        centroid: Tensor = coords_xy.mean(dim=0)
        first_idx: int = int(((coords_xy - centroid) ** 2).sum(dim=1).argmin())
        chosen: list[int] = [first_idx]
        min_distance_sq: Tensor = ((coords_xy - coords_xy[first_idx]) ** 2).sum(dim=1)
        target_count: int = min(6, int(coords_xy.shape[0]))
        while len(chosen) < target_count:
            next_idx: int = int(min_distance_sq.argmax())
            if next_idx in chosen:
                break
            if len(chosen) >= 4 and float(min_distance_sq[next_idx]) < radius**2:
                break
            chosen.append(next_idx)
            distance_sq: Tensor = ((coords_xy - coords_xy[next_idx]) ** 2).sum(dim=1)
            min_distance_sq = torch.minimum(min_distance_sq, distance_sq)
        return coords_xy[chosen]

    def _select_candidate(self, result, propagated_mask: Bool[Tensor, "h w"], frame_points: list[Point]):
        """Choose a multimask candidate by user-click satisfaction, then prior-mask IoU."""
        from sam2.modeling.sam2_result import SAM2Result

        masks: Bool[Tensor, "n h w"] = result.masks_logits[0] > 0.0
        ranked: list[tuple[bool, int, float, int]] = []
        for candidate_idx in range(int(masks.shape[0])):
            mask: Bool[Tensor, "h w"] = masks[candidate_idx]
            satisfied: list[bool] = []
            for point in frame_points:
                x: int = max(0, min(round(point.x), int(mask.shape[1]) - 1))
                y: int = max(0, min(round(point.y), int(mask.shape[0]) - 1))
                satisfied.append(bool(mask[y, x]) == point.positive)
            intersection: int = int((mask & propagated_mask).sum())
            union: int = int((mask | propagated_mask).sum())
            overlap: float = float(intersection / union) if union else 0.0
            ranked.append((all(satisfied), sum(satisfied), overlap, candidate_idx))
        selected_idx: int = max(ranked)[-1]
        return SAM2Result(
            masks_logits=result.masks_logits[:, selected_idx : selected_idx + 1],
            ious=result.ious[:, selected_idx : selected_idx + 1],
            obj_ptrs=result.obj_ptrs,
            obj_scores_logits=result.obj_score_logits,
        )

    @torch.inference_mode()
    def preview(self, frame_idx: int) -> MaskResult | None:
        """Mask on any frame conditioned on the prompted frames only; writes no memory (Kineo's scrub preview).

        Returns:
            ``None`` when nothing has been prompted yet.
        """
        if not self.points:
            return None
        if self.points_on(frame_idx):
            return self.refresh(frame_idx)
        img_embeddings, img_pos_embeddings = self._embeddings(frame_idx)
        with _INFERENCE_LOCK:
            results = self.predictor.forward_embeddings(
                self._state, frame_idx, img_embeddings, img_pos_embeddings, prompts=[], multimask_output=True, create_memory=False
            )
        return self._result(frame_idx, results[OBJ_ID]) if OBJ_ID in results else None

    @torch.inference_mode()
    def track(self, *, chunk: int = 32) -> Iterator[MaskResult]:
        """Propagate from the first prompt to the end, then backward to frame zero.

        Non-conditional memories from a previous run are dropped first; the
        prompted frames' conditional memories are kept, so a corrective click
        anywhere re-anchors the next run.

        Yields:
            One :class:`MaskResult` per frame: forward results first, then backward results.

        Raises:
            ValueError: If no point has been placed.
        """
        prompted: list[int] = self.prompted_frames()
        if not prompted:
            raise ValueError("Place at least one point before tracking.")
        self._state.memory_bank.clear_all_non_conditional_memories()
        self._embedding_cache = None
        for start in range(prompted[0], self.num_frames, chunk):
            stop: int = min(start + chunk, self.num_frames)
            batch = self._decode(lambda a=start, b=stop: self._decoder.get_frames_in_range(a, b))
            frames_rgb: UInt8[Tensor, "c h w 3"] = batch.data.contiguous()
            for offset in range(int(frames_rgb.shape[0])):
                frame_idx: int = start + offset
                if self.points_on(frame_idx):
                    result = self.refresh(frame_idx)
                    assert result is not None
                    yield result
                    continue
                frame_chw: UInt8[Tensor, "3 h w"] = rearrange(frames_rgb[offset], "h w c -> c h w").contiguous()
                with _INFERENCE_LOCK:
                    results = self.predictor.forward(
                        self._state, frame_idx, frame_chw, prompts=[], multimask_output=True, create_memory=True
                    )
                yield self._result(frame_idx, results[OBJ_ID])

        self._state.memory_bank.clear_all_non_conditional_memories()
        self._embedding_cache = None
        for stop in range(prompted[0], 0, -chunk):
            start: int = max(0, stop - chunk)
            batch = self._decode(lambda a=start, b=stop: self._decoder.get_frames_in_range(a, b))
            frames_rgb = batch.data.contiguous()
            for offset in range(int(frames_rgb.shape[0]) - 1, -1, -1):
                frame_idx = start + offset
                frame_chw = rearrange(frames_rgb[offset], "h w c -> c h w").contiguous()
                with _INFERENCE_LOCK:
                    results = self.predictor.forward(
                        self._state,
                        frame_idx,
                        frame_chw,
                        prompts=[],
                        multimask_output=True,
                        reverse_tracking=True,
                        create_memory=True,
                    )
                yield self._result(frame_idx, results[OBJ_ID])

    def _result(self, frame_idx: int, result) -> MaskResult:
        mask_hw: Bool[Tensor, "h w"] = result.best_mask_logits[0, 0] > 0.0
        return MaskResult(
            frame_idx=frame_idx,
            mask=mask_hw,
            score=float(result.ious.max()),
            object_score=float(torch.sigmoid(result.obj_score_logits).reshape(-1)[0]),
        )


__all__ = ("ClickTracker", "MaskResult", "Point")
