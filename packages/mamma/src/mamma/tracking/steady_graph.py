"""CUDA-graph replay of the steady-state batched tracker tick.

The steady tick (one object per camera, no prompts) runs ~1200 small kernels:
``encode_image`` over the camera batch, memory attention, mask decoding with
full-resolution upsampling, and memory encoding. The GPU work is ~25 ms but
the Python dispatch of those launches costs far more wall time on the Grace
cores. This module captures the whole tick as one CUDA graph and replays it.

What stays dynamic each tick:

- The frames (copied into a static uint8 buffer).
- The selected memories (copied into static slot buffers with one ``cat``
  each for embeddings, positional embeddings, and object pointers).

What is baked into a graph, and therefore forms the cache key: the number of
conditional, non-conditional, and pointer memories. The temporal positional
encodings depend only on a memory's list position, and pointer temporal
encodings are off for EfficientTAM (``add_tpos_enc_to_obj_ptrs=False``), so
frame offsets never enter the graph. Models with pointer temporal encodings
are rejected at construction.

Graph outputs live in static buffers that the next replay overwrites, so the
caller clones what it keeps (the memory bank copies its slices; the masks and
object pointers are cloned here).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool, Float32, UInt8

from mamma.tracking.batched_forward import first_hypothesis_stats, select_aligned_memories

GraphKey = tuple[int, int, int, torch.dtype]
"""(n_conditional, n_non_conditional, n_pointer, memory dtype)."""


@dataclass(slots=True)
class SteadyTickOutputs:
    """Per-tick outputs of the steady-state tracker step."""

    masks: Bool[torch.Tensor, "c h w"]
    """Hypothesis-0 binary masks (owned; safe to keep across ticks)."""
    stats: Float32[torch.Tensor, "c 7"]
    """GPU stats rows ``[bbox_h_rows, x_min, y_min, x_max, y_max, pred_iou, area_px]``."""


@dataclass(slots=True)
class _Captured:
    """One captured graph with its static inputs and outputs."""

    graph: torch.cuda.CUDAGraph
    frames: UInt8[torch.Tensor, "c 3 h w"]
    mem_emb: torch.Tensor
    """Static (n_spatial * c, C, H, W) spatial memory embeddings, slot-major."""
    mem_pos: torch.Tensor
    """Static positional embeddings, same layout as ``mem_emb``."""
    ptrs: torch.Tensor
    """Static (n_ptr * c, C) object pointers, slot-major."""
    out: dict[str, torch.Tensor]


class SteadyTickGraphs:
    """Cache of CUDA graphs for the steady batched tick, keyed by memory structure."""

    def __init__(self, predictor: Any, video_hw: tuple[int, int], max_graphs: int = 6) -> None:
        if predictor.add_tpos_enc_to_obj_ptrs:
            raise ValueError("pointer temporal encodings depend on frame offsets; CUDA graphs unsupported")
        self.predictor: Any = predictor
        self.video_hw: tuple[int, int] = video_hw
        self.max_graphs: int = max_graphs
        self._graphs: OrderedDict[GraphKey, _Captured] = OrderedDict()
        self._last_key: GraphKey | None = None
        self.replays: int = 0
        self.captures: int = 0
        self.misses: int = 0

    def _body(
        self,
        frames: UInt8[torch.Tensor, "c 3 h w"],
        mem_emb: torch.Tensor,
        mem_pos: torch.Tensor,
        ptrs: torch.Tensor,
        n_cond: int,
        n_ptr: int,
    ) -> dict[str, torch.Tensor]:
        """The steady tick, identical to the eager encode + ``batched_propagate`` path."""
        from sam2.modeling.memory import ObjectMemory

        p: Any = self.predictor
        n_cams: int = frames.shape[0]
        n_spatial: int = mem_emb.shape[0] // n_cams

        def slot(k: int, ptr: torch.Tensor | None) -> ObjectMemory:
            sl = slice(k * n_cams, (k + 1) * n_cams)
            return ObjectMemory(
                obj_id=0,
                frame_idx=0,  # unused: no pointer temporal encodings
                memory_embeddings=mem_emb[sl],
                memory_pos_embeddings=mem_pos[sl],
                ptr=ptr,  # type: ignore[arg-type]
                is_conditional=k < n_cond,
            )

        spatial: list[ObjectMemory] = [slot(k, None) for k in range(n_spatial)]
        ptr_mems: list[ObjectMemory] = [
            ObjectMemory(
                obj_id=0,
                frame_idx=0,
                memory_embeddings=None,  # type: ignore[arg-type]
                memory_pos_embeddings=None,  # type: ignore[arg-type]
                ptr=ptrs[k * n_cams : (k + 1) * n_cams],
            )
            for k in range(n_ptr)
        ]
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            emb, pos = p.encode_image(frames)
            conditioned = p.condition_image_embeddings_on_memories(
                frame_idx=0,
                img_embeddings=emb,
                img_pos_embeddings=pos,
                conditional_memories=spatial[:n_cond],
                non_conditional_memories=spatial[n_cond:],
                ptr_memories=ptr_mems,
            )
            sparse, dense = p.empty_prompt_embeddings
            result = p.generate_masks(
                orig_hw=self.video_hw,
                img_embeddings=conditioned,
                prompt_embeddings=(
                    sparse.to(p.device).expand(n_cams, -1, -1),
                    dense.to(p.device).expand(n_cams, -1, -1, -1),
                ),
                multimask_output=True,
            )
            memory_embeddings, memory_pos_embeddings = p.encode_memory(
                img_embeddings=emb,
                masks_logits=result.best_mask_logits,
                obj_score_logits=result.obj_score_logits,
                is_prompt=torch.zeros(n_cams, dtype=torch.bool, device=frames.device),
            )
            masks, stats = first_hypothesis_stats(result.masks_logits, result.ious)
        return {
            "masks": masks,
            "stats": stats,
            "memory_embeddings": memory_embeddings,
            "memory_pos_embeddings": memory_pos_embeddings,
            "obj_ptrs": result.obj_ptrs,
            "ious": result.ious,
            "obj_scores_logits": result.obj_score_logits,
        }

    def _capture(
        self, key: GraphKey, frames: list[UInt8[torch.Tensor, "3 h w"]], selections: list[Any]
    ) -> _Captured:
        """Allocate static inputs, warm up on a side stream, then capture."""
        n_cond, n_noncond, n_ptr, _ = key
        n_cams: int = len(frames)
        first = selections[0]
        ref = (first.conditional_memories + first.non_conditional_memories)[0]
        n_spatial: int = n_cond + n_noncond
        static_frames = frames[0].new_empty((n_cams, *frames[0].shape))
        mem_emb = ref.memory_embeddings.new_empty((n_spatial * n_cams, *ref.memory_embeddings.shape[1:]))
        mem_pos = ref.memory_pos_embeddings.new_empty((n_spatial * n_cams, *ref.memory_pos_embeddings.shape[1:]))
        ptr_ref: torch.Tensor = first.ptr_memories[0].ptr if n_ptr else ref.ptr
        ptrs = ptr_ref.new_empty((n_ptr * n_cams, *ptr_ref.shape[1:]))
        self._fill(static_frames, mem_emb, mem_pos, ptrs, frames, selections)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._body(static_frames, mem_emb, mem_pos, ptrs, n_cond, n_ptr)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # thread_local: the fit/decode threads keep launching while we capture.
        with torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
            out = self._body(static_frames, mem_emb, mem_pos, ptrs, n_cond, n_ptr)
        self.captures += 1
        return _Captured(graph, static_frames, mem_emb, mem_pos, ptrs, out)

    @staticmethod
    def _fill(
        static_frames: torch.Tensor,
        mem_emb: torch.Tensor,
        mem_pos: torch.Tensor,
        ptrs: torch.Tensor,
        frames: list[UInt8[torch.Tensor, "3 h w"]],
        selections: list[Any],
    ) -> None:
        """Copy this tick's frames and memories into the static input buffers."""
        torch.stack(frames, dim=0, out=static_frames)
        spatial = [
            [sel.conditional_memories[k] for sel in selections] for k in range(len(selections[0].conditional_memories))
        ] + [
            [sel.non_conditional_memories[k] for sel in selections]
            for k in range(len(selections[0].non_conditional_memories))
        ]
        torch.cat([m.memory_embeddings for per_cam in spatial for m in per_cam], dim=0, out=mem_emb)
        torch.cat([m.memory_pos_embeddings for per_cam in spatial for m in per_cam], dim=0, out=mem_pos)
        if ptrs.shape[0]:
            torch.cat(
                [sel.ptr_memories[k].ptr for k in range(len(selections[0].ptr_memories)) for sel in selections],
                dim=0,
                out=ptrs,
            )

    def step(
        self, states: list[Any], frame_idx: int, frames: list[UInt8[torch.Tensor, "3 h w"]]
    ) -> SteadyTickOutputs | None:
        """Run one steady tick from a graph and update the memory banks.

        Returns None (nothing run, banks untouched) when the tick cannot be
        batched or its memory structure has no graph yet; a structure seen on
        two consecutive ticks is captured on the second.
        """
        from sam2.modeling.sam2_result import SAM2Result

        selections: list[Any] | None = select_aligned_memories(self.predictor, states, frame_idx)
        if selections is None:
            self._last_key = None
            return None
        first = selections[0]
        spatial = first.conditional_memories + first.non_conditional_memories
        key: GraphKey = (
            len(first.conditional_memories),
            len(first.non_conditional_memories),
            len(first.ptr_memories),
            spatial[0].memory_embeddings.dtype,
        )
        captured: _Captured | None = self._graphs.get(key)
        if captured is None:
            if key != self._last_key:
                self._last_key = key
                self.misses += 1
                return None
            captured = self._capture(key, frames, selections)
            self._graphs[key] = captured
            if len(self._graphs) > self.max_graphs:
                self._graphs.popitem(last=False)
        else:
            self._graphs.move_to_end(key)
            self._fill(captured.frames, captured.mem_emb, captured.mem_pos, captured.ptrs, frames, selections)
        self._last_key = key
        captured.graph.replay()
        self.replays += 1

        out = captured.out
        # Only obj_ptrs of a result is stored by the bank (as a view), so clone
        # it; the bank's memory_embeddings[[i]] indexing already copies.
        result = SAM2Result(
            masks_logits=out["ious"][:, :, None, None],
            ious=out["ious"],
            obj_ptrs=out["obj_ptrs"].clone(),
            obj_scores_logits=out["obj_scores_logits"],
        )
        for cam_idx, state in enumerate(states):
            state.memory_bank.try_add_memories(
                frame_idx=frame_idx,
                obj_ids=[0],
                memory_embeddings=out["memory_embeddings"][cam_idx : cam_idx + 1],
                memory_pos_embeddings=out["memory_pos_embeddings"][cam_idx : cam_idx + 1],
                results=result[cam_idx],
                prompts=[],
            )
            state.memory_bank.prune_memories(obj_ids=[0], current_frame_idx=frame_idx)
        return SteadyTickOutputs(masks=out["masks"].clone(), stats=out["stats"])
