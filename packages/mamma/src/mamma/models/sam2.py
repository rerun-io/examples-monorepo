"""SAM2-family streaming mask tracker over synchronized views (the vendored ``sam2`` fork).

One predictor (one set of weights) serves every view; the per-view memory states belong to the caller (the ``track`` node),
so the model holds no state across ticks. Each tick batches ``encode_image`` over all views; with no prompts and one
object per view the memory/decoder step also runs as one B=views batch (``batched_propagate``), else per view through the
fork's ``forward_embeddings``.

Batched propagation: the fork's ``forward_embeddings`` loops per object at B=1 per view (pure Python + launch overhead), but
the modules are batch-native and synchronized views advance in lockstep, so their selected memories share one frame-index
structure: stacking each memory slot across views gives one pass numerically identical to the per-view loop. Its
preconditions (else it returns None and the caller falls back): exactly one object (id 0) per view, no prompts this tick,
aligned memory selections.
"""

from pathlib import Path
from typing import Any, cast

import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from torch import Tensor

ViewMasks = list[dict[int, tuple[Bool[Tensor, "h w"], Float32[Tensor, " 1"]]]]
"""Per view: object id -> (mask, predicted IoU)."""


class Sam2Masks:
    def __init__(self, config: str, checkpoint: Path, device: str = "cuda", memory_window_size: int = 7) -> None:
        from sam2.build_sam import build_sam2_generic_video_predictor

        self.predictor = build_sam2_generic_video_predictor(config, str(checkpoint), device=device)
        self.device, self.memory_window_size = device, memory_window_size

    def new_states(self, views: int, video_hw: tuple[int, int]) -> list[Any]:
        """Fresh per-view memory (a bounded window of non-conditional memories per object)."""
        from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
        from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

        def bank() -> SAM2ForgetfulObjectMemoryBank:
            return SAM2ForgetfulObjectMemoryBank(memory_temporal_stride=1, memory_window_size=self.memory_window_size, storage_device=torch.device(self.device))

        return [SAM2GenericVideoPredictorState.create(video_hw=video_hw, memory_bank=bank()) for _ in range(views)]

    def __call__(self, states: list[Any], frame_index: int, frames: UInt8[Tensor, "v 3 h w"], prompts: list[dict[int, Float32[ndarray, "4"]]]) -> ViewMasks:
        """Masks of every known (or newly prompted) object in every view; ``prompts`` = per view, object id -> box xyxy."""
        from sam2.modeling.sam2_prompt import SAM2Prompt

        video_hw: tuple[int, int] = (int(frames.shape[2]), int(frames.shape[3]))
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)  # one context per tick: entering one per view costs CPU at this rate
        with torch.inference_mode(), autocast:
            encoded = cast("tuple[list[Tensor], list[Tensor]]", self.predictor.encode_image(frames))  # the fork returns per-level lists
            if not any(prompts):
                batched = batched_propagate(self.predictor, states, frame_index, encoded[0], encoded[1], video_hw)
                if batched is not None:
                    return [{0: (batched.masks_logits[v, 0] > 0.0, batched.ious[v, 0:1].float())} for v in range(len(states))]
        results: ViewMasks = []
        for v, state in enumerate(states):
            boxes: list[SAM2Prompt] = [SAM2Prompt(obj_id=obj, boxes=torch.as_tensor(box, device=self.device).reshape(1, 4)) for obj, box in prompts[v].items()]
            if not boxes and not state.memory_bank.known_obj_ids:
                results.append({})
                continue
            with torch.inference_mode(), autocast:
                # Single-mask output: with multimask the IoU argmax sometimes picks the whole-scene candidate while propagating.
                raw: dict = self.predictor.forward_embeddings(
                    state, frame_index, [level[v : v + 1] for level in encoded[0]], [level[v : v + 1] for level in encoded[1]], prompts=boxes, multimask_output=False
                )
            results.append({obj: (result.masks_logits[0, 0] > 0.0, result.ious[0, 0:1].float()) for obj, result in raw.items()})
        return results


def batched_propagate(
    predictor: Any, states: list[Any], frame_idx: int, img_embeddings: list[Tensor], img_pos_embeddings: list[Tensor], video_hw: tuple[int, int]
) -> Any | None:
    """One B=views propagation step over the outputs of one batched ``encode_image``; a batched ``SAM2Result`` or None."""
    from sam2.modeling.memory import ObjectMemory

    n_cams: int = len(states)
    selections: list[Any] = []
    for state in states:
        known = state.memory_bank.known_obj_ids
        if known != {0}:
            return None
        selection = state.memory_bank.select_memories(
            obj_ids=[0],
            current_frame_idx=frame_idx,
            max_conditional_memories=predictor.max_cond_frames_in_attn,
            max_non_conditional_memories=predictor.num_maskmem - 1,
            max_ptr_memories=predictor.max_obj_ptrs_in_encoder,
            only_include_pointers_in_past=predictor.only_obj_ptrs_in_the_past_for_eval,
            reverse_tracking=False,
        )[0]
        if len(selection.conditional_memories) == 0 and len(selection.non_conditional_memories) == 0:
            return None
        selections.append(selection)

    def signature(sel: Any) -> tuple:
        return (
            [m.frame_idx for m in sel.conditional_memories],
            [m.frame_idx for m in sel.non_conditional_memories],
            [m.frame_idx for m in sel.ptr_memories],
        )

    first_sig: tuple = signature(selections[0])
    if any(signature(sel) != first_sig for sel in selections[1:]):
        return None

    def stack_slot(per_cam: list[Any]) -> Any:
        return ObjectMemory(
            obj_id=0,
            frame_idx=per_cam[0].frame_idx,
            memory_embeddings=torch.cat([m.memory_embeddings.to(predictor.device) for m in per_cam], dim=0),
            memory_pos_embeddings=torch.cat([m.memory_pos_embeddings.to(predictor.device) for m in per_cam], dim=0),
            ptr=torch.cat([m.ptr.to(predictor.device) for m in per_cam], dim=0),
            is_conditional=per_cam[0].is_conditional,
        )

    conditional = [
        stack_slot([sel.conditional_memories[k] for sel in selections])
        for k in range(len(first_sig[0]))
    ]
    non_conditional = [
        stack_slot([sel.non_conditional_memories[k] for sel in selections])
        for k in range(len(first_sig[1]))
    ]
    ptrs = [stack_slot([sel.ptr_memories[k] for sel in selections]) for k in range(len(first_sig[2]))]

    conditioned = predictor.condition_image_embeddings_on_memories(
        frame_idx=frame_idx,
        img_embeddings=img_embeddings,
        img_pos_embeddings=img_pos_embeddings,
        conditional_memories=conditional,
        non_conditional_memories=non_conditional,
        ptr_memories=ptrs,
    )
    # The fork's propagation branch hardcodes multimask_output=True; match it
    # exactly so batched numerics equal the per-camera path. The built-in
    # empty prompt embeddings are B=1 — expand them to the camera batch.
    sparse, dense = predictor.empty_prompt_embeddings
    sparse = sparse.to(predictor.device).expand(n_cams, -1, -1)
    dense = dense.to(predictor.device).expand(n_cams, -1, -1, -1)
    result = predictor.generate_masks(
        orig_hw=video_hw,
        img_embeddings=conditioned,
        prompt_embeddings=(sparse, dense),
        multimask_output=True,
    )

    is_prompt: torch.Tensor = torch.zeros(n_cams, dtype=torch.bool, device=result.device)
    memory_embeddings, memory_pos_embeddings = predictor.encode_memory(
        img_embeddings=img_embeddings,
        masks_logits=result.best_mask_logits,
        obj_score_logits=result.obj_score_logits,
        is_prompt=is_prompt,
    )
    for cam_idx, state in enumerate(states):
        state.memory_bank.try_add_memories(
            frame_idx=frame_idx,
            obj_ids=[0],
            memory_embeddings=memory_embeddings[cam_idx : cam_idx + 1],
            memory_pos_embeddings=memory_pos_embeddings[cam_idx : cam_idx + 1],
            results=result[cam_idx],
            prompts=[],
        )
        state.memory_bank.prune_memories(obj_ids=[0], current_frame_idx=frame_idx)
    return result
