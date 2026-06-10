"""Batched steady-state propagation across cameras for the SAM2 fork.

The fork's ``forward_embeddings`` loops per object: memory conditioning, mask
decoding, and memory encoding each run at B=1 per camera — pure Python +
launch overhead that neither torch.compile nor threads removed (GIL). But all
of those modules are batch-native, and synchronized cameras advance in
lockstep so their selected memories share identical frame-index structure:
stacking each memory slot across cameras yields one B=n_cams pass that is
numerically identical to the per-camera loop.

Preconditions (else return None and the caller falls back to the fork path):
exactly one tracked object (id 0) per camera, no prompts this tick, and
aligned memory selections.
"""

from __future__ import annotations

from typing import Any

import torch


def batched_propagate(
    predictor: Any,
    states: list[Any],
    frame_idx: int,
    img_embeddings: list[torch.Tensor],
    img_pos_embeddings: list[torch.Tensor],
    video_hw: tuple[int, int],
) -> Any | None:
    """One B=n_cams propagation step; returns a batched ``SAM2Result`` or None.

    ``img_embeddings``/``img_pos_embeddings`` are the multi-level outputs of a
    single batched ``encode_image`` over all cameras (batch dim = camera).
    """
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
