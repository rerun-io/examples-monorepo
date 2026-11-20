import numpy as np
import torch
from dataclasses import dataclass, field

from sam2.modeling.sam2_generic import SAM2Generic
from sam2.modeling.sam2_prompt import SAM2Prompt
from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.sam2_memory import ObjectMemoryBank, SAM2ObjectMemoryBank


@dataclass
class SAM2GenericVideoPredictorState:
    video_hw: tuple[int, int]
    memory_bank: ObjectMemoryBank = field(default_factory=SAM2ObjectMemoryBank)

    def __post_init__(self):
        if self.memory_bank is None:
            raise ValueError("Memory bank cannot be None")
        if self.video_hw is None:
            raise ValueError("Video height and width cannot be None")

    @staticmethod
    def create(
        video_hw: tuple[int, int], memory_bank: ObjectMemoryBank | None = None
    ) -> "SAM2GenericVideoPredictorState":
        if memory_bank is None:
            memory_bank = SAM2ObjectMemoryBank()
        return SAM2GenericVideoPredictorState(
            video_hw=video_hw, memory_bank=memory_bank
        )


class SAM2GenericVideoPredictor(SAM2Generic):
    """
    SAM2GenericVideoPredictor provides a handy video prediction interface.

    Note: works in a forward-only manner.
    """

    @torch.inference_mode()
    def forward(
        self,
        state: SAM2GenericVideoPredictorState,
        frame_idx: int,
        frame: torch.Tensor,
        prompts: list[SAM2Prompt] = [],
        multimask_output: bool = True,
        reverse_tracking: bool = False,
        create_memory: bool = True,
    ) -> dict[int, SAM2Result]:
        assert frame.shape in [
            (1, *state.video_hw),
            (3, *state.video_hw),
        ], f"Expected frame to be of shape (C, H, W) or (1, C, H, W) with H and W equal to {state.video_hw}, got {frame.shape}"

        img_embeddings, img_pos_embeddings = self.encode_image(frame)

        return self.forward_embeddings(
            state=state,
            frame_idx=frame_idx,
            img_embeddings=img_embeddings,
            img_pos_embeddings=img_pos_embeddings,
            prompts=prompts,
            multimask_output=multimask_output,
            reverse_tracking=reverse_tracking,
            create_memory=create_memory,
        )

    @torch.inference_mode()
    def forward_embeddings(
        self,
        state: SAM2GenericVideoPredictorState,
        frame_idx: int,
        img_embeddings: list[torch.Tensor],
        img_pos_embeddings: list[torch.Tensor],
        prompts: list[SAM2Prompt] = [],
        multimask_output: bool = True,
        reverse_tracking: bool = False,
        create_memory: bool = True,
    ) -> dict[int, SAM2Result]:

        assert prompts is None or np.unique([p.obj_id for p in prompts]).size == len(
            prompts
        ), "Only one prompt per object should be provided"

        # Unique list of all objects to propagate the masks for (includes previous objects and new prompts).
        all_obj_ids = state.memory_bank.known_obj_ids | set([p.obj_id for p in prompts])
        n_objs = len(all_obj_ids)

        prompts_dicts: dict[int, SAM2Prompt] = {
            prompt.obj_id: prompt for prompt in prompts
        }

        objects_selected_memories = state.memory_bank.select_memories(
            obj_ids=state.memory_bank.known_obj_ids,
            current_frame_idx=frame_idx,
            max_conditional_memories=self.max_cond_frames_in_attn,
            max_non_conditional_memories=self.num_maskmem - 1,
            max_ptr_memories=self.max_obj_ptrs_in_encoder,
            only_include_pointers_in_past=self.only_obj_ptrs_in_the_past_for_eval,
            reverse_tracking=reverse_tracking,
        )

        results: list[SAM2Result] = []

        for obj_id in all_obj_ids:
            prompt = prompts_dicts.get(obj_id, None)
            has_prompt = prompt is not None

            if has_prompt:
                prompt = prompt.to(self.device)

                point_coords = (
                    prompt.points_coords.unsqueeze(0)
                    if prompt.points_coords is not None
                    else None
                )
                point_labels = (
                    prompt.points_labels.unsqueeze(0)
                    if prompt.points_labels is not None
                    else None
                )
                boxes = prompt.boxes.unsqueeze(0) if prompt.boxes is not None else None
                masks_logits = (
                    prompt.masks_logits.unsqueeze(0)
                    if prompt.masks_logits is not None
                    else None
                )

                prompt_embeddings = self.encode_prompts(
                    orig_hw=state.video_hw,
                    points_coords=point_coords,
                    points_labels=point_labels,
                    boxes=boxes,
                    masks_logits=masks_logits,
                )

                result = self.generate_masks(
                    orig_hw=state.video_hw,
                    img_embeddings=img_embeddings,
                    prompt_embeddings=prompt_embeddings,
                    multimask_output=multimask_output,
                )

            else:
                assert (
                    obj_id in objects_selected_memories
                ), f"Expected memory bank to have a memory for object {obj_id} but it does not."

                object_selected_memories = objects_selected_memories[obj_id]
                # Transfer the memories to the correct device
                object_selected_memories = object_selected_memories.to(self.device)

                # Edge case if we forward a frame without any memories.
                if (
                    len(object_selected_memories.conditional_memories) == 0
                    and len(object_selected_memories.non_conditional_memories) == 0
                ):
                    continue

                conditioned_img_embeddings = self.condition_image_embeddings_on_memories(
                    frame_idx=frame_idx,
                    img_embeddings=img_embeddings,
                    img_pos_embeddings=img_pos_embeddings,
                    non_conditional_memories=object_selected_memories.non_conditional_memories,
                    conditional_memories=object_selected_memories.conditional_memories,
                    ptr_memories=object_selected_memories.ptr_memories,
                )

                result = self.generate_masks(
                    orig_hw=state.video_hw,
                    img_embeddings=conditioned_img_embeddings,
                    multimask_output=True,
                )

            results.append(result)

        # Edge case if we forward a frame without any prompts or memories.
        if len(results) == 0:
            return {}

        batched_results = SAM2Result.cat(results)

        if create_memory:
            is_prompt = torch.tensor(
                [obj_id in prompts_dicts for obj_id in all_obj_ids],
                dtype=torch.bool,
                device=batched_results.device,
            )

            memory_embeddings, memory_pos_embeddings = self.encode_memory(
                img_embeddings=[m.expand((n_objs, -1, -1, -1)) for m in img_embeddings],
                masks_logits=batched_results.best_mask_logits,
                obj_score_logits=batched_results.obj_score_logits,
                is_prompt=is_prompt,
            )

            state.memory_bank.try_add_memories(
                frame_idx=frame_idx,
                obj_ids=all_obj_ids,
                memory_embeddings=memory_embeddings,
                memory_pos_embeddings=memory_pos_embeddings,
                results=batched_results,
                prompts=prompts,
            )

            state.memory_bank.prune_memories(
                obj_ids=all_obj_ids,
                current_frame_idx=frame_idx,
            )

        return {obj_id: result for obj_id, result in zip(all_obj_ids, batched_results)}

    @torch.inference_mode()
    def forward_batched(
        self,
        states: list[SAM2GenericVideoPredictorState],
        frame_idx: int,
        frames: torch.Tensor,
        prompts: list[list[SAM2Prompt]],
        multimask_output: list[bool],
    ) -> list[dict[int, SAM2Result]]:
        """``forward`` for several videos at the same timestep, each module run once per group instead of per object.

        One image-encoder call for all ``frames`` (B, C, H, W), one per video state. Prompted objects are grouped by
        prompt layout, propagated objects by memory layout (numbers of conditional, non-conditional and pointer
        memories); each group is one batch through memory attention and the mask decoder, and one memory-encoder call
        covers every object. Per object the operations are those of ``forward`` (each pointer memory keeps its own
        frame index); only batch-size-dependent kernel numerics differ. The masks carry the best hypothesis only.

        Returns:
            Per state, the results by object id.
        """
        from sam2.modeling.memory import ObjectMemory

        assert len(states) == len(prompts) == len(multimask_output) == frames.shape[0]
        assert not self.non_overlap_masks_for_mem_enc, "the memory encoder batch mixes videos"
        video_hw: tuple[int, int] = states[0].video_hw
        assert all(s.video_hw == video_hw for s in states) and tuple(frames.shape[-2:]) == video_hw
        img_embeddings, img_pos_embeddings = self.encode_image(frames)

        # (slot, obj_id, prompt or None, memory selection or None) per object, in forward's per-state order.
        items: list[tuple[int, int, SAM2Prompt | None, object]] = []
        for slot, state in enumerate(states):
            slot_prompts: dict[int, SAM2Prompt] = {p.obj_id: p for p in prompts[slot]}
            assert len(slot_prompts) == len(prompts[slot]), "Only one prompt per object should be provided"
            selected = state.memory_bank.select_memories(
                obj_ids=state.memory_bank.known_obj_ids,
                current_frame_idx=frame_idx,
                max_conditional_memories=self.max_cond_frames_in_attn,
                max_non_conditional_memories=self.num_maskmem - 1,
                max_ptr_memories=self.max_obj_ptrs_in_encoder,
                only_include_pointers_in_past=self.only_obj_ptrs_in_the_past_for_eval,
                reverse_tracking=False,
            )
            for obj_id in state.memory_bank.known_obj_ids | set(slot_prompts):
                prompt: SAM2Prompt | None = slot_prompts.get(obj_id)
                if prompt is not None:
                    items.append((slot, obj_id, prompt.to(self.device), None))
                    continue
                selection = selected[obj_id]
                if len(selection.conditional_memories) == 0 and len(selection.non_conditional_memories) == 0:
                    continue
                items.append((slot, obj_id, None, selection.to(self.device)))

        groups: dict[tuple, list[int]] = {}
        for index, (slot, _, prompt, selection) in enumerate(items):
            if prompt is not None:
                key = (
                    "prompt",
                    multimask_output[slot],
                    None if prompt.points_coords is None else tuple(prompt.points_coords.shape),
                    None if prompt.boxes is None else tuple(prompt.boxes.shape),
                    None if prompt.masks_logits is None else tuple(prompt.masks_logits.shape),
                )
            else:
                key = ("memory", len(selection.conditional_memories), len(selection.non_conditional_memories), len(selection.ptr_memories))
            groups.setdefault(key, []).append(index)

        results: list[SAM2Result | None] = [None] * len(items)
        for key, members in groups.items():
            slots = torch.tensor([items[i][0] for i in members], device=self.device)
            group_img = [level.index_select(0, slots) for level in img_embeddings]
            if key[0] == "prompt":
                group_prompts: list[SAM2Prompt] = [items[i][2] for i in members]
                # One layout per group, so each field is None for all of them or stacks to (B, ...).
                fields: dict[str, torch.Tensor | None] = {
                    name: None if getattr(group_prompts[0], name) is None else torch.stack([getattr(p, name) for p in group_prompts])
                    for name in ("points_coords", "points_labels", "boxes", "masks_logits")
                }
                prompt_embeddings = self.encode_prompts(orig_hw=video_hw, batch_size=len(members), **fields)
                result = self.generate_masks(
                    orig_hw=video_hw, img_embeddings=group_img, prompt_embeddings=prompt_embeddings, multimask_output=key[1], best_only=True
                )
            else:
                selections = [items[i][3] for i in members]

                def stack_memories(per_object: list[ObjectMemory]) -> ObjectMemory:
                    # frame_idx is a list: one pointer frame per batch element (see _prepare_obj_ptrs_for_memory_conditioning).
                    return ObjectMemory(
                        obj_id=-1,
                        frame_idx=[m.frame_idx for m in per_object],
                        memory_embeddings=torch.cat([m.memory_embeddings for m in per_object]),
                        memory_pos_embeddings=torch.cat([m.memory_pos_embeddings for m in per_object]),
                        ptr=torch.cat([m.ptr for m in per_object]),
                    )

                conditioned = self.condition_image_embeddings_on_memories(
                    frame_idx=frame_idx,
                    img_embeddings=group_img,
                    img_pos_embeddings=[level.index_select(0, slots) for level in img_pos_embeddings],
                    conditional_memories=[stack_memories([s.conditional_memories[k] for s in selections]) for k in range(key[1])],
                    non_conditional_memories=[stack_memories([s.non_conditional_memories[k] for s in selections]) for k in range(key[2])],
                    ptr_memories=[stack_memories([s.ptr_memories[k] for s in selections]) for k in range(key[3])],
                )
                sparse, dense = self.empty_prompt_embeddings
                result = self.generate_masks(
                    orig_hw=video_hw,
                    img_embeddings=conditioned,
                    prompt_embeddings=(sparse.to(self.device).expand(len(members), -1, -1), dense.to(self.device).expand(len(members), -1, -1, -1)),
                    multimask_output=True,
                    best_only=True,
                )
            for row, index in enumerate(members):
                results[index] = result[row]

        outputs: list[dict[int, SAM2Result]] = [{} for _ in states]
        if not items:
            return outputs
        all_results = SAM2Result.cat(results)
        slots = torch.tensor([item[0] for item in items], device=self.device)
        memory_embeddings, memory_pos_embeddings = self.encode_memory(
            img_embeddings=[img_embeddings[-1].index_select(0, slots)],
            masks_logits=all_results.masks_logits,
            obj_score_logits=all_results.obj_score_logits,
            is_prompt=torch.tensor([item[2] is not None for item in items], dtype=torch.bool, device=self.device),
        )
        for slot, state in enumerate(states):
            rows = [i for i, item in enumerate(items) if item[0] == slot]
            if not rows:
                continue
            obj_ids = [items[i][1] for i in rows]
            index = torch.tensor(rows, device=self.device)
            state.memory_bank.try_add_memories(
                frame_idx=frame_idx,
                obj_ids=obj_ids,
                memory_embeddings=memory_embeddings.index_select(0, index),
                memory_pos_embeddings=memory_pos_embeddings.index_select(0, index),
                results=SAM2Result.cat([results[i] for i in rows]),
                prompts=prompts[slot],
            )
            state.memory_bank.prune_memories(obj_ids=obj_ids, current_frame_idx=frame_idx)
            outputs[slot] = {items[i][1]: results[i] for i in rows}
        return outputs
