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
