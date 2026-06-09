from __future__ import annotations

import torch

from sam2.modeling.memory import (
    ObjectMemoryBank,
    ObjectMemory,
    ObjectMemorySelection,
)
from sam2.modeling.sam2_result import SAM2Result
from sam2.modeling.sam2_prompt import SAM2Prompt

import bisect


class SAM2ObjectMemoryBank(ObjectMemoryBank):
    """
    Default implementation for the memory bank, as per SAM2 original paper.

    The memory bank from the original implementation as infinite storage (no forgetting strategy).
    """

    def __init__(
        self,
        memory_temporal_stride: int = 1,
        storage_device: torch.device = torch.device("cpu"),
    ):
        super(SAM2ObjectMemoryBank, self).__init__()
        # Create the storage for the memories
        # The key is the object ID, the value is a list of memories sorted by frame index.
        self.conditional_memories: dict[int, list[ObjectMemory]] = {}
        self.non_conditional_memories: dict[int, list[ObjectMemory]] = {}
        self.memory_temporal_stride = memory_temporal_stride
        self.storage_device = storage_device

    def count_conditional_memories(self) -> int:
        return sum(len(memories) for memories in self.conditional_memories.values())

    def count_non_conditional_memories(self) -> int:
        return sum(len(memories) for memories in self.non_conditional_memories.values())

    def count_object_conditional_memories(self, obj_id: int) -> int:
        return len(self.conditional_memories.get(obj_id, []))

    def count_object_non_conditional_memories(self, obj_id: int) -> int:
        return len(self.non_conditional_memories.get(obj_id, []))

    def try_add_memories(
        self,
        frame_idx: int,
        obj_ids: list[int],
        memory_embeddings: torch.Tensor,
        memory_pos_embeddings: torch.Tensor,
        results: SAM2Result,
        prompts: list[SAM2Prompt],
    ) -> list[tuple[bool, ObjectMemory]]:
        n_objs = len(obj_ids)
        assert len(set(obj_ids)) == len(
            obj_ids
        ), f"obj_ids must be unique, got {obj_ids}"

        assert (
            memory_embeddings.ndim == 4
        ), f"Expected memory_embeddings to be of shape (B, N, H, W), got {memory_embeddings.shape}"
        assert (
            memory_pos_embeddings.ndim == 4
        ), f"Expected memory_pos_embeddings to be of shape (B, N, H, W), got {memory_pos_embeddings.shape}"
        assert (
            memory_embeddings.shape[0] == n_objs
        ), f"Expected memory_embeddings to have batch size {n_objs}, got {memory_embeddings.shape[0]}"
        assert (
            memory_pos_embeddings.shape[0] == n_objs
        ), f"Expected memory_pos_embeddings to have batch size {n_objs}, got {memory_pos_embeddings.shape[0]}"
        assert (
            results.batch_size == n_objs
        ), f"Expected {n_objs} results, got {results.batch_size}"

        prompts_dict = {p.obj_id: p for p in prompts}
        prompts = [prompts_dict.get(obj_id, None) for obj_id in obj_ids]

        ret = []

        for i, obj_id in enumerate(obj_ids):
            memory_embedding = memory_embeddings[[i]]
            memory_pos_embedding = memory_pos_embeddings[[i]]
            result = results[i]
            prompt = prompts[i]
            is_conditional = prompt is not None

            self.known_obj_ids.add(obj_id)

            memory = ObjectMemory(
                obj_id=obj_id,
                frame_idx=frame_idx,
                memory_embeddings=memory_embedding,
                memory_pos_embeddings=memory_pos_embedding,
                ptr=result.obj_ptrs,
                is_conditional=is_conditional,
            )

            # Store the memory on the correct device.
            memory = memory.to(self.storage_device)

            if is_conditional:
                cond_obj_memories = self.conditional_memories.setdefault(obj_id, [])

                # Find where to insert the memory using binary search on the frame_index key,
                # replacing existing memory for the same frame_idx if it exists.
                pos = bisect.bisect_left(
                    [m.frame_idx for m in cond_obj_memories],
                    frame_idx,
                )

                if (
                    pos < len(cond_obj_memories)
                    and cond_obj_memories[pos].frame_idx == frame_idx
                ):
                    cond_obj_memories[pos] = memory
                else:
                    cond_obj_memories.insert(pos, memory)
            else:
                non_cond_obj_memories = self.non_conditional_memories.setdefault(
                    obj_id, []
                )
                # Find where to insert the memory using binary search on the frame_index key,
                # replacing existing memory for the same frame_idx if it exists.
                pos = bisect.bisect_left(
                    [m.frame_idx for m in non_cond_obj_memories],
                    frame_idx,
                )
                if (
                    pos < len(non_cond_obj_memories)
                    and non_cond_obj_memories[pos].frame_idx == frame_idx
                ):
                    non_cond_obj_memories[pos] = memory
                else:
                    non_cond_obj_memories.insert(pos, memory)

            ret.append((True, memory))

        return ret

    def prune_memories(
        self, obj_ids: list[int], current_frame_idx: int
    ) -> dict[int, list[ObjectMemory]]:
        # The original SAM2 implementation has no forgetting strategy, so we don't remove any memories.
        return {}

    def select_memories(
        self,
        obj_ids: list[int],
        current_frame_idx: int,
        max_conditional_memories: int,
        max_non_conditional_memories: int,
        max_ptr_memories: int,
        only_include_pointers_in_past: bool = False,
        reverse_tracking: bool = False,
    ) -> dict[int, ObjectMemorySelection]:

        assert len(set(obj_ids)) == len(
            obj_ids
        ), f"obj_ids must be unique, got {obj_ids}"

        ret = {}

        for obj_id in obj_ids:

            # 1. Select the conditional memories
            obj_conditional_memories = self.conditional_memories.get(obj_id, [])
            obj_non_conditional_memories = self.non_conditional_memories.get(obj_id, [])

            selected_obj_conditional_memories, unselected_obj_conditional_memories = (
                _select_N_closest_conditional_memories(
                    conditional_memories=obj_conditional_memories,
                    N=max_conditional_memories,
                    current_frame_idx=current_frame_idx,
                )
            )

            # 2. Select the non-conditional memories
            # If an unselected conditioning frame is among the last frames, we still attend to it as if it's a non-conditioning frame.

            selected_obj_non_conditional_memories = obj_non_conditional_memories
            selected_obj_non_conditional_memories.extend(
                unselected_obj_conditional_memories
            )
            selected_obj_non_conditional_memories = sorted(
                selected_obj_non_conditional_memories, key=lambda x: x.frame_idx
            )

            selected_obj_non_conditional_memories = (
                _select_N_last_non_conditional_memories(
                    non_conditional_memories=selected_obj_non_conditional_memories,
                    N=max_non_conditional_memories,
                    current_frame_idx=current_frame_idx,
                    reverse_tracking=reverse_tracking,
                    temporal_stride=self.memory_temporal_stride,
                )
            )

            # 3. Select the object pointer memories

            # First add those object pointers from selected conditioning frames
            # (optionally, only include object pointers in the past during evaluation)
            selected_obj_ptrs_memories = [
                selected_obj_conditional_memory
                for selected_obj_conditional_memory in selected_obj_conditional_memories
                if selected_obj_conditional_memory.frame_idx < current_frame_idx
                or not only_include_pointers_in_past
            ]

            # Add up to (max_object_memories - 1) non-conditioning frames before current frame
            selected_obj_ptrs_memories.extend(
                _select_N_last_non_conditional_memories(
                    non_conditional_memories=selected_obj_non_conditional_memories,
                    N=max_ptr_memories,
                    current_frame_idx=current_frame_idx,
                    reverse_tracking=reverse_tracking,
                    temporal_stride=self.memory_temporal_stride,
                )
            )

            ret[obj_id] = ObjectMemorySelection(
                conditional_memories=selected_obj_conditional_memories,
                non_conditional_memories=selected_obj_non_conditional_memories,
                ptr_memories=selected_obj_ptrs_memories,
            )

        return ret

    def clear_all_conditional_memories(self) -> list[ObjectMemory]:
        removed_memories = []
        for obj_id in self.conditional_memories:
            removed_memories.extend(self.conditional_memories[obj_id])
            self.conditional_memories[obj_id] = []
        return removed_memories

    def clear_all_non_conditional_memories(self) -> list[ObjectMemory]:
        removed_memories = []
        for obj_id in self.non_conditional_memories:
            removed_memories.extend(self.non_conditional_memories[obj_id])
            self.non_conditional_memories[obj_id] = []
        return removed_memories

    def clear_conditional_memories_in_frame(self, frame_idx: int) -> list[ObjectMemory]:
        removed_memories = []
        for obj_id in self.conditional_memories:
            kept_memories = [
                m for m in self.conditional_memories[obj_id] if m.frame_idx != frame_idx
            ]
            removed_memories.extend(
                [
                    m
                    for m in self.conditional_memories[obj_id]
                    if m.frame_idx == frame_idx
                ]
            )
            self.conditional_memories[obj_id] = kept_memories
        return removed_memories

    def clear_non_conditional_memories_in_frame(self, frame_idx: int) -> list[ObjectMemory]:
        removed_memories = []
        for obj_id in self.non_conditional_memories:
            kept_memories = [m for m in self.non_conditional_memories[obj_id] if m.frame_idx != frame_idx]
            removed_memories.extend([m for m in self.non_conditional_memories[obj_id] if m.frame_idx == frame_idx])
            self.non_conditional_memories[obj_id] = kept_memories
        return removed_memories

    def clear_object_conditional_memories_in_frame(self, obj_id: int, frame_idx: int) -> list[ObjectMemory]:
        if obj_id not in self.conditional_memories:
            return []
        kept_memories = [m for m in self.conditional_memories[obj_id] if m.frame_idx != frame_idx]
        removed_memories = [m for m in self.conditional_memories[obj_id] if m.frame_idx == frame_idx]
        self.conditional_memories[obj_id] = kept_memories
        return removed_memories
    
    def clear_object_non_conditional_memories_in_frame(self, obj_id: int, frame_idx: int) -> list[ObjectMemory]:
        if obj_id not in self.non_conditional_memories:
            return []
        kept_memories = [m for m in self.non_conditional_memories[obj_id] if m.frame_idx != frame_idx]
        removed_memories = [m for m in self.non_conditional_memories[obj_id] if m.frame_idx == frame_idx]
        self.non_conditional_memories[obj_id] = kept_memories
        return removed_memories


def _select_N_closest_conditional_memories(
    conditional_memories: list[ObjectMemory],
    N: int,
    current_frame_idx: int,
) -> tuple[list[ObjectMemory], list[ObjectMemory]]:
    """
    Select up to `N` conditioning frames from `conditional_memories`
    that are temporally closest to the current frame at `current_frame_idx`. Here, we take
        - a) the closest conditioning frame before `current_frame_idx` (if any);
        - b) the closest conditioning frame after `current_frame_idx` (if any);
        - c) any other temporally closest conditioning frames until reaching a total of `N` conditioning frames.

    Returns:
        - selected_memories: the selected memories.
        - unselected_memories: the memories that were not selected.
    """

    # No limit on the number of conditional memories, return all of them.
    if N == -1:
        return conditional_memories, []

    frame_indices = [m.frame_idx for m in conditional_memories]

    last_idx_before = bisect.bisect_right(frame_indices, current_frame_idx - 1)
    first_idx_after = bisect.bisect_left(frame_indices, current_frame_idx + 1)

    selected_outputs = []

    # Add the closest conditioning frame before `current_frame_idx` (if any)
    if last_idx_before > 0:
        selected_outputs.append(conditional_memories[last_idx_before - 1])
    # Add the closest conditioning frame after `current_frame_idx` (if any)
    if first_idx_after < len(conditional_memories):
        selected_outputs.append(conditional_memories[first_idx_after])

    n_remaining = N - len(selected_outputs)

    # Add other temporally closest conditioning frames until reaching a total of `N` conditioning frames.
    remaining_indices = sorted(
        (t for t in conditional_memories if t not in selected_outputs),
        key=lambda x: abs(x.frame_idx - current_frame_idx),
    )
    selected_outputs.extend(conditional_memories[remaining_indices[:n_remaining]])
    unselected_outputs = conditional_memories[remaining_indices[n_remaining:]]

    return selected_outputs, unselected_outputs


def _select_N_last_non_conditional_memories(
    non_conditional_memories: list[ObjectMemory],
    N: int,
    current_frame_idx: int,
    reverse_tracking: bool,
    temporal_stride: int,
) -> list[ObjectMemory]:
    """
    Select up to `N` non-conditional memories from `non_conditional_memories`.
    If `reverse_tracking` is True, we select the memories after `current_frame_idx`. Otherwise, we select the memories before `current_frame_idx`.
    In case one of the last `N` frames is missing, we return an `EmptyObjectMemory` for that frame so that the model
    will encode a memory with a NO_OBJ_SCORE filled mask and a dummy object pointer.

    Args:
        non_conditional_memories: a list of non-conditional memories, assumed to be sorted by frame_idx.
        N: the maximum number of memories to select.
        current_frame_idx: the current frame index.
        reverse_tracking: whether we are in reverse tracking mode.
        temporal_stride: the temporal stride of the memory bank.
    """

    tpos_sign = 1 if reverse_tracking else -1
    frame_indices = [m.frame_idx for m in non_conditional_memories]

    selected_memories = []

    for i in range(N):

        if i == 0:
            frame_idx = current_frame_idx + tpos_sign
        else:
            frame_idx = current_frame_idx + tpos_sign * (i + 1) * temporal_stride

        pos = bisect.bisect_left(frame_indices, frame_idx)

        # If we don't have a memory for this frame, return an empty memory.
        # This is to be consistent with the original SAM2 implementation,
        # which encodes a memory corresponding to a NO_OBJ_SCORE filled mask and a dummy object pointer.
        if pos < len(non_conditional_memories) and frame_indices[pos] == frame_idx:
            selected_memories.append(non_conditional_memories[pos])

    return selected_memories
