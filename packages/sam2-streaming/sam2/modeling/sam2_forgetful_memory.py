from __future__ import annotations

import torch

from sam2.modeling.memory import ObjectMemory
from sam2.modeling.sam2_memory import SAM2ObjectMemoryBank


class SAM2ForgetfulObjectMemoryBank(SAM2ObjectMemoryBank):
    """
    Memory bank with a forgetting strategy:
    - Conditional memories are kept indefinitely.
    - Non-conditional memories are pruned if their frame index is outside
      the range [current_frame_idx - memory_window_size, current_frame_idx + memory_window_size].
    """

    def __init__(
        self,
        memory_temporal_stride: int = 1,
        memory_window_size: int = 7,
        storage_device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            memory_temporal_stride=memory_temporal_stride, storage_device=storage_device
        )
        self.memory_window_size = memory_window_size

    def prune_memories(
        self, obj_ids: list[int], current_frame_idx: int
    ) -> dict[int, list[ObjectMemory]]:
        removed_memories: dict[int, list[ObjectMemory]] = {}

        for obj_id in obj_ids:
            non_cond_obj_memories = self.non_conditional_memories.get(obj_id, [])
            kept, removed = [], []

            for m in non_cond_obj_memories:
                if (
                    current_frame_idx - self.memory_window_size
                    <= m.frame_idx
                    <= current_frame_idx + self.memory_window_size
                ):
                    kept.append(m)
                else:
                    removed.append(m)

            if removed:
                removed_memories[obj_id] = removed

            self.non_conditional_memories[obj_id] = kept

        return removed_memories
