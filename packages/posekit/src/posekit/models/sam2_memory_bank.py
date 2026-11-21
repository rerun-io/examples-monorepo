"""A SAM 2 memory bank that also forgets prompted memories, for trackers re-prompted through long videos."""

from collections.abc import Iterable

import torch
from sam2.modeling.memory import ObjectMemory
from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank

HOST: torch.device = torch.device("cpu")


class BoundedForgetfulMemoryBank(SAM2ForgetfulObjectMemoryBank):
    """The forgetful bank, plus a bound on prompted memories: per object it keeps the ``max_conditional_memories`` most
    recent and those inside the window the bank keeps non-prompted memories for, and drops the rest.

    Forward tracking never selects a dropped one again: a frame attends to its ``max_conditional_memories`` nearest
    prompted memories (``max_cond_frames_in_attn``, set to the same value), and an unselected prompted memory serves
    only as a non-prompted one, which SAM 2 takes from the last ``num_maskmem - 1`` frames, inside the window. So the
    masks stay the same while storage stops growing with the number of prompts.
    """

    def __init__(self, max_conditional_memories: int, memory_window_size: int = 7, storage_device: torch.device = HOST) -> None:
        """Args:
        max_conditional_memories: Most recent prompted memories kept per object (the predictor's ``max_cond_frames_in_attn``).
        memory_window_size: Frames back that memories are kept regardless (at least ``num_maskmem - 1``).
        storage_device: Where the memories are kept (host by default, the inference device saves the per-step copies).
        """
        super().__init__(memory_window_size=memory_window_size, storage_device=storage_device)
        self.max_conditional_memories: int = max_conditional_memories

    def prune_memories(self, obj_ids: Iterable[int], current_frame_idx: int) -> dict[int, list[ObjectMemory]]:
        """Prune non-prompted memories as the forgetful bank does, then prompted memories past the bound (the predictor
        passes ``obj_ids`` as a set)."""
        obj_ids = list(obj_ids)
        removed: dict[int, list[ObjectMemory]] = super().prune_memories(obj_ids, current_frame_idx)
        for obj_id in obj_ids:
            memories: list[ObjectMemory] = self.conditional_memories.get(obj_id, [])
            first_recent: int = len(memories) - self.max_conditional_memories
            kept: list[ObjectMemory] = []
            for index, memory in enumerate(memories):
                if index >= first_recent or memory.frame_idx >= current_frame_idx - self.memory_window_size:
                    kept.append(memory)
                else:
                    removed.setdefault(obj_id, []).append(memory)
            self.conditional_memories[obj_id] = kept
        return removed
