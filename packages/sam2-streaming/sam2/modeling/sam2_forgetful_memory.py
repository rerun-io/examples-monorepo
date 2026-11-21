from __future__ import annotations

import torch

from sam2.modeling.memory import ObjectMemory
from sam2.modeling.sam2_memory import SAM2ObjectMemoryBank


class SAM2ForgetfulObjectMemoryBank(SAM2ObjectMemoryBank):
    """
    Memory bank with a forgetting strategy:
    - Non-conditional memories are pruned if their frame index is outside
      the range [current_frame_idx - memory_window_size, current_frame_idx + memory_window_size].
    - Conditional (prompted) memories are kept indefinitely with ``max_conditional_memories=-1``; bounded, the bank
      keeps per object the ``max_conditional_memories`` most recent and those inside the selector's look-back
      (``memory_window_size`` x ``memory_temporal_stride`` frames), and drops the rest.

    Bounded, pair it with the predictor's ``max_cond_frames_in_attn`` set to the same value: forward tracking attends
    to that many nearest prompted memories and uses an unselected one only as a non-prompted memory (SAM 2 takes
    those from the last ``num_maskmem - 1`` frames, inside the window), so a dropped memory is never selected again.
    A tracker re-prompted through a long video then keeps a constant per-frame cost (one CUDA graph signature) and
    constant storage.
    """

    def __init__(
        self,
        memory_temporal_stride: int = 1,
        memory_window_size: int = 7,
        storage_device: torch.device = torch.device("cpu"),
        max_conditional_memories: int = -1,
    ):
        super().__init__(
            memory_temporal_stride=memory_temporal_stride, storage_device=storage_device
        )
        self.memory_window_size = memory_window_size
        self.max_conditional_memories = max_conditional_memories

    def prune_memories(
        self, obj_ids: list[int], current_frame_idx: int
    ) -> dict[int, list[ObjectMemory]]:
        obj_ids = list(obj_ids)  # the predictor passes a set
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

            if self.max_conditional_memories < 0:
                continue
            cond_obj_memories = self.conditional_memories.get(obj_id, [])
            first_recent = len(cond_obj_memories) - self.max_conditional_memories
            # An unselected prompted memory can still fill a non-prompted slot, and the selector reaches back
            # (slots x temporal stride) frames: keep those inside window x stride.
            lookback = self.memory_window_size * self.memory_temporal_stride
            kept_cond = []
            for index, m in enumerate(cond_obj_memories):
                if index >= first_recent or m.frame_idx >= current_frame_idx - lookback:
                    kept_cond.append(m)
                else:
                    removed_memories.setdefault(obj_id, []).append(m)
            self.conditional_memories[obj_id] = kept_cond

        return removed_memories
