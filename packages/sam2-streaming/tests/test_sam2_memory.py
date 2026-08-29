"""Regression tests for SAM2 memory selection."""

import torch

from sam2.modeling.memory import ObjectMemory
from sam2.modeling.sam2_memory import SAM2ObjectMemoryBank, _select_N_closest_conditional_memories


def _memory(frame_idx: int, *, conditional: bool) -> ObjectMemory:
    return ObjectMemory(
        obj_id=0,
        frame_idx=frame_idx,
        memory_embeddings=torch.zeros(1, 1, 1, 1),
        memory_pos_embeddings=torch.zeros(1, 1, 1, 1),
        ptr=torch.zeros(1, 1),
        is_conditional=conditional,
    )


def test_conditional_memory_selection_is_bounded_and_does_not_mutate_bank() -> None:
    conditional: list[ObjectMemory] = [_memory(frame_idx, conditional=True) for frame_idx in (0, 10, 20, 30, 40)]
    selected: list[ObjectMemory]
    unselected: list[ObjectMemory]
    selected, unselected = _select_N_closest_conditional_memories(conditional, N=3, current_frame_idx=25)
    assert [memory.frame_idx for memory in selected] == [20, 30, 10]
    assert {memory.frame_idx for memory in unselected} == {0, 40}

    bank: SAM2ObjectMemoryBank = SAM2ObjectMemoryBank()
    bank.known_obj_ids.add(0)
    bank.conditional_memories[0] = conditional
    bank.non_conditional_memories[0] = [_memory(24, conditional=False)]
    bank.select_memories(
        obj_ids=[0],
        current_frame_idx=25,
        max_conditional_memories=3,
        max_non_conditional_memories=2,
        max_ptr_memories=2,
    )
    assert [memory.frame_idx for memory in bank.non_conditional_memories[0]] == [24]
