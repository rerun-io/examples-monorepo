"""SAM2ForgetfulObjectMemoryBank(max_conditional_memories=N): the unbounded bank's selections, with bounded storage of prompted memories."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

NUM_MASKMEM: int = 7
MAX_PTRS: int = 16


def _selected(bank, frame: int, max_conditional: int) -> tuple[list[int], list[int], list[int]]:
    selection = bank.select_memories([0], frame, max_conditional, NUM_MASKMEM - 1, MAX_PTRS)[0]
    conditional, other, pointers = (selection.conditional_memories, selection.non_conditional_memories, selection.ptr_memories)
    return [m.frame_idx for m in conditional], [m.frame_idx for m in other], [m.frame_idx for m in pointers]


@settings(max_examples=60, deadline=None)
@given(
    prompted=st.lists(st.booleans(), min_size=1, max_size=120),
    max_conditional=st.integers(min_value=1, max_value=6),
)
def test_bounded_bank_selects_what_the_forgetful_bank_selects(prompted: list[bool], max_conditional: int) -> None:
    from sam2.modeling.memory import ObjectMemory
    from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank

    prompted[0] = True
    banks = [SAM2ForgetfulObjectMemoryBank(memory_window_size=7), SAM2ForgetfulObjectMemoryBank(memory_window_size=7, max_conditional_memories=max_conditional)]
    for frame, is_prompt in enumerate(prompted):
        assert _selected(banks[0], frame, max_conditional) == _selected(banks[1], frame, max_conditional)
        for bank in banks:
            store = bank.conditional_memories if is_prompt else bank.non_conditional_memories
            store.setdefault(0, []).append(ObjectMemory(0, frame, torch.zeros(1), torch.zeros(1), torch.zeros(1), is_conditional=is_prompt))
            bank.prune_memories([0], frame)
    assert len(banks[1].conditional_memories[0]) <= max_conditional + 8
    assert len(banks[0].conditional_memories[0]) == sum(prompted)
