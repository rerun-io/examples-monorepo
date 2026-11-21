"""``forward_batched``: the per-object forward's results, with every slot and object in one pass."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from sam2.modeling.sam2_generic import SAM2Generic


def test_pointer_frames_per_batch_element_match_separate_calls() -> None:
    torch.manual_seed(0)
    fake = SimpleNamespace(
        add_tpos_enc_to_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        max_obj_ptrs_in_encoder=16,
        proj_tpos_enc_in_obj_ptrs=True,
        hidden_dim=256,
        mem_dim=64,
        obj_ptr_tpos_proj=torch.nn.Linear(256, 64),
    )
    ptrs: torch.Tensor = torch.randn(3, 2, 256)
    frames: torch.Tensor = torch.tensor([[7, 2], [9, 5], [11, 8]])
    batched = SAM2Generic._prepare_obj_ptrs_for_memory_conditioning(fake, 12, ptrs, frames)  # pyrefly: ignore[bad-argument-type]
    for element in range(2):
        alone = SAM2Generic._prepare_obj_ptrs_for_memory_conditioning(fake, 12, ptrs[:, element : element + 1], frames[:, element : element + 1])  # pyrefly: ignore[bad-argument-type]
        for got, want in zip(batched, alone, strict=True):
            torch.testing.assert_close(got[:, element : element + 1], want)
    # (n_ptr, 1): one frame per pointer for the whole batch, the ObjectMemory path's layout.
    shared = SAM2Generic._prepare_obj_ptrs_for_memory_conditioning(fake, 12, ptrs, frames[:, :1])  # pyrefly: ignore[bad-argument-type]
    first = SAM2Generic._prepare_obj_ptrs_for_memory_conditioning(fake, 12, ptrs, frames[:, :1].expand(-1, 2))  # pyrefly: ignore[bad-argument-type]
    for got, want in zip(shared, first, strict=True):
        torch.testing.assert_close(got, want)


@pytest.mark.integration
def test_forward_batched_matches_per_slot_forward_in_fp32(efficienttam_ti, scene: Callable[[int], torch.Tensor]) -> None:
    from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
    from sam2.modeling.sam2_prompt import SAM2Prompt
    from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

    predictor = efficienttam_ti
    device = torch.device("cuda")

    def states() -> list[SAM2GenericVideoPredictorState]:
        return [SAM2GenericVideoPredictorState.create((256, 256), SAM2ForgetfulObjectMemoryBank(storage_device=device)) for _ in range(2)]

    def box(obj_id: int, xyxy: list[float]) -> SAM2Prompt:
        return SAM2Prompt(obj_id=obj_id, boxes=torch.tensor([xyxy], device=device))

    # Objects started on different frames, so propagated objects carry different memory layouts and pointer frames.
    prompts_at: dict[int, list[list[SAM2Prompt]]] = {
        0: [[box(0, [30, 80, 100, 150])], [box(0, [93, 25, 163, 95])]],
        2: [[], [box(1, [20, 180, 70, 230])]],
        4: [[box(0, [46, 80, 116, 150])], []],
    }
    per_slot, batched = states(), states()
    for step in range(8):
        frames: torch.Tensor = scene(step).to(device).permute(0, 3, 1, 2)
        prompts: list[list[SAM2Prompt]] = prompts_at.get(step, [[], []])
        want = [predictor.forward(per_slot[s], step, frames[s].contiguous(), prompts=prompts[s], multimask_output=False) for s in range(2)]
        got = predictor.forward_batched(batched, step, frames, prompts, [False, False])
        for slot in range(2):
            assert list(got[slot]) == list(want[slot])
            for obj_id, result in got[slot].items():
                mask_want: torch.Tensor = want[slot][obj_id].best_mask_logits > 0
                mask_got: torch.Tensor = result.masks_logits > 0
                iou: float = float((mask_want & mask_got).sum() / (mask_want | mask_got).sum().clamp_min(1))
                assert iou > 0.999, (step, slot, obj_id, iou)
                assert mask_got.any()


@pytest.mark.integration
def test_all_hypotheses_keep_the_best_one_and_the_same_memories(efficienttam_ti, scene: Callable[[int], torch.Tensor]) -> None:
    """``best_only=False`` returns the decoder's three propagated hypotheses; the best of them is the ``best_only`` mask."""
    from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
    from sam2.modeling.sam2_prompt import SAM2Prompt
    from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

    predictor = efficienttam_ti
    device = torch.device("cuda")

    def states() -> list[SAM2GenericVideoPredictorState]:
        return [SAM2GenericVideoPredictorState.create((256, 256), SAM2ForgetfulObjectMemoryBank(storage_device=device)) for _ in range(2)]

    first: list[list[SAM2Prompt]] = [
        [SAM2Prompt(obj_id=0, boxes=torch.tensor([[30.0, 80.0, 100.0, 150.0]], device=device))],
        [SAM2Prompt(obj_id=0, boxes=torch.tensor([[93.0, 25.0, 163.0, 95.0]], device=device))],
    ]
    best, every = states(), states()
    for step in range(6):
        frames: torch.Tensor = scene(step).to(device).permute(0, 3, 1, 2)
        prompts: list[list[SAM2Prompt]] = first if step == 0 else [[], []]
        want = predictor.forward_batched(best, step, frames, prompts, [False, False])
        got = predictor.forward_batched(every, step, frames, prompts, [False, False], best_only=False)
        for slot in range(2):
            result = got[slot][0]
            assert result.masks_logits.shape[1] == (1 if step == 0 else 3)
            torch.testing.assert_close(result.best_mask_logits, want[slot][0].masks_logits, rtol=0, atol=0)
            torch.testing.assert_close(result.obj_ptrs, want[slot][0].obj_ptrs, rtol=0, atol=0)
