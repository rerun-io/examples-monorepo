"""sam2-streaming ``forward_batched``: the per-object forward's results, with every slot and object in one pass."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sam2", reason="the streaming SAM 2 fork is in the posekit envs only")


def test_pointer_frames_per_batch_element_match_separate_calls() -> None:
    from sam2.modeling.sam2_generic import SAM2Generic

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
    frames: list[list[int]] = [[7, 2], [9, 5], [11, 8]]
    batched = SAM2Generic._prepare_obj_ptrs_for_memory_conditioning(fake, 12, ptrs, frames)  # pyrefly: ignore[bad-argument-type]
    for element in range(2):
        alone = SAM2Generic._prepare_obj_ptrs_for_memory_conditioning(fake, 12, ptrs[:, element : element + 1], [row[element] for row in frames])  # pyrefly: ignore[bad-argument-type]
        for got, want in zip(batched, alone, strict=True):
            torch.testing.assert_close(got[:, element : element + 1], want)


def _scene(step: int) -> torch.Tensor:
    """Two lenses, 256 px: a bright square moving right on lens 0, a disc moving down on lens 1."""
    frames: torch.Tensor = torch.full((2, 256, 256, 3), 40, dtype=torch.uint8)
    frames[0, 80:150, 30 + 4 * step : 100 + 4 * step] = torch.tensor([220, 60, 60], dtype=torch.uint8)
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    frames[1][(yy - 60 - 5 * step) ** 2 + (xx - 128) ** 2 < 35**2] = torch.tensor([60, 200, 90], dtype=torch.uint8)
    frames[1, 180:230, 20:70] = torch.tensor([200, 200, 40], dtype=torch.uint8)
    return frames


@pytest.mark.integration
def test_forward_batched_matches_per_slot_forward_in_fp32() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (EfficientTAM-ti from the Hugging Face hub)")
    from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
    from sam2.modeling.sam2_prompt import SAM2Prompt
    from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

    from posekit.models.sam2_video import cached_predictor

    predictor = cached_predictor("efficienttam-ti-512")
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
        frames: torch.Tensor = _scene(step).to(device).permute(0, 3, 1, 2)
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
