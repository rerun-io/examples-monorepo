"""``GraphCache``: replayed CUDA graphs give the eager results, bit for bit."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

pytestmark = pytest.mark.integration


def _needs_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")


def test_graph_cache_replays_match_eager_per_signature() -> None:
    _needs_gpu()
    from sam2.cuda_graphs import GraphCache

    def fn(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        y = torch.relu(x @ w)
        return y, [y.sum(dim=0), x.float() * 2]

    cache = GraphCache()
    torch.manual_seed(0)
    w = torch.randn(16, 8, device="cuda")
    for rows in (4, 9, 4, 9, 4):
        # A transposed (non-contiguous) input: the static buffer keeps its strides.
        x = torch.randn(16, rows, device="cuda").t()
        got = cache(("fn",), fn, [x, w])
        want = fn(x, w)
        other = cache(("fn",), fn, [torch.randn(16, rows, device="cuda").t(), w])  # a later replay must not overwrite `got`
        assert other is not None
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert (cache.eager, cache.captures) == (2, 2)
    assert cache.replays == 8


@pytest.mark.parametrize("one_graph", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_forward_batched_with_graphs_is_bit_identical(bf16: bool, one_graph: bool, efficienttam_ti, scene: Callable[[int], torch.Tensor]) -> None:
    from sam2.cuda_graphs import GraphCache
    from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
    from sam2.modeling.sam2_prompt import SAM2Prompt
    from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

    predictor = efficienttam_ti
    device = torch.device("cuda")

    def states() -> list[SAM2GenericVideoPredictorState]:
        return [SAM2GenericVideoPredictorState.create((256, 256), SAM2ForgetfulObjectMemoryBank(storage_device=device)) for _ in range(2)]

    def box(obj_id: int, xyxy: list[float]) -> SAM2Prompt:
        return SAM2Prompt(obj_id=obj_id, boxes=torch.tensor([xyxy], device=device))

    prompts_at: dict[int, list[list[SAM2Prompt]]] = {
        0: [[box(0, [30, 80, 100, 150])], [box(0, [93, 25, 163, 95])]],
        2: [[], [box(1, [20, 180, 70, 230])]],
        4: [[box(0, [46, 80, 116, 150])], []],
        9: [[box(0, [66, 80, 136, 150])], [box(1, [20, 180, 70, 230])]],
    }
    eager, graphed = states(), states()
    cache = GraphCache()
    for step in range(14):
        frames: torch.Tensor = scene(step % 8).to(device).permute(0, 3, 1, 2)
        prompts: list[list[SAM2Prompt]] = prompts_at.get(step, [[], []])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            want = predictor.forward_batched(eager, step, frames, prompts, [False, False])
            got = predictor.forward_batched(graphed, step, frames, prompts, [False, False], graphs=cache, one_graph=one_graph)
        for slot in range(2):
            assert list(got[slot]) == list(want[slot])
            for obj_id, result in got[slot].items():
                torch.testing.assert_close(result.masks_logits, want[slot][obj_id].masks_logits, rtol=0, atol=0)
                torch.testing.assert_close(result.obj_ptrs, want[slot][obj_id].obj_ptrs, rtol=0, atol=0)
    assert cache.replays > cache.captures > 0


def test_re_prompts_past_the_bound_keep_steady_steps_on_graphs(efficienttam_ti, scene: Callable[[int], torch.Tensor]) -> None:
    """With max_cond_frames_in_attn and the bank bounded to N, re-prompting more than N times leaves the steady step's
    signature unchanged: after warm-up every steady step replays a graph (unbounded, each prompt adds a conditional
    memory and the next steady step is a new, eager signature)."""
    from sam2.cuda_graphs import GraphCache
    from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank
    from sam2.modeling.sam2_prompt import SAM2Prompt
    from sam2.sam2_generic_video_predictor import SAM2GenericVideoPredictorState

    predictor = efficienttam_ti
    device = torch.device("cuda")
    bound: int = 2
    saved: int = predictor.max_cond_frames_in_attn
    predictor.max_cond_frames_in_attn = bound
    try:
        states = [
            SAM2GenericVideoPredictorState.create((256, 256), SAM2ForgetfulObjectMemoryBank(storage_device=device, max_conditional_memories=bound))
            for _ in range(2)
        ]
        cache = GraphCache()
        eager_on_late_steady_steps: list[int] = []
        for step in range(60):
            frames: torch.Tensor = scene(step % 8).to(device).permute(0, 3, 1, 2)
            prompted: bool = step % 5 == 0  # 12 prompts, far past the bound of 2
            prompts = [[SAM2Prompt(obj_id=0, boxes=torch.tensor([[30.0 + step % 7, 80.0, 100.0, 150.0]], device=device))] if prompted else [] for _ in states]
            eager_before: int = cache.eager
            predictor.forward_batched(states, step, frames, prompts, [False, False], graphs=cache, one_graph=True)
            if not prompted and step >= 30:
                eager_on_late_steady_steps.append(cache.eager - eager_before)
        assert sum(eager_on_late_steady_steps) == 0, eager_on_late_steady_steps
        assert all(len(state.memory_bank.conditional_memories[0]) <= bound + 2 for state in states)
    finally:
        predictor.max_cond_frames_in_attn = saved
