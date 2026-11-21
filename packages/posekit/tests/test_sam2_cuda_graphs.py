"""sam2-streaming ``GraphCache``: replayed CUDA graphs give the eager results, bit for bit."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("sam2", reason="the streaming SAM 2 fork is in the posekit envs only")

pytestmark = pytest.mark.integration


def _scene(step: int) -> torch.Tensor:
    """Two lenses, 256 px: a square moving right on lens 0, a disc moving down and a fixed square on lens 1."""
    frames: torch.Tensor = torch.full((2, 256, 256, 3), 40, dtype=torch.uint8)
    frames[0, 80:150, 30 + 4 * step : 100 + 4 * step] = torch.tensor([220, 60, 60], dtype=torch.uint8)
    yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
    frames[1][(yy - 60 - 5 * step) ** 2 + (xx - 128) ** 2 < 35**2] = torch.tensor([60, 200, 90], dtype=torch.uint8)
    frames[1, 180:230, 20:70] = torch.tensor([200, 200, 40], dtype=torch.uint8)
    return frames


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


@pytest.mark.parametrize("bf16", [False, True])
def test_forward_batched_with_graphs_is_bit_identical(bf16: bool) -> None:
    _needs_gpu()
    from sam2.cuda_graphs import GraphCache
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

    prompts_at: dict[int, list[list[SAM2Prompt]]] = {
        0: [[box(0, [30, 80, 100, 150])], [box(0, [93, 25, 163, 95])]],
        2: [[], [box(1, [20, 180, 70, 230])]],
        4: [[box(0, [46, 80, 116, 150])], []],
        9: [[box(0, [66, 80, 136, 150])], [box(1, [20, 180, 70, 230])]],
    }
    eager, graphed = states(), states()
    cache = GraphCache()
    for step in range(14):
        frames: torch.Tensor = _scene(step % 8).to(device).permute(0, 3, 1, 2)
        prompts: list[list[SAM2Prompt]] = prompts_at.get(step, [[], []])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            want = predictor.forward_batched(eager, step, frames, prompts, [False, False])
            got = predictor.forward_batched(graphed, step, frames, prompts, [False, False], graphs=cache)
        for slot in range(2):
            assert list(got[slot]) == list(want[slot])
            for obj_id, result in got[slot].items():
                torch.testing.assert_close(result.masks_logits, want[slot][obj_id].masks_logits, rtol=0, atol=0)
                torch.testing.assert_close(result.obj_ptrs, want[slot][obj_id].obj_ptrs, rtol=0, atol=0)
    assert cache.replays > cache.captures > 0
