"""Sam2VideoSegmenter: which prompts ask the predictor for SAM 2's three masks."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("sam2", reason="the streaming SAM 2 fork is in the posekit envs only")


class _SpyPredictor:
    def __init__(self) -> None:
        self.multimask: list[bool] = []

    def forward_batched(self, states, frame_idx: int, frames, prompts, multimask_output: list[bool], graphs=None) -> list[dict]:
        self.multimask.extend(multimask_output)
        return [{} for _ in states]


def _multimask_flags(single_click_multimask: bool) -> list[bool]:
    from posekit.models.base import SegmentationPrompts
    from posekit.models.sam2_video import Sam2VideoSegmenter, Sam2VideoSegmenterConfig

    segmenter: Sam2VideoSegmenter = Sam2VideoSegmenter.__new__(Sam2VideoSegmenter)
    segmenter.config = Sam2VideoSegmenterConfig(device="cpu", single_click_multimask=single_click_multimask)
    spy: _SpyPredictor = _SpyPredictor()
    segmenter.predictor = spy  # pyrefly: ignore[bad-assignment]
    segmenter._graphs = None
    segmenter.reset()
    frame: torch.Tensor = torch.zeros((1, 8, 8, 3), dtype=torch.uint8)
    zero: torch.Tensor = torch.zeros(1, dtype=torch.long)
    click: SegmentationPrompts = SegmentationPrompts(frame_indices=zero, points_xy=torch.tensor([[4.0, 4.0]]), track_ids=zero)
    box: SegmentationPrompts = SegmentationPrompts(frame_indices=zero, boxes_xyxy=torch.tensor([[1.0, 1.0, 6.0, 6.0]]), track_ids=zero)
    for prompts in (click, box, None):
        segmenter.step(frame, prompts)
    return spy.multimask


def test_single_click_takes_the_single_mask_by_default() -> None:
    assert _multimask_flags(single_click_multimask=False) == [False, False, False]


def test_single_click_multimask_is_opt_in_and_only_for_clicks() -> None:
    assert _multimask_flags(single_click_multimask=True) == [True, False, False]
