"""Pure tests for the ultrawide teacher-reference diagnostics."""

import numpy as np
import pytest
import torch
from jaxtyping import Bool, Float32
from numpy import ndarray
from torch import Tensor

from zipdepth.apis.eval_teacher_reference import (
    PeripheryReachability,
    prompt_output_range,
    replicate_prompt_canvas,
    score_periphery_reachability,
)
from zipdepth.catalog.ultrawide import DEFAULT_ULTRAWIDE_PROMPT_SCALE, PromptPlacement, prompt_placement


def test_prompt_output_range_ignores_the_zero_padding_only_when_absent() -> None:
    """Report the padded canvas minimum, which is the zero a padded prompt carries."""
    canvas_hw: Float32[ndarray, "192 256"] = np.zeros((192, 256), dtype=np.float32)
    canvas_hw[50:141, 67:189] = np.linspace(1.5, 2.5, 91 * 122, dtype=np.float32).reshape(91, 122)

    assert prompt_output_range(canvas_hw) == pytest.approx((1.5, 2.5))


def test_prompt_output_range_falls_back_to_the_shared_window_when_empty() -> None:
    """An all-zero canvas has no in-range reading, so the head keeps the full window."""
    assert prompt_output_range(np.zeros((192, 256), dtype=np.float32)) == pytest.approx((0.1, 4.0))


def test_replicate_prompt_canvas_fills_the_periphery_from_the_block_edge() -> None:
    """Replacing the zero padding leaves the block untouched and removes every zero."""
    placement: PromptPlacement = prompt_placement(DEFAULT_ULTRAWIDE_PROMPT_SCALE)
    canvas_bchw: Float32[Tensor, "1 1 192 256"] = torch.zeros((1, 1, 192, 256), dtype=torch.float32)
    block: Float32[Tensor, "block_h block_w"] = torch.linspace(1.5, 2.5, placement.height * placement.width).reshape(
        placement.height, placement.width
    )
    canvas_bchw[0, 0, placement.top : placement.top + placement.height, placement.left : placement.left + placement.width] = block

    filled_bchw: Float32[Tensor, "1 1 192 256"] = replicate_prompt_canvas(canvas_bchw, placement)

    assert filled_bchw.shape == canvas_bchw.shape
    assert torch.equal(
        filled_bchw[0, 0, placement.top : placement.top + placement.height, placement.left : placement.left + placement.width], block
    )
    assert float(filled_bchw.min()) == pytest.approx(1.5)
    assert torch.equal(filled_bchw[0, 0, 0, 0], block[0, 0])
    assert torch.equal(filled_bchw[0, 0, -1, -1], block[-1, -1])


def test_score_periphery_reachability_splits_on_the_prompt_range() -> None:
    """Pixels beyond the prompt's own range are scored apart from the rest of the periphery."""
    target_hw: Float32[ndarray, "8 8"] = np.full((8, 8), 2.0, dtype=np.float32)
    target_hw[:4, :] = 3.5
    prediction_hw: Float32[ndarray, "8 8"] = np.full((8, 8), 2.0, dtype=np.float32)
    periphery_hw: Bool[ndarray, "8 8"] = np.ones((8, 8), dtype=bool)

    result: PeripheryReachability = score_periphery_reachability(prediction_hw, target_hw, periphery_hw, (1.0, 3.0))

    assert result.unreachable_fraction == pytest.approx(0.5)
    assert result.unreachable_abs_rel == pytest.approx(1.5 / 3.5)
    assert result.reachable_abs_rel == pytest.approx(0.0)


def test_score_periphery_reachability_returns_none_for_an_empty_subset() -> None:
    """A range covering the whole target leaves no unreachable pixels to score."""
    target_hw: Float32[ndarray, "8 8"] = np.full((8, 8), 2.0, dtype=np.float32)
    prediction_hw: Float32[ndarray, "8 8"] = np.full((8, 8), 2.0, dtype=np.float32)
    periphery_hw: Bool[ndarray, "8 8"] = np.ones((8, 8), dtype=bool)

    result: PeripheryReachability = score_periphery_reachability(prediction_hw, target_hw, periphery_hw, (0.1, 4.0))

    assert result.unreachable_fraction == pytest.approx(0.0)
    assert result.unreachable_abs_rel is None
    assert result.reachable_abs_rel == pytest.approx(0.0)
