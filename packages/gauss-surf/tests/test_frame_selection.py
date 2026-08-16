"""Behavior checks for sharp-frame scoring and selection."""

import numpy as np
import torch
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from gauss_surf.frame_selection import score_frames, select_ultrawide_frames, select_wide_frames


def test_wide_selection_keeps_every_window_argmax_unconditionally() -> None:
    """Every fixed index window contributes its argmax, including a final partial window."""
    sharpness_n: Float32[ndarray, "n_frames=14"] = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 1.0, 2.0],
        dtype=np.float32,
    )
    timestamps_n: ndarray = np.arange(14, dtype="timedelta64[ns]")

    result = select_wide_frames(
        sharpness_n,
        timestamps_n,
        window_size=6,
        min_chosen=1,
    )

    expected_indices_n: Int64[ndarray, "n_chosen=3"] = np.array([5, 6, 13], dtype=np.int64)
    np.testing.assert_array_equal(result.chosen_indices, expected_indices_n)
    np.testing.assert_array_equal(result.windows, np.array([0, 1, 2], dtype=np.int64))
    assert len(result.chosen_indices) == 3
    assert not result.rejected


def test_scoring_separates_a_blank_frame_from_a_textured_frame() -> None:
    """Sharpness and texture measurements remain available as diagnostics."""
    checkerboard_hw: UInt8[Tensor, "h=60 w=80"] = (
        (torch.arange(60)[:, None] + torch.arange(80)[None, :]) % 2 * 255
    ).to(torch.uint8)
    blank_hwc: UInt8[Tensor, "h=60 w=80 3"] = torch.zeros((60, 80, 3), dtype=torch.uint8)
    textured_hwc: UInt8[Tensor, "h=60 w=80 3"] = torch.stack((checkerboard_hw, checkerboard_hw, checkerboard_hw), dim=-1)
    frames_bhw3: UInt8[Tensor, "b=2 h=60 w=80 3"] = torch.stack((blank_hwc, textured_hwc))

    scores = score_frames(frames_bhw3)

    assert scores.texture[0].item() == 0.0
    assert scores.texture[1].item() > 10.0
    assert scores.sharpness[0].item() == 0.0
    assert scores.sharpness[1].item() > 100.0


def test_ultrawide_selection_keeps_every_frame() -> None:
    """Ultrawide selection admits the full 10 Hz sampling grid."""
    sharpness_n: Float32[ndarray, "n_frames=5"] = np.array([99.0, 100.0, 130.0, 5.0, 100.0], dtype=np.float32)
    timestamps_n: ndarray = np.arange(5, dtype="timedelta64[ns]")

    result = select_ultrawide_frames(sharpness_n, timestamps_n)

    np.testing.assert_array_equal(result.chosen_indices, np.arange(5, dtype=np.int64))
    np.testing.assert_array_equal(result.windows, result.chosen_indices)
    assert result.rules == ("all",) * 5
    assert not result.rejected


def test_wide_selection_rejects_a_segment_below_min_chosen() -> None:
    """Coverage rejection is explicit and retains the sparse audit result."""
    sharpness_n: Float32[ndarray, "n_frames=12"] = np.arange(12, dtype=np.float32) + 200.0
    timestamps_n: ndarray = np.arange(12, dtype="timedelta64[ns]")

    result = select_wide_frames(sharpness_n, timestamps_n, min_chosen=3)

    np.testing.assert_array_equal(result.chosen_indices, np.array([5, 11], dtype=np.int64))
    assert result.rejected
    assert result.rejection_reason == "wide selection kept 2 frames, fewer than min_chosen=3"


def test_scoring_is_invariant_to_source_resolution() -> None:
    """The same pixels at two integer scales produce the same normalized score."""
    rows_h1: Tensor = torch.arange(540, dtype=torch.int64)[:, None]
    columns_1w: Tensor = torch.arange(720, dtype=torch.int64)[None, :]
    pattern_hw: UInt8[Tensor, "h=540 w=720"] = ((rows_h1 * 7 + columns_1w * 11) % 251).to(torch.uint8)
    reference_bhw3: UInt8[Tensor, "b=1 h=540 w=720 3"] = torch.stack((pattern_hw, pattern_hw, pattern_hw), dim=-1)[None]
    doubled_bhw3: UInt8[Tensor, "b=1 h=1080 w=1440 3"] = reference_bhw3.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    reference_scores = score_frames(reference_bhw3)
    doubled_scores = score_frames(doubled_bhw3)

    torch.testing.assert_close(reference_scores.sharpness, doubled_scores.sharpness, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reference_scores.texture, doubled_scores.texture, rtol=1e-5, atol=1e-5)


def test_selection_preserves_packet_timestamps_as_timedelta64_nanoseconds() -> None:
    """Dense and chosen timestamps remain durations and never become plain integer seconds."""
    sharpness_n: Float32[ndarray, "n_frames=3"] = np.array([10.0, 200.0, 300.0], dtype=np.float32)
    timestamps_us_n: ndarray = np.array([1, 2, 3], dtype="timedelta64[us]")

    result = select_ultrawide_frames(sharpness_n, timestamps_us_n)

    assert result.timestamps.dtype == np.dtype("timedelta64[ns]")
    assert result.chosen_timestamps.dtype == np.dtype("timedelta64[ns]")
    np.testing.assert_array_equal(result.chosen_timestamps, np.array([1_000, 2_000, 3_000], dtype="timedelta64[ns]"))
