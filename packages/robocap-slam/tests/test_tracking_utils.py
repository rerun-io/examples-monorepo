from __future__ import annotations

import pytest

from robocap_slam.apis.tracking_utils import bounded_frame_count


def test_bounded_frame_count_defaults_to_all_frames() -> None:
    assert bounded_frame_count(995, None) == 995


def test_bounded_frame_count_clamps_to_dataset_length() -> None:
    assert bounded_frame_count(995, 30) == 30
    assert bounded_frame_count(12, 30) == 12


def test_bounded_frame_count_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="max_frames"):
        bounded_frame_count(995, 0)
