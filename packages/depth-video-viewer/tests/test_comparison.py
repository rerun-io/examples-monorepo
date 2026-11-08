import numpy as np
import pytest

from depth_video_viewer.apis.comparison import to_codes


def test_codes_are_linear_in_depth_with_zero_kept_as_no_depth() -> None:
    depth = np.array([[[0, 1023, 2046]]], np.uint16)
    codes, unit = to_codes(depth)
    assert unit == 2.0
    assert codes.tolist() == [[[0, 512, 1023]]]


def test_positive_depths_that_would_become_no_depth_are_rejected() -> None:
    # unit = 3000 / 1023 mm, so 1 mm rounds to code 0.
    with pytest.raises(ValueError, match="1 depths below"):
        to_codes(np.array([[[0, 1, 3000]]], np.uint16))
