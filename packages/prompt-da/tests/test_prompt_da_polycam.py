"""Unit tests for the PromptDA Polycam API helpers."""

import numpy as np
from simplecv.data.polycam import DepthConfidenceLevel

from rerun_prompt_da.apis.prompt_da_polycam import filter_depth


def test_filter_depth_zeroes_low_confidence_and_far_values() -> None:
    """Filtered depth should drop both low-confidence and too-distant pixels."""

    depth_mm = np.array([[1000, 2000], [5000, 1500]], dtype=np.uint16)
    confidence = np.array(
        [
            [int(DepthConfidenceLevel.HIGH), int(DepthConfidenceLevel.LOW)],
            [int(DepthConfidenceLevel.MEDIUM), int(DepthConfidenceLevel.HIGH)],
        ],
        dtype=np.uint8,
    )

    filtered = filter_depth(
        depth_mm=depth_mm,
        confidence=confidence,
        confidence_threshold=DepthConfidenceLevel.MEDIUM,
        max_depth_meter=4.0,
    )

    np.testing.assert_array_equal(
        filtered,
        np.array([[1000, 0], [0, 1500]], dtype=np.uint16),
    )
