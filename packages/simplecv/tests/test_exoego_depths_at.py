"""Depth-only access to an exo/ego sequence returns what full indexing returns, without decoding video."""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

SAMPLE: Path = Path(__file__).resolve().parents[1] / "data" / "hocap" / "sample"


def test_exo_depths_at_matches_indexing() -> None:
    from simplecv.data.exoego.hocap import HocapConfig, HocapSequence

    if not (SAMPLE / "subject_8" / "20231024_180733").is_dir():
        pytest.skip(f"missing HOCap sample (pixi run -e simplecv simplecv-view-exoego-data downloads it): {SAMPLE}")
    sequence: HocapSequence = HocapSequence(HocapConfig(root_directory=SAMPLE))
    for idx in (0, 517, len(sequence) - 1):
        timestamp_ns, cam_params, depths = sequence.exo_depths_at(idx)
        sample = sequence[idx]
        assert timestamp_ns == sample.canonical_timestamp_ns
        assert cam_params == sample.exo_cam_params_list
        assert depths is not None and sample.exo_depth_list is not None
        assert len(depths) == len(sample.exo_depth_list)
        for ours, reference in zip(depths, sample.exo_depth_list, strict=True):
            np.testing.assert_array_equal(ours, reference)
