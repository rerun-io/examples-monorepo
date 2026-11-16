"""Keyframe selection must reproduce `feature_extractor_main` frame for frame.

The reference outputs are real `feature_extractor_main` runs over
`data/r2b_galileo/frames_meta.json` with the flags in
`data/cusfm_re/analysis/sweep_extractor.sh`, kept under
`data/cusfm_re/runs/sweep_extractor/`. Each case pins both the surviving frame
ids and the renumbered rig ids, so a right-count / wrong-frames port fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from colsfm.frames_meta import FramesMeta, read_frames_meta
from colsfm.keyframe_selection import KeyframeSelection, SyncedSamples, select_keyframes, synchronise_samples

SWEEP_CASES: list[tuple[str, float, float]] = [
    ("d0_r5", 0.0, 5.0),
    ("d05_r5", 0.5, 5.0),
    ("d01_r5", 0.1, 5.0),
    ("d002_r5", 0.02, 5.0),
    ("d05_r0", 0.5, 0.0),
    ("d05_r05", 0.5, 0.5),
    ("d99_r99", 99.0, 999.0),
]
"""`(sweep directory, --min_inter_frame_distance, --min_inter_frame_rotation_degrees)`."""


@pytest.fixture(scope="module")
def galileo_input(galileo_input_meta: Path) -> FramesMeta:
    """The 226-keyframe Galileo input metadata."""
    return read_frames_meta(galileo_input_meta)


def test_sample_sync_reproduces_the_input_grouping_as_a_dense_rank(galileo_input: FramesMeta) -> None:
    """At the shipped 100 us the recomputed grouping equals the input's, renumbered 1..29.

    feature_extractor_main.md §4: Galileo's samples span at most 19 us while the
    gaps between them are 33 ms, so the threshold cannot change the partition.
    """
    samples: SyncedSamples = synchronise_samples(galileo_input, 100)
    assert sorted(samples.keyframe_ids_by_synced_sample_id) == list(range(1, 30))
    input_rank: dict[int, int] = {
        original: rank for rank, original in enumerate(sorted({k.synced_sample_id for k in galileo_input.keyframes}), start=1)
    }
    for keyframe in galileo_input.keyframes:
        assert samples.synced_sample_id_by_keyframe_id[keyframe.keyframe_id] == input_rank[keyframe.synced_sample_id]


def test_isaac_defaults_keep_thirty_four_frames_over_five_rig_samples(galileo_input: FramesMeta) -> None:
    """The upstream defaults (0.5 m, 5 deg) keep 34 of 226 frames (NOTES.md decision 6)."""
    selection: KeyframeSelection = select_keyframes(
        galileo_input, min_inter_frame_distance_m=0.5, min_inter_frame_rotation_degrees=5.0
    )
    assert len(selection.kept_keyframe_ids) == 34
    assert selection.kept_synced_sample_ids == (1, 8, 15, 22, 29)


