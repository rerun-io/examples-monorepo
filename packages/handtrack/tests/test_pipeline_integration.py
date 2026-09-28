"""The pipeline on a real UmeTrack testing segment, in oracle mode: catalog read, NVDEC decode, tracker, fit and scores."""

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import torch
from beartype.roar import BeartypeException

from handtrack.apis.run_pipeline import RunConfig, load_networks, run_segment, select_segments
from handtrack.data.catalog import CATALOG_URL, UMETRACK
from handtrack.eval.segment import SegmentMetrics
from handtrack.results import SegmentTrack, load_track

pytestmark = pytest.mark.integration

SEGMENT: str = "umetrack__real__hand_hand__testing__user_12__recording_13"


def _entry() -> rr.catalog.DatasetEntry:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device for NVDEC decoding")
    try:
        return rr.catalog.CatalogClient(CATALOG_URL).get_dataset(UMETRACK)
    except BeartypeException:
        raise
    except Exception as error:  # the catalog server is an asset: skip when it is unreachable
        pytest.skip(f"catalog {CATALOG_URL} ({UMETRACK}) unreachable: {error}")


def test_oracle_pipeline_tracks_both_hands_within_a_few_millimetres(tmp_path: Path) -> None:
    entry: rr.catalog.DatasetEntry = _entry()
    config: RunConfig = RunConfig(name="it", segments=(SEGMENT,), max_frames=30, detector="oracle", keypoints="oracle", output_root=tmp_path)
    device: torch.device = torch.device("cuda")
    metrics: list[SegmentMetrics] = run_segment(config, entry, select_segments(config, entry)[0], load_networks(config, device), device)
    assert len(metrics) == 1
    score: SegmentMetrics = metrics[0]
    assert score.position.mkpe_mm is not None and score.position.mkpe_mm < 8.0
    assert all(hand.tracking.visible_tracked_fraction is not None and hand.tracking.visible_tracked_fraction > 0.9 for hand in score.hands)
    track: SegmentTrack = load_track(tmp_path / "it" / "known" / f"{SEGMENT}.npz")
    assert track.tracked.shape == (30, 2) and track.box.shape == (30, 4, 2, 4)
    assert bool(np.isfinite(track.landmarks[track.tracked]).all())
