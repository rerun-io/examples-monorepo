"""Per-modality round-trip tests for ``slam_evals.ingest.ingest_sequence``.

Each test builds a tiny synthetic sequence (see ``conftest.build_fixture``),
ingests it, and then mounts the resulting RRD as a one-element Rerun catalog
to verify the ``segment_table()`` properties line up with the expected
modality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import rerun as rr

from slam_evals.catalog import segment_summary
from slam_evals.data.types import Modality
from slam_evals.ingest import ingest_sequence

_ALL_MODALITIES: list[Modality] = [
    Modality.MONO,
    Modality.MONO_VI,
    Modality.STEREO,
    Modality.STEREO_VI,
    Modality.RGBD,
    Modality.RGBD_VI,
]


def _ingest_fixture(out_dir: Path, fixture) -> Path:
    rrd = out_dir / f"{fixture.sequence.dataset}__{fixture.sequence.name}.rrd"
    ingest_sequence(fixture.sequence, rrd)
    return rrd


@pytest.mark.parametrize("modality", _ALL_MODALITIES, ids=lambda m: m.value)
def test_ingest_modality_roundtrip(tmp_path: Path, fixture_factory, modality: Modality) -> None:
    fx = fixture_factory(modality)

    rrd_dir = tmp_path / "rrd"
    rrd_dir.mkdir()
    rrd = _ingest_fixture(rrd_dir, fx)

    assert rrd.exists(), f"RRD not written for {modality}"
    assert rrd.stat().st_size > 0, f"RRD empty for {modality}"

    with rr.server.Server(datasets={"vslam": [str(rrd)]}) as server:
        df = segment_summary(server, dataset_name="vslam")

    assert len(df) == 1, f"expected exactly one segment, got {len(df)}"
    row = df.iloc[0]

    # Values in segment_table come back wrapped in a single-element list per
    # property column — unwrap with [0] before comparing.
    assert row["property:info:modality"][0] == modality.value
    assert row["property:info:dataset"][0] == fx.sequence.dataset
    assert row["property:info:sequence"][0] == fx.sequence.name
    assert row["property:info:num_rgb_frames"][0] == fx.n_frames
    assert row["property:info:num_gt_poses"][0] == fx.n_frames
    # Segment-table cells come back as numpy scalars / arrow-derived values,
    # so compare via bool(...) rather than ``is True``.
    assert bool(row["property:info:has_calibration"][0])
    assert bool(row["property:info:has_imu"][0]) == modality.has_imu
    assert bool(row["property:info:has_depth"][0]) == modality.has_depth
    assert bool(row["property:info:has_stereo"][0]) == modality.has_stereo

    if modality.has_imu:
        assert row["property:info:num_imu_samples"][0] == fx.n_frames * 10
    else:
        assert row["property:info:num_imu_samples"][0] == 0


def test_ingest_handles_empty_groundtruth(tmp_path: Path, fixture_factory) -> None:
    fx = fixture_factory(Modality.MONO)

    # Truncate groundtruth.csv to just the header — mirrors HAMLYN / HILTI2026
    # sequences that ship without trajectory data.
    gt_path = fx.sequence.root / "groundtruth.csv"
    header = gt_path.read_text().splitlines()[0]
    gt_path.write_text(header + "\n")

    rrd_dir = tmp_path / "rrd"
    rrd_dir.mkdir()
    rrd = _ingest_fixture(rrd_dir, fx)
    assert rrd.exists()
    assert rrd.stat().st_size > 0

    with rr.server.Server(datasets={"vslam": [str(rrd)]}) as server:
        df = segment_summary(server, dataset_name="vslam")
    assert df.iloc[0]["property:info:num_gt_poses"][0] == 0
