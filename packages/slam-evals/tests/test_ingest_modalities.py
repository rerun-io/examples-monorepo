"""Per-modality round-trip tests for ``slam_evals.ingest.ingest_sequence``.

Each test builds a tiny synthetic sequence (see ``conftest.build_fixture``),
ingests it into per-stream layer files, mounts the resulting directory as
a one-segment Rerun catalog, and verifies the segment_table aggregates
properties from every applicable layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slam_evals.catalog import mount_catalog, segment_summary
from slam_evals.data.types import Modality
from slam_evals.ingest import applicable_layers, ingest_sequence

_ALL_MODALITIES: list[Modality] = [
    Modality.MONO,
    Modality.MONO_VI,
    Modality.STEREO,
    Modality.STEREO_VI,
    Modality.RGBD,
    Modality.RGBD_VI,
]


@pytest.mark.parametrize("modality", _ALL_MODALITIES, ids=lambda m: m.value)
def test_ingest_modality_roundtrip(tmp_path: Path, fixture_factory, modality: Modality) -> None:
    fx = fixture_factory(modality)
    rrd_dir = tmp_path / "rrd"

    written = ingest_sequence(fx.sequence, rrd_dir)
    expected_layers = applicable_layers(fx.sequence)
    assert len(written) == len(expected_layers), f"expected {len(expected_layers)} layer files, got {len(written)}"
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0, f"empty layer file {path}"

    seq_dir = rrd_dir / fx.sequence.dataset / fx.sequence.name
    written_layers = {p.stem for p in seq_dir.glob("*.rrd")}
    assert written_layers == set(expected_layers), f"on-disk layers {written_layers} != expected {set(expected_layers)}"

    with mount_catalog(rrd_dir, dataset_name="vslam") as server:
        pdf = segment_summary(server, dataset_name="vslam")

    assert len(pdf) == 1, f"expected exactly one segment, got {len(pdf)}"
    row = pdf.iloc[0]

    # Cross-cutting `info` properties live on calibration.rrd.
    # Single-element list per cell is rerun's catalog representation.
    assert row["property:info:modality"][0] == modality.value
    assert row["property:info:dataset"][0] == fx.sequence.dataset
    assert row["property:info:sequence"][0] == fx.sequence.name
    assert bool(row["property:info:has_calibration"][0])
    assert bool(row["property:info:has_imu"][0]) == modality.has_imu
    assert bool(row["property:info:has_depth"][0]) == modality.has_depth
    assert bool(row["property:info:has_stereo"][0]) == modality.has_stereo

    # Per-layer counts now live on their own layer's property bag.
    assert row["property:rgb_0:num_frames"][0] == fx.n_frames
    assert row["property:groundtruth:num_poses"][0] == fx.n_frames

    if modality.has_stereo:
        assert row["property:rgb_1:num_frames"][0] == fx.n_frames
    if modality.has_depth:
        assert row["property:depth_0:num_frames"][0] == fx.n_frames
    if modality.has_imu:
        assert row["property:imu_0:num_samples"][0] == fx.n_frames * 10


def test_ingest_handles_empty_groundtruth(tmp_path: Path, fixture_factory) -> None:
    fx = fixture_factory(Modality.MONO)

    # Truncate groundtruth.csv to just the header — mirrors HAMLYN / HILTI2026
    # sequences that ship without trajectory data.
    gt_path = fx.sequence.root / "groundtruth.csv"
    header = gt_path.read_text().splitlines()[0]
    gt_path.write_text(header + "\n")

    rrd_dir = tmp_path / "rrd"
    written = ingest_sequence(fx.sequence, rrd_dir)
    assert len(written) > 0

    with mount_catalog(rrd_dir, dataset_name="vslam") as server:
        pdf = segment_summary(server, dataset_name="vslam")
    assert pdf.iloc[0]["property:groundtruth:num_poses"][0] == 0


def test_layers_subset_is_idempotent(tmp_path: Path, fixture_factory) -> None:
    """Re-emitting only ``rgb_0`` should leave other layer mtimes untouched."""
    fx = fixture_factory(Modality.STEREO_VI)
    rrd_dir = tmp_path / "rrd"
    ingest_sequence(fx.sequence, rrd_dir)

    seq_dir = rrd_dir / fx.sequence.dataset / fx.sequence.name
    other_layers = {p.stem: p.stat().st_mtime_ns for p in seq_dir.glob("*.rrd") if p.stem != "rgb_0"}

    written = ingest_sequence(fx.sequence, rrd_dir, layers={"rgb_0"})
    assert {p.stem for p in written} == {"rgb_0"}

    for stem, mtime in other_layers.items():
        new_mtime = (seq_dir / f"{stem}.rrd").stat().st_mtime_ns
        assert new_mtime == mtime, f"layer {stem!r} was unexpectedly rewritten"
