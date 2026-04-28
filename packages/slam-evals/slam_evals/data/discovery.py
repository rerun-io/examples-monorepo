"""Walk a VSLAM-LAB benchmark root and classify each sequence by modality.

Required-files triad (skip otherwise): ``rgb_0/``, ``rgb.csv``, ``groundtruth.csv``.

Modality classification rules, from most-to-least specific — a sequence that
has both stereo and depth streams is classified as stereo (stereo matches the
PyCuVSLAM baseline surface; the depth channel still gets ingested as an extra
entity downstream):

1. ``rgb_1/`` and ``path_rgb_1`` column → ``stereo[-vi]``
2. else ``depth_0/`` and ``path_depth_0`` column → ``rgbd[-vi]``
3. else → ``mono[-vi]``

``-vi`` suffix appends when ``imu_0.csv`` exists alongside.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from serde import SerdeError

from slam_evals.data.parse import parse_rgb_csv
from slam_evals.data.types import Modality, Sequence


def _classify(rgb_csv: Path, has_rgb_1_dir: bool, has_depth_0_dir: bool, has_imu: bool) -> Modality | None:
    try:
        parsed = parse_rgb_csv(rgb_csv)
    except (ValueError, KeyError, OSError, SerdeError, csv.Error):
        return None

    has_rgb_1 = has_rgb_1_dir and parsed.path_rgb_1 is not None
    has_depth_0 = has_depth_0_dir and parsed.path_depth_0 is not None

    if has_rgb_1:
        return Modality.STEREO_VI if has_imu else Modality.STEREO
    if has_depth_0:
        return Modality.RGBD_VI if has_imu else Modality.RGBD
    return Modality.MONO_VI if has_imu else Modality.MONO


def discover_sequences(benchmark_root: Path) -> list[Sequence]:
    """Enumerate valid sequences under ``benchmark_root`` (depth 2)."""
    sequences: list[Sequence] = []
    for dataset_dir in sorted(p for p in benchmark_root.iterdir() if p.is_dir()):
        for seq_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            rgb_0 = seq_dir / "rgb_0"
            rgb_csv = seq_dir / "rgb.csv"
            gt_csv = seq_dir / "groundtruth.csv"
            if not (rgb_0.is_dir() and rgb_csv.is_file() and gt_csv.is_file()):
                continue

            modality = _classify(
                rgb_csv=rgb_csv,
                has_rgb_1_dir=(seq_dir / "rgb_1").is_dir(),
                has_depth_0_dir=(seq_dir / "depth_0").is_dir(),
                has_imu=(seq_dir / "imu_0.csv").is_file(),
            )
            if modality is None:
                continue

            sequences.append(
                Sequence(
                    dataset=dataset_dir.name,
                    name=seq_dir.name,
                    root=seq_dir,
                    modality=modality,
                    has_calibration=(seq_dir / "calibration.yaml").is_file(),
                )
            )
    return sequences


def filter_sequences(
    sequences: Iterable[Sequence],
    *,
    only: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    modalities: Iterable[Modality] | None = None,
) -> list[Sequence]:
    """Subset sequences by exact slug, dataset name, or modality."""
    out = list(sequences)
    if only is not None:
        wanted = set(only)
        out = [s for s in out if s.slug in wanted or s.name in wanted]
    if datasets is not None:
        want = set(datasets)
        out = [s for s in out if s.dataset in want]
    if modalities is not None:
        want_m = set(modalities)
        out = [s for s in out if s.modality in want_m]
    return out
