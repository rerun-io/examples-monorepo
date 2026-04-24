#!/usr/bin/env python3
"""Scan a VSLAM-LAB-style benchmark root and write a manifest of ingestible sequences."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from slam_evals.data import discover_sequences


@dataclass
class DiscoverConfig:
    benchmark_root: Path = field(default_factory=lambda: Path("/home/pablo/0Dev/work/VSLAM-LAB-Benchmark"))
    """Root directory containing <dataset>/<sequence>/ layouts."""

    out: Path = field(default_factory=lambda: Path("data/slam-evals/manifest.json"))
    """Where to write the JSON manifest."""


def main(cfg: DiscoverConfig) -> None:
    sequences = discover_sequences(cfg.benchmark_root.expanduser().resolve())

    manifest = {
        "benchmark_root": str(cfg.benchmark_root.expanduser().resolve()),
        "sequences": [
            {
                "dataset": s.dataset,
                "name": s.name,
                "slug": s.slug,
                "root": str(s.root),
                "modality": str(s.modality),
                "has_calibration": s.has_calibration,
            }
            for s in sequences
        ],
    }

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cfg.out.write_text(json.dumps(manifest, indent=2))

    modality_counts = Counter(str(s.modality) for s in sequences)
    dataset_counts = Counter(s.dataset for s in sequences)
    print(f"Discovered {len(sequences)} sequences -> {cfg.out}")
    print("By modality:")
    for m, n in sorted(modality_counts.items()):
        print(f"  {m:<12} {n}")
    print("By dataset:")
    for d, n in sorted(dataset_counts.items()):
        print(f"  {d:<18} {n}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(DiscoverConfig, description="Discover VSLAM-LAB benchmark sequences."))
