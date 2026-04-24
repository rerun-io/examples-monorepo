#!/usr/bin/env python3
"""Materialise one tiny RRD per modality so each iteration is openable in the viewer.

Run via ``pixi run -e slam-evals --frozen slam-evals-smoke`` to drop six
RRDs under ``data/slam-evals/smoke/`` — one per
mono / mono-vi / stereo / stereo-vi / rgbd / rgbd-vi — then open any with::

    rerun data/slam-evals/smoke/<modality>.rrd
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from slam_evals.data.synthetic import build_fixture
from slam_evals.data.types import Modality
from slam_evals.ingest import ingest_sequence

_ALL_MODALITIES: tuple[Modality, ...] = (
    Modality.MONO,
    Modality.MONO_VI,
    Modality.STEREO,
    Modality.STEREO_VI,
    Modality.RGBD,
    Modality.RGBD_VI,
)


@dataclass
class SmokeConfig:
    out: Path = field(default_factory=lambda: Path("data/slam-evals/smoke"))
    """Where the per-modality RRDs (and their fixture trees) get written."""

    n_frames: int = 30
    """Frames per smoke fixture. ~30 gives the GT trajectory enough points to read."""


def main(cfg: SmokeConfig) -> None:
    cfg.out.mkdir(parents=True, exist_ok=True)
    fixtures_root = cfg.out / "fixtures"

    print(f"Building smoke RRDs into {cfg.out.resolve()}\n")
    rows: list[tuple[str, Path, int]] = []
    for modality in _ALL_MODALITIES:
        fixture = build_fixture(
            fixtures_root,
            modality=modality,
            n_frames=cfg.n_frames,
            dataset="SMOKE",
        )
        out_rrd = cfg.out / f"{modality.value}.rrd"
        ingest_sequence(fixture.sequence, out_rrd)
        size_kb = out_rrd.stat().st_size // 1024
        rows.append((modality.value, out_rrd, size_kb))
        print(f"  ok  {modality.value:<10}  {size_kb:>6} KB  ->  {out_rrd}")

    print("\nOpen any of these in the viewer:")
    for modality_value, path, _ in rows:
        print(f"  rerun {path}   # {modality_value}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(SmokeConfig, description="Build one RRD per modality for visual inspection."))
