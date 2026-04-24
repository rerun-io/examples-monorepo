#!/usr/bin/env python3
"""Ingest sequences from a manifest into per-sequence RRD files."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro
from tqdm import tqdm

from slam_evals.data import discover_sequences
from slam_evals.data.discovery import filter_sequences
from slam_evals.data.types import Modality, Sequence
from slam_evals.ingest import ingest_sequence


@dataclass
class IngestConfig:
    manifest: Path | None = None
    """Manifest JSON produced by ``discover.py``. If unset, ``benchmark_root`` is scanned on the fly."""

    benchmark_root: Path = field(default_factory=lambda: Path("/home/pablo/0Dev/work/VSLAM-LAB-Benchmark"))
    """Used only when ``manifest`` is unset."""

    out: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Output directory. Each sequence writes to ``<out>/<dataset>/<name>.rrd``."""

    only: tuple[str, ...] = ()
    """Restrict to specific slugs or sequence names (e.g. ``EUROC/MH_01_easy`` or ``MH_01_easy``)."""

    datasets: tuple[str, ...] = ()
    """Restrict to specific datasets (e.g. ``EUROC KITTI``)."""

    modalities: tuple[Modality, ...] = ()
    """Restrict to specific modalities."""

    force: bool = False
    """Re-ingest sequences whose RRD already exists."""

    limit: int | None = None
    """Cap the number of sequences processed (post-filter)."""


def _load_sequences(cfg: IngestConfig) -> list[Sequence]:
    if cfg.manifest is not None:
        raw = json.loads(cfg.manifest.read_text())
        return [
            Sequence(
                dataset=s["dataset"],
                name=s["name"],
                root=Path(s["root"]),
                modality=Modality(s["modality"]),
                has_calibration=bool(s.get("has_calibration", False)),
            )
            for s in raw["sequences"]
        ]
    return discover_sequences(cfg.benchmark_root.expanduser().resolve())


def main(cfg: IngestConfig) -> None:
    sequences = _load_sequences(cfg)
    sequences = filter_sequences(
        sequences,
        only=cfg.only or None,
        datasets=cfg.datasets or None,
        modalities=cfg.modalities or None,
    )
    if cfg.limit is not None:
        sequences = sequences[: cfg.limit]

    print(f"Ingesting {len(sequences)} sequences -> {cfg.out}")
    cfg.out.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed: list[tuple[str, str]] = []
    for seq in tqdm(sequences, desc="sequences"):
        out_rrd = cfg.out / seq.dataset / f"{seq.name}.rrd"
        if out_rrd.exists() and not cfg.force:
            tqdm.write(f"skip (exists): {seq.slug}")
            succeeded += 1
            continue
        t0 = time.monotonic()
        try:
            ingest_sequence(seq, out_rrd)
        except (OSError, ValueError, RuntimeError) as exc:
            failed.append((seq.slug, f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"FAIL {seq.slug}: {exc}")
            continue
        dur = time.monotonic() - t0
        tqdm.write(f"ok   {seq.slug} [{seq.modality}] in {dur:.1f}s -> {out_rrd}")
        succeeded += 1

    print(f"\nDone: {succeeded}/{len(sequences)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failures:")
        for slug, err in failed:
            print(f"  {slug}: {err}")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(IngestConfig, description="Ingest VSLAM-LAB sequences into Rerun RRDs."))
