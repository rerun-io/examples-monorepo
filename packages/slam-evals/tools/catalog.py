#!/usr/bin/env python3
"""Mount a directory of slam-evals RRDs as a Rerun catalog and query / view it."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from slam_evals.catalog import mount_catalog, segment_summary


@dataclass
class CatalogConfig:
    rrd_dir: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Directory containing ``*.rrd`` files (recursively scanned)."""

    dataset_name: str = "vslam"
    """Name of the catalog dataset to register the RRDs under."""

    print_summary: bool = True
    """Print the per-sequence segment summary to stdout."""

    filter_modality: str | None = None
    """Optional substring filter on ``property:info:modality`` (e.g. ``stereo-vi``)."""

    filter_dataset: str | None = None
    """Optional substring filter on ``property:info:dataset`` (e.g. ``EUROC``)."""

    serve: bool = False
    """After mounting, block so a Rerun viewer can connect. Ctrl-C to stop."""


def main(cfg: CatalogConfig) -> None:
    with mount_catalog(cfg.rrd_dir, dataset_name=cfg.dataset_name) as server:
        print(f"Mounted catalog '{cfg.dataset_name}' from {cfg.rrd_dir.resolve()}")

        if cfg.print_summary:
            df = segment_summary(server, dataset_name=cfg.dataset_name)
            if cfg.filter_dataset and "property:info:dataset" in df.columns:
                df = df[df["property:info:dataset"].astype(str).str.contains(cfg.filter_dataset, case=False, na=False)]
            if cfg.filter_modality and "property:info:modality" in df.columns:
                df = df[df["property:info:modality"].astype(str).str.contains(cfg.filter_modality, case=False, na=False)]
            print(f"\n{len(df)} recordings:\n")
            print(df.to_string(index=False))

        if cfg.serve:
            print("\nServer is up. Connect a Rerun viewer to it; Ctrl-C to stop.")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("shutting down")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(CatalogConfig, description="Mount + query a slam-evals catalog."))
