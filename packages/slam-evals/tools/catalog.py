#!/usr/bin/env python3
"""Mount a directory of slam-evals RRDs as a Rerun catalog and serve it.

Defaults: spin up the catalog on a fixed port, register the slam-evals
default blueprint at the dataset level (so the viewer applies it to every
recording without each RRD needing a baked-in copy), print the
viewer-connectable URL, optionally print the per-recording segment
summary, and block until Ctrl-C.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from slam_evals.blueprint import build_blueprint
from slam_evals.catalog import mount_catalog, segment_summary


@dataclass
class CatalogConfig:
    rrd_dir: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Directory containing ``*.rrd`` files (recursively scanned)."""

    dataset_name: str = "vslam"
    """Name of the catalog dataset to register the RRDs under."""

    port: int = 9987
    """gRPC port for the catalog server. Pick something different from the running viewer's port (default 9876)."""

    print_summary: bool = True
    """Print the per-sequence segment summary to stdout once mounted."""

    filter_modality: str | None = None
    """Optional case-insensitive substring filter on ``property:info:modality``."""

    filter_dataset: str | None = None
    """Optional case-insensitive substring filter on ``property:info:dataset``."""

    serve: bool = True
    """Block after mounting so a Rerun viewer can connect. Ctrl-C to stop."""


def main(cfg: CatalogConfig) -> None:
    with mount_catalog(
        cfg.rrd_dir,
        dataset_name=cfg.dataset_name,
        port=cfg.port,
        blueprint=build_blueprint(),
    ) as server:
        url = server.url()
        print(f"Mounted catalog '{cfg.dataset_name}' from {cfg.rrd_dir.resolve()}")
        print()
        print("─" * 72)
        print(f"  Catalog URL:  {url}")
        print()
        print("  In the Rerun viewer:  + → Open Data Source → paste the URL")
        print(f"  Or from a terminal:   rerun {url}")
        print("─" * 72)

        if cfg.print_summary:
            df = segment_summary(server, dataset_name=cfg.dataset_name)
            if cfg.filter_dataset and "property:info:dataset" in df.columns:
                df = df[df["property:info:dataset"].astype(str).str.contains(cfg.filter_dataset, case=False, na=False)]
            if cfg.filter_modality and "property:info:modality" in df.columns:
                df = df[df["property:info:modality"].astype(str).str.contains(cfg.filter_modality, case=False, na=False)]
            print(f"\n{len(df)} recordings:\n")
            print(df.to_string(index=False))

        if cfg.serve:
            print("\nServer is up. Ctrl-C to stop.")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                print("shutting down")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(CatalogConfig, description="Mount + serve a slam-evals catalog over gRPC."))
