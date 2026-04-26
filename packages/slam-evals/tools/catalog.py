#!/usr/bin/env python3
"""Mount a directory of slam-evals RRDs as a Rerun catalog and serve it.

Spins up the catalog on a fixed port, registers the slam-evals default
blueprint at the dataset level (so the viewer applies it to every recording
without each RRD needing a baked-in copy), prints the viewer-connectable URL,
and blocks until Ctrl-C. Per-segment metadata is browsable in the viewer's
catalog panel — nothing extra to print here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
import tyro

from slam_evals.blueprint import build_blueprint
from slam_evals.catalog import mount_catalog


@dataclass
class CatalogConfig:
    rrd_dir: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Directory containing ``*.rrd`` files (recursively scanned)."""

    dataset_name: str = "vslam"
    """Name of the catalog dataset to register the RRDs under."""

    port: int = 9987
    """gRPC port for the catalog server. Pick something different from the running viewer's port (default 9876)."""

    serve: bool = True
    """Block after mounting so a Rerun viewer can connect. Ctrl-C to stop."""

    open_browser: bool = False
    """Opt-in: also host a local web viewer (port 9090 by default) and open the system browser pointed at the catalog URL. Default is off — point your existing native viewer at the printed catalog URL instead."""

    web_port: int = 9090
    """Port for the served web viewer. Only used when ``open_browser`` is true."""


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

        if cfg.open_browser:
            # Hosts a local web viewer at ``http://127.0.0.1:<web_port>`` and
            # opens the system browser pointed at it; the page auto-loads
            # the catalog via ``connect_to`` so the user doesn't have to do
            # the ``+ → Open Data Source → paste`` dance manually.
            rr.serve_web_viewer(web_port=cfg.web_port, open_browser=True, connect_to=url)
            print(f"\nWeb viewer hosted at http://127.0.0.1:{cfg.web_port} (catalog auto-loaded).")

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
