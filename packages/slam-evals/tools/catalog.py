#!/usr/bin/env python3
"""Mount or refresh a slam-evals catalog over gRPC.

Two modes share this CLI:

* **mount** (default) — start an in-process gRPC catalog server, register
  every layer ``.rrd`` under ``--rrd-dir``, optionally host a web viewer,
  and block until Ctrl-C. The full mount parses ~25 GB of RRDs and takes
  roughly half a minute on the local benchmark corpus.

* **refresh** (``--refresh``) — *don't* start a server. Connect to an
  already-running catalog at ``--catalog-url`` and push only the files
  matching ``--datasets`` / ``--layers`` / ``--only`` using
  ``OnDuplicateSegmentLayer.REPLACE``. Used during the dev loop after a
  partial ``ingest --layers X --force`` so the running viewer picks up
  the change without paying the cold-mount cost again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
import tyro

from slam_evals.blueprint import build_blueprint
from slam_evals.catalog import mount_catalog, refresh_catalog


@dataclass
class CatalogConfig:
    rrd_dir: Path = field(default_factory=lambda: Path("data/slam-evals/rrd"))
    """Directory containing ``*.rrd`` files (recursively scanned)."""

    dataset_name: str = "vslam"
    """Name of the catalog dataset to register the RRDs under."""

    port: int = 9987
    """gRPC port for the catalog server. Pick something different from the running viewer's port (default 9876)."""

    serve: bool = True
    """Block after mounting so a Rerun viewer can connect. Ctrl-C to stop. Ignored in --refresh mode."""

    open_browser: bool = False
    """Opt-in: also host a local web viewer (port 9090 by default) and open the system browser pointed at the catalog URL. Default is off — point your existing native viewer at the printed catalog URL instead. Ignored in --refresh mode."""

    web_port: int = 9090
    """Port for the served web viewer. Only used when ``open_browser`` is true."""

    refresh: bool = False
    """If true, connect to an already-running catalog at ``--catalog-url`` and push only matching files with OnDuplicateSegmentLayer.REPLACE instead of starting a new server."""

    catalog_url: str = "rerun+http://127.0.0.1:9987"
    """URL of the running catalog server. Only used in --refresh mode. Defaults match the mount-mode ``--port`` so a refresh against a default-mounted catalog is a no-arg invocation."""

    datasets: tuple[str, ...] = ()
    """Refresh-mode filter: restrict to specific dataset directories (e.g. ``EUROC KITTI``). Empty = all."""

    layers: tuple[str, ...] = ()
    """Refresh-mode filter: restrict to specific layer names (e.g. ``view_coordinates groundtruth``). Empty = all."""

    only: tuple[str, ...] = ()
    """Refresh-mode filter: restrict to specific segment slugs (e.g. ``EUROC/MH_01_easy``). Empty = all."""


def _run_refresh(cfg: CatalogConfig) -> None:
    n = refresh_catalog(
        cfg.rrd_dir,
        catalog_url=cfg.catalog_url,
        dataset_name=cfg.dataset_name,
        datasets=cfg.datasets,
        layers=cfg.layers,
        only=cfg.only,
    )
    if n == 0:
        return
    print(f"\nRefreshed {n} layer files. Reload the viewer to see the changes.")


def _run_mount(cfg: CatalogConfig) -> None:
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
        print()
        print("  To push edits without restart, in another terminal:")
        print("    pixi run -e slam-evals slam-evals-catalog --refresh \\")
        print("        --datasets <NAME> --layers <stem>")
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


def main(cfg: CatalogConfig) -> None:
    if cfg.refresh:
        _run_refresh(cfg)
    else:
        _run_mount(cfg)


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(CatalogConfig, description="Mount + serve, or refresh, a slam-evals catalog over gRPC."))
