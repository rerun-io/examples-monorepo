#!/usr/bin/env python3
"""Mount HF-hosted egoexo-forge RRDs as a Rerun catalog over gRPC.

Resolves ``.rrd`` files from HuggingFace via ``hf_hub_download`` (cache-first,
no full pre-download), then starts an in-process catalog server with one
Dataset per source. Defaults to 5 RRDs per source (Phase 1 spike); pass
``--max-per-source`` to widen. Block until Ctrl-C so a viewer can connect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import rerun as rr
import tyro

from egoexo_forge.catalog import mount_catalog, resolve_hf_rrds
from egoexo_forge.catalog.hf_resolve import ALL_SOURCES, SourceName


@dataclass
class CatalogConfig:
    sources: tuple[SourceName, ...] = ALL_SOURCES
    """Which HF sources to mount. Each becomes one catalog Dataset."""

    max_per_source: int | None = 5
    """Cap files per source to keep the spike cheap. ``None`` = full corpus."""

    port: int = 9988
    """gRPC port for the catalog server. Different from slam-evals' 9987 so both can run."""

    open_browser: bool = False
    """Also host a web viewer on ``--web-port`` and open the system browser pointed at the catalog."""

    web_port: int = 9091
    """Port for the served web viewer. Only used when ``open_browser`` is true."""


def main(cfg: CatalogConfig) -> None:
    uris_by_source: dict[SourceName, list[str]] = resolve_hf_rrds(
        sources=cfg.sources,
        max_per_source=cfg.max_per_source,
    )
    with mount_catalog(uris_by_source, port=cfg.port) as server:
        url: str = server.url()
        print()
        print("─" * 72)
        print(f"  Catalog URL:  {url}")
        print()
        print("  In the Rerun viewer:  + → Open Data Source → paste the URL")
        print(f"  Or from a terminal:   rerun {url}")
        print("─" * 72)

        if cfg.open_browser:
            rr.serve_web_viewer(web_port=cfg.web_port, open_browser=True, connect_to=url)
            print(f"\nWeb viewer hosted at http://127.0.0.1:{cfg.web_port} (catalog auto-loaded).")

        print("\nServer is up. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("shutting down")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(CatalogConfig, description="Mount egoexo-forge HF RRDs as a Rerun catalog over gRPC."))
