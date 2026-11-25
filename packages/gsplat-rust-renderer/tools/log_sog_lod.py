"""Log a Streamed SOG (lod-meta.json + SOG files) as SogLodSplats3D plus a stock-viewer preview."""

import tyro

from gsplat_rust_renderer.apis.log_sog_lod import LogSogLodConfig, main

if __name__ == "__main__":
    main(tyro.cli(LogSogLodConfig))
