"""Log a SOG scene as SogSplats3D (decoded by the custom viewer) plus a stock-viewer preview."""

import tyro

from gsplat_rust_renderer.apis.log_sog import LogSogConfig, main

if __name__ == "__main__":
    main(tyro.cli(LogSogConfig))
