"""Build the custom gsplat viewer for the web in the stock Rerun web viewer layout."""

import tyro

from gsplat_rust_renderer.apis.build_web_viewer import BuildWebViewerConfig, main

if __name__ == "__main__":
    main(tyro.cli(BuildWebViewerConfig))
