"""Load a Gaussian PLY in Python and log it to the external Rust viewer."""

import tyro

from gsplat_rust_renderer.apis.log_gaussian_ply import LogPlyConfig, main

if __name__ == "__main__":
    main(tyro.cli(LogPlyConfig))
