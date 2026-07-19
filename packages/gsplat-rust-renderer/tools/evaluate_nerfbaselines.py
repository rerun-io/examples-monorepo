"""Thin CLI shim for full-split nerfbaselines evaluation."""

import tyro

from gsplat_rust_renderer.apis.evaluate_nerfbaselines import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
