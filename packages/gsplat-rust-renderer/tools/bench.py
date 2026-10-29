"""Run the Hexels 34-scene benchmark queue (Brush 7k, Hexels 7k, published 30k CUDA and WGSL)."""

import tyro

from gsplat_rust_renderer.apis.bench import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
