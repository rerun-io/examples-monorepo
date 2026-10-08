"""Prepare reproducible NeRF-synthetic training inputs."""
import tyro

from gsplat_rust_renderer.apis.prepare_nerf_init import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
