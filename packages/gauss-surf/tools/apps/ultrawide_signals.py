"""CLI shim for rectified ultrawide depth and normal generation."""

import tyro

from gauss_surf.apis.ultrawide_signals import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
