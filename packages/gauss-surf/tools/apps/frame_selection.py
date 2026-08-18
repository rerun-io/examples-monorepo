"""CLI shim for the catalog frame-selection stage."""

import tyro

from gauss_surf.apis.frame_selection import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
