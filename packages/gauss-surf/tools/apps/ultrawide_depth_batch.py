"""CLI shim for the corpus-wide ultrawide depth layer pass."""

import tyro

from gauss_surf.apis.ultrawide_depth_batch import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
