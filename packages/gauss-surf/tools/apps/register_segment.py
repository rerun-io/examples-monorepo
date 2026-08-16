"""CLI shim for reseeding one development-catalog segment."""

import tyro

from gauss_surf.apis.register_segment import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
