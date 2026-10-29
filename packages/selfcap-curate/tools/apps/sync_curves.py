"""Catalog-only rotation-speed evidence CLI."""

import tyro

from selfcap_curate.apis.sync_curves import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
