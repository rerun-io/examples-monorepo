"""Thin Tyro shim for Robocap catalog LAMP tracking."""

import tyro

from lamptrack.apis.lamp_catalog import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
