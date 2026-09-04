"""Thin Tyro shim for the LAMP fixture replay."""

import tyro

from lamptrack.apis.lamp_replay import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
