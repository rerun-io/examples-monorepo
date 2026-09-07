"""Thin CLI shim: print the version of the compiled slam-rs core."""

import tyro

from slam_rs.apis.version import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
