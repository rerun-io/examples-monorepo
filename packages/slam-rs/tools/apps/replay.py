"""Thin CLI shim: replay a reference segment through the slam-rs core."""

import tyro

from slam_rs.apis.replay import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
