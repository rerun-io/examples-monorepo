"""CLI shim: replay one RoboCap session on this machine, nothing logged."""

import tyro

from slam_rs.apis.robocap_fleet import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
