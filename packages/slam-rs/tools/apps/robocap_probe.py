"""CLI shim: run the port on one RoboCap session against the basalt C++ trajectory."""

import tyro

from slam_rs.apis.robocap_probe import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
