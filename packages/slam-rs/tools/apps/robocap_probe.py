"""CLI shim: replay one RoboCap segment through the VIO graph, draw it, and report its regression agreement."""

import tyro

from slam_rs.apis.robocap_probe import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
