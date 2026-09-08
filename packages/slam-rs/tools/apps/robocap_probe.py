"""CLI shim: run the port on one RoboCap session against the basalt C++ trajectory."""

from slam_rs.apis import run
from slam_rs.apis.robocap_probe import Config, main

if __name__ == "__main__":
    run(Config, main)
