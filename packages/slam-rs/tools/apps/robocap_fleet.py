"""CLI shim: replay one RoboCap session on this machine, nothing logged."""

from slam_rs.apis import run
from slam_rs.apis.robocap_fleet import Config, main

if __name__ == "__main__":
    run(Config, main)
