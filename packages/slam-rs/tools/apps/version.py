"""Thin CLI shim: print the version of the compiled slam-rs core."""

from slam_rs.apis import run
from slam_rs.apis.version import Config, main

if __name__ == "__main__":
    run(Config, main)
