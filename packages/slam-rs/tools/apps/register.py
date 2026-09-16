"""Thin CLI shim: register downloaded MSD recordings into a catalog."""

from slam_rs.apis import run
from slam_rs.apis.register import Config, main

if __name__ == "__main__":
    run(Config, main)
