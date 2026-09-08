"""Thin CLI shim: replay a reference segment through the slam-rs core."""

from slam_rs.apis import run
from slam_rs.apis.replay import Config, main

if __name__ == "__main__":
    run(Config, main)
