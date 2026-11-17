"""Thin CLI shim: convert an Insta360 .insv clip to an exoego:v2 .rrd."""

from slam_rs.apis import run
from slam_rs.apis.insv import Config, main

if __name__ == "__main__":
    run(Config, main)
