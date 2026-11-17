"""Thin CLI shim: write the stitched panorama of an Insta360 recording as a derived AV1 layer .rrd."""

from slam_rs.apis import run
from slam_rs.apis.insv_panorama import Config, main

if __name__ == "__main__":
    run(Config, main)
