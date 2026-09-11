"""CLI shim: run the reference smoke clips on this machine and report the ground-truth verdict beside them."""

from slam_rs.apis import run
from slam_rs.apis.fleet_check import Config, main

if __name__ == "__main__":
    run(Config, main)
