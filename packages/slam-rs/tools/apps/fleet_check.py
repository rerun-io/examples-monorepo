"""CLI shim: run the reference smoke clips on this machine and report the D60 verdict beside them."""

import tyro

from slam_rs.apis.fleet_check import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
