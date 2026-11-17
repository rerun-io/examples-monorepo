"""Thin CLI shim: measure an Insta360 recording's R_P_imu, frame time offset and rolling-shutter readout from its pixels."""

from slam_rs.apis import run
from slam_rs.apis.insv_calibrate import Config, main

if __name__ == "__main__":
    run(Config, main)
