"""Rebuild factory-fixed OAK RGB/IMU rotations from session fits."""

import tyro

from selfcap_curate.apis.pool import RotationConfig, rotation_main

if __name__ == "__main__":
    rotation_main(tyro.cli(RotationConfig))
