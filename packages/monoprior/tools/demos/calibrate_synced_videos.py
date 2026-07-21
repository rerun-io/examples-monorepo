"""CLI shim for synchronized-video calibration."""

import sys

import tyro

from monopriors.apis.calibrate_synced_videos import CalibConfig, main

if __name__ == "__main__":
    sys.exit(main(tyro.cli(CalibConfig)))
