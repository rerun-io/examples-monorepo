"""Solve and write one session's calibration without catalog mutation."""

import tyro

from selfcap_curate.apis.calibrate import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
