"""Backfill legacy RoboCap sensor metadata."""

import tyro

from dataforge.apis.robocap_calibration import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
