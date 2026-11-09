"""Thin CLI for offline held-out catalog calibration."""
import tyro

from posekit.apis.hands_cached_calibrate import Config, main

if __name__ == '__main__':
    main(tyro.cli(Config))
