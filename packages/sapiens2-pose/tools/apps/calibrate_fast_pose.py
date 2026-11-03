"""Regenerate the fast pose calibration from person crops."""

import tyro

from sapiens2_pose.api.calibrate_fast_pose import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
