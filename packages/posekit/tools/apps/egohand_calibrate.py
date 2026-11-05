"""Regenerate the fast EgoHand calibration from the Rerun catalog."""

import tyro

from posekit.apis.egohand_calibrate import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
