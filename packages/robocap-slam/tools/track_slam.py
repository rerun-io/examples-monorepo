#!/usr/bin/env python3
"""CLI wrapper for multicamera visual SLAM."""

import tyro

from robocap_slam.apis.track_slam import TrackSlamConfig, main

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    main(tyro.cli(TrackSlamConfig, description="Run multicamera visual SLAM."))
