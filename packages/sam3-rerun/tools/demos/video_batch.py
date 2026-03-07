"""Thin CLI wrapper to launch the SAM3 video batch demo."""

import tyro

from sam3_rerun.api.video_batch import Sam3VideoDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3VideoDemoConfig))
