"""Thin CLI wrapper to launch the streaming SAM3 video demo."""

import tyro

from sam3_rerun.api.video_stream import Sam3StreamDemoConfig, main

if __name__ == "__main__":
    main(tyro.cli(Sam3StreamDemoConfig))
