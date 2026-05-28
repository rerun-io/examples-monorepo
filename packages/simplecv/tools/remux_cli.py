"""Query video stream from a recording and mux it to an mp4 video file."""

import sys

import tyro

from simplecv.apis.remux_example import RemuxConfig, main

if __name__ == "__main__":
    try:
        main(tyro.cli(RemuxConfig))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
