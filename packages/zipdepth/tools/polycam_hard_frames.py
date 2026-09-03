"""Thin CLI shim for Polycam hard-frame mining."""

import tyro

from zipdepth.apis.polycam_hard_frames import PolycamHardFramesConfig, main

if __name__ == "__main__":
    main(tyro.cli(PolycamHardFramesConfig))
