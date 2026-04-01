"""CLI entrypoint: convert a VRS file to Rerun .rrd format."""

import tyro

from pyvrs_viewer.vrs_to_rerun import VrsToRerunConfig, vrs_to_rerun

if __name__ == "__main__":
    vrs_to_rerun(tyro.cli(VrsToRerunConfig))
