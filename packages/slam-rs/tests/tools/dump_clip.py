"""Dump catalog inputs for offline replay and timing."""

import tyro

from slam_rs.apis.dump_clip import Config, main, write_pgm

__all__ = ["Config", "main", "write_pgm"]

if __name__ == "__main__":
    main(tyro.cli(Config))
