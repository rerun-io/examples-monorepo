"""Thin CLI shim: measure `Vio.track` on a dumped segment, lane against lane."""

import tyro

from slam_rs.apis.bench_track import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
