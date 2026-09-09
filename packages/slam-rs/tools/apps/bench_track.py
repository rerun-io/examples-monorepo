"""Thin CLI shim: measure `Vio.track` on a dumped segment, lane against lane."""

from slam_rs.apis import run
from slam_rs.apis.bench_track import Config, main

if __name__ == "__main__":
    run(Config, main)
