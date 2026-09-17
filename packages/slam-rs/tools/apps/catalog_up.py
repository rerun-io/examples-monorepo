"""Thin CLI shim: start or reuse the local catalog."""

from slam_rs.apis import run
from slam_rs.apis.catalog_up import Config, main

if __name__ == "__main__":
    run(Config, main)
