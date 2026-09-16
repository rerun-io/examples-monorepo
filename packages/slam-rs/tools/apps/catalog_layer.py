"""CLI shim for catalog-to-SLAM layer publication."""

from slam_rs.apis import run
from slam_rs.apis.catalog_layer import Config, main

if __name__ == "__main__":
    run(Config, main)
