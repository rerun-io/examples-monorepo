"""Snapshot all catalog references without changing the source."""

import tyro

from dataforge.apis.catalog_snapshot import snapshot_catalog

if __name__ == "__main__":
    tyro.cli(snapshot_catalog)
