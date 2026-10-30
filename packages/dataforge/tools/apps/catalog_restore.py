"""Restore catalog references on a separate server."""

import tyro

from dataforge.apis.catalog_snapshot import restore_catalog

if __name__ == "__main__":
    tyro.cli(restore_catalog)
