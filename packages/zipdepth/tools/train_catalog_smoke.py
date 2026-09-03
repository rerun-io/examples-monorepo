"""Tyro shim for the live catalog training smoke gate."""

import tyro

from zipdepth.apis.train_catalog_smoke import TrainCatalogSmokeConfig, main

if __name__ == "__main__":
    main(tyro.cli(TrainCatalogSmokeConfig))
