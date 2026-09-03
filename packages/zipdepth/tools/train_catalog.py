"""Tyro shim for catalog ZipDepth fine-tuning."""

import tyro

from zipdepth.apis.train_catalog import TrainCatalogConfig, main

if __name__ == "__main__":
    main(tyro.cli(TrainCatalogConfig))
