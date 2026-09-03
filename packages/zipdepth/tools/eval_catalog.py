"""Tyro shim for held-out catalog evaluation."""

import tyro

from zipdepth.apis.eval_catalog import EvalCatalogConfig, main

if __name__ == "__main__":
    main(tyro.cli(EvalCatalogConfig))
