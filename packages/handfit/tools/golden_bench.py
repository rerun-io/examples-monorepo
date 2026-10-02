"""Thin CLI: replay the numpy golden set through the native warm fit (``handfit.apis.golden_bench``)."""

import tyro

from handfit.apis.golden_bench import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
