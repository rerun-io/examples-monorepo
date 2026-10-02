"""Thin CLI: write the golden set's cold rows for the Rust cold_bench example (``handfit.apis.export_cold_golden``)."""

import tyro

from handfit.apis.export_cold_golden import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
