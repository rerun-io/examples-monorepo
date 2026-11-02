"""Build runs/TABLE.json and runs/table.rrd from the Hexels benchmark run directories."""

import tyro

from hexels_bench.apis.bench_table import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
