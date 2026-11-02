import tyro

from hexels_bench.apis.surface_table import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
