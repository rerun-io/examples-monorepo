import tyro

from hexels_bench.apis.upstream_surface import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
