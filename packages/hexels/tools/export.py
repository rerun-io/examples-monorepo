import tyro

from hexels_bench.apis.export import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
