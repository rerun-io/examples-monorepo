import tyro

from hexels_bench.apis.decode2d import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
