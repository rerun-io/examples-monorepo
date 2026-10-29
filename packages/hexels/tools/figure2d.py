import tyro

from hexels_bench.apis.figure2d import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
