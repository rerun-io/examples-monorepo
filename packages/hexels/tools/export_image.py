import tyro

from hexels_bench.apis.export_image import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
