import tyro

from mvs.apis.catalog_depth import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
