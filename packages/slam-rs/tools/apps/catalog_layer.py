import tyro

from slam_rs.apis.catalog_layer import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
