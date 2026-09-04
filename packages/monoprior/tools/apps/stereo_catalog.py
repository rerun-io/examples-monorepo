import tyro

from monopriors.apis.stereo_catalog import StereoCatalogConfig, main

if __name__ == "__main__":
    main(tyro.cli(StereoCatalogConfig))
