import tyro

from monopriors.apis.stereo_catalog_layer import StereoCatalogLayerConfig, main

if __name__ == "__main__":
    main(tyro.cli(StereoCatalogLayerConfig))
