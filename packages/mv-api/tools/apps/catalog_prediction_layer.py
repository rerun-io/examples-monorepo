import tyro

from mv_api.api.catalog_prediction_layer import CatalogPredictionLayerConfig, main

if __name__ == "__main__":
    main(tyro.cli(CatalogPredictionLayerConfig))
