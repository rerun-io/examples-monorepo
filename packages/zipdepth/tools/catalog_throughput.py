import tyro

from zipdepth.apis.catalog_throughput import CatalogThroughputConfig, main

if __name__ == "__main__":
    main(tyro.cli(CatalogThroughputConfig))
