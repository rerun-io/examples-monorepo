import tyro

from simplecv.apis.exoego_forge_catalog import CatalogConfig, main

if __name__ == "__main__":
    main(tyro.cli(CatalogConfig, description="Mount local ExoEgo Forge RRDs as a Rerun catalog."))
