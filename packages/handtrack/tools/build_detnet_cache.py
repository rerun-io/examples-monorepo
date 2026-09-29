"""CLI shim: decode a split once into a DetNet cache."""
import tyro

from handtrack.apis.build_cache import Config, main

if __name__ == '__main__':
    main(tyro.cli(Config))
