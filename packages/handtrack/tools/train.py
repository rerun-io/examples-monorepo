"""CLI shim for shared-source handtrack training."""
import tyro

from handtrack.apis.train import Config, main

if __name__ == '__main__':
    main(tyro.cli(Config))
