"""Refresh a dataset's device assets and default and segment-table blueprints."""

import tyro

from dataforge.apis.publish_blueprints import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
