"""Refresh a SelfCap dataset's catalog blueprints only."""

import tyro

from dataforge.apis.selfcap_blueprints import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
