"""Focal-profile CLI; all logic lives in the checked package."""

import tyro

from selfcap_curate.apis.pool import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
