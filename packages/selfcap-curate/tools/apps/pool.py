"""Rebuild the shipped device table from typed focal-profile results."""

import tyro

from selfcap_curate.apis.pool import PoolConfig, pool_main

if __name__ == "__main__":
    pool_main(tyro.cli(PoolConfig))
