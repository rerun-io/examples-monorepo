"""Thin Tyro shim for direct gsplat training."""

import tyro

from gauss_surf.apis.train_gsplat import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
