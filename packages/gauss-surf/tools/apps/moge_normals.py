"""CLI shim for chosen-frame MoGe normal inference."""

import tyro

from gauss_surf.apis.moge_normals import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
