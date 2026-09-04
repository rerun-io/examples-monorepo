"""CLI shim for calibrated rig-depth catalog streaming."""

import tyro

from monopriors.apis.rig_depth_catalog import RigDepthCatalogConfig, main

if __name__ == "__main__":
    main(tyro.cli(RigDepthCatalogConfig))
