"""slam-evals: ingest VSLAM-LAB sequences into Rerun RRDs + catalog."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
