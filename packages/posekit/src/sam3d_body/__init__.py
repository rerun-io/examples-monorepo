"""SAM 3D Body inference subset from facebookresearch/sam-3d-body b5c765a (see VENDORED.md)."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
