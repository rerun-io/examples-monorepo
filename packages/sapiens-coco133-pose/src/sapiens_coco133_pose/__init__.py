"""COCO-133 human pose estimation with RTMLib YOLOX and Sapiens2."""

from __future__ import annotations

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
