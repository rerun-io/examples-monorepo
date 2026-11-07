"""Depth as AV1 10-bit monochrome video: encoder and `DepthVideoStream` logging helper for the depth-video-viewer."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
