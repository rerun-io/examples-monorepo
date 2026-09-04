"""LAMP multi-camera 3D people tracking.

The vendored LAMP model code and public weights are CC-BY-NC 4.0 and are for
non-commercial use only. See ``lamptrack.third_party.lamp.LICENSE``.
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    beartype_this_package(conf=BeartypeConf(claw_skip_package_names=("lamptrack.third_party.lamp",)))
