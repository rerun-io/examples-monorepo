"""ZipDepth training package. The network lives in ``monopriors.third_party.zipdepth``; nothing is re-exported here."""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
