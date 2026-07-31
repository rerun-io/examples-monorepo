import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    beartype_this_package(conf=BeartypeConf(claw_skip_package_names=("monopriors.third_party.g3t", "monopriors.third_party.moge")))
