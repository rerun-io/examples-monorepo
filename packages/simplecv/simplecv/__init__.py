import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # Check function boundaries only. With PEP 526 checks on, every annotated local
    # such as ``x: Float32[ndarray, "n 3"] = ...`` re-evaluates its jaxtyping hint on
    # each execution; jaxtyping returns a new class every time and beartype compiles
    # and caches a new checker for it. Skeleton mappers such as
    # ``assembly21_to_coco133`` run once per frame per hand and spent 1.1 ms per call
    # in those rebuilds (3.3 s of a 3.9 s SHOW3D hand layer), so this package opts
    # out the same way dataforge, exo-calib and slam-rs do.
    beartype_this_package(conf=BeartypeConf(claw_is_pep526=False))
