import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # Check function boundaries only. With PEP 526 checks on, every annotated local
    # such as ``x: Float64[ndarray, "n 3"] = ...`` re-evaluates its jaxtyping hint on
    # each execution; jaxtyping returns a new class every time and beartype compiles
    # and caches a new checker for it, so Stage B spent over half its time in
    # ``compile`` and its memory grew without bound over a 4,000-frame capture.
    beartype_this_package(conf=BeartypeConf(claw_is_pep526=False))
