import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # third_party.zipdepth: the training package torch.compile()s the network; dynamo tracing
    # through beartype's jaxtyping wrapper raises a beartype desynchronization error. The
    # annotations are enforced statically instead (pyrefly, see the monoprior PYREFLY_TARGET).
    beartype_this_package(
        conf=BeartypeConf(claw_skip_package_names=("monopriors.third_party.g3t", "monopriors.third_party.zipdepth"))
    )
