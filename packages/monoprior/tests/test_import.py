"""Smoke test: verify the monopriors package can be imported."""


def test_import_monopriors() -> None:
    import monopriors  # noqa: F401


def test_import_vendored_moge() -> None:
    from monopriors.third_party import utils3d  # noqa: F401
    from monopriors.third_party.moge.model.v1 import MoGeModel as MoGeModelV1  # noqa: F401
    from monopriors.third_party.moge.model.v2 import MoGeModel as MoGeModelV2  # noqa: F401
