"""Registry-consistency tests for the typed model zoo (CPU-only)."""

from posekit import skeletons
from posekit.zoo import PRESETS, ZOO, preset

KNOWN_SKELETONS: dict[str, skeletons.KeypointSkeleton] = {
    skeleton.name: skeleton for skeleton in (skeletons.COCO_17, skeletons.COCO_133, skeletons.HAND_21)
}


def test_zoo_keys_match_entries() -> None:
    for key, model in ZOO.items():
        assert model.key == key


def test_zoo_skeletons_resolve() -> None:
    for model in ZOO.values():
        if model.skeleton is not None:
            assert model.skeleton in KNOWN_SKELETONS, f"{model.key} references unknown skeleton {model.skeleton!r}"


def test_zoo_pose_roles_have_skeletons() -> None:
    for model in ZOO.values():
        if model.role in ("pose2d", "instance-pose2d", "pose3d"):
            assert model.skeleton is not None, f"{model.key} is a pose model without a skeleton"
        if model.role == "detector":
            assert model.skeleton is None, f"{model.key} is a detector with a skeleton"


def test_presets_resolve_and_are_setup_able() -> None:
    for name in PRESETS:
        pair = preset(name)
        assert callable(pair.detector.setup)
        assert callable(pair.pose.setup)
        assert pair.description


def test_unknown_preset_raises() -> None:
    try:
        preset("does-not-exist")
    except KeyError as error:
        assert "does-not-exist" in str(error)
    else:
        raise AssertionError("expected KeyError")
