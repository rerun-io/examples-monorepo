"""Registry-consistency tests for the typed model zoo (CPU-only)."""

from posekit import skeletons
from posekit.zoo import PRESETS, ZOO, preset

KNOWN_SKELETONS: dict[str, skeletons.KeypointSkeleton] = {
    skeleton.name: skeleton for skeleton in (skeletons.COCO_17, skeletons.COCO_133, skeletons.HAND_21, skeletons.SOMA_HAND)
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
        if model.role in ("pose2d", "instance-pose2d", "pose3d", "hand-pose3d"):
            assert model.skeleton is not None, f"{model.key} is a pose model without a skeleton"
        if model.role in ("detector", "hand-detector"):
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


def test_hand_models_are_registered_in_the_zoo() -> None:
    """Both hand stacks are implemented: a hand detector and a top-down 3D hand model each."""
    assert {key: (ZOO[key].role, ZOO[key].skeleton, ZOO[key].status) for key in ("wilor-yolo-hand", "wilor", "rtdetr-hand-v1", "egohand-v1")} == {
        "wilor-yolo-hand": ("hand-detector", None, "implemented"),
        "wilor": ("hand-pose3d", "hand_21", "implemented"),
        "rtdetr-hand-v1": ("hand-detector", None, "implemented"),
        "egohand-v1": ("hand-pose3d", "soma_hand_25", "implemented"),
    }


def test_hand_configs_are_tyro_subcommands() -> None:
    """``--hand-detector wilor --hand-pose egohand`` style selection, in the base posekit env too."""
    import tyro

    from posekit.models import AnnotatedHandDetectorConfig, AnnotatedHandPose3dConfig
    from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig
    from posekit.models.wilor import WilorHandDetectorConfig, WilorHandPoseConfig

    assert isinstance(tyro.cli(AnnotatedHandDetectorConfig, args=["wilor"]), WilorHandDetectorConfig)
    assert isinstance(tyro.cli(AnnotatedHandDetectorConfig, args=["egohand"]), EgoHandDetectorConfig)
    assert isinstance(tyro.cli(AnnotatedHandPose3dConfig, args=["wilor"]), WilorHandPoseConfig)
    assert isinstance(tyro.cli(AnnotatedHandPose3dConfig, args=["egohand"]), EgoHandConfig)


def test_existing_2d_configs_parse_on_the_cli() -> None:
    """The graph-transform hook is code-only; it must not break CLI parsing (posekit-video-pose --help)."""
    import tyro

    from posekit.models import AnnotatedDetectorConfig, AnnotatedPose2dConfig, RtmPoseConfig, YoloxDetectorConfig

    assert isinstance(tyro.cli(AnnotatedDetectorConfig, args=["yolox"]), YoloxDetectorConfig)
    assert isinstance(tyro.cli(AnnotatedPose2dConfig, args=["rtmpose"]), RtmPoseConfig)
