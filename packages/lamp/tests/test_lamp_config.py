"""Public tracker configuration and registry tests."""

from posekit.models.rtdetr import RtDetrDetectorConfig
from posekit.models.vitpose import VitPoseConfig

from lamptrack.models.lamp import LampConfig, lamp_tracker_defaults


def test_default_config_matches_released_lamp_2d_models() -> None:
    """The default uses RT-DETR and LAMP's exact COCO ViTPose+ expert."""
    config = LampConfig()
    assert isinstance(config.detector, RtDetrDetectorConfig)
    assert isinstance(config.pose, VitPoseConfig)
    assert config.pose.model_id == "usyd-community/vitpose-plus-base"
    assert config.pose.dataset_index == 0
    assert config.window == 20
    assert config.keypoint_conf_min == 0.5


def test_registry_exposes_default_lamp_tracker() -> None:
    """The Tyro registry contains the single supported tracker."""
    assert set(lamp_tracker_defaults) == {"lamp"}
    assert isinstance(lamp_tracker_defaults["lamp"], LampConfig)
