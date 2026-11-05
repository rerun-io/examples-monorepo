"""Fast hand roles remain selectable through the ordinary Tyro unions."""

import pytest
import tyro

pytest.importorskip("warp", reason="EgoHand configs require posekit-hands")

from trtkit import TensorRtBackendConfig

from posekit.models import HandDetectorConfig, HandPose3dConfig
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig


def test_fast_role_subcommands_select_the_calibrated_config() -> None:
    detector = tyro.cli(HandDetectorConfig, args=["egohand-fast", "tensorrt"])
    pose = tyro.cli(HandPose3dConfig, args=["egohand-fast"])
    assert isinstance(detector, EgoHandDetectorConfig)
    assert isinstance(detector.backend, TensorRtBackendConfig)
    assert detector.gpu_preprocessing and detector.fp16_trunk
    assert isinstance(pose, EgoHandConfig)
    assert pose.cuda_execution and pose.nvfp4_backbone and pose.fp8_tail_blocks == 8
    original = tyro.cli(HandPose3dConfig, args=["egohand"])
    assert isinstance(original, EgoHandConfig)
    assert not original.cuda_execution and not original.nvfp4_backbone
