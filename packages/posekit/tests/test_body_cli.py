"""The shipped app command accepts an appended body-model selection."""

import shlex
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("yacs", reason="body app CLI requires the posekit-body environment")


@pytest.mark.parametrize("name,precision,compiled", [
    ("sam3d-body-fast", "hybrid", False),
    ("sam3d-body-fast-bf16", "bf16", False),
    ("sam3d-body-fast-torch", "hybrid", True),
    ("sam3d-body-fast-nvfp4", "nvfp4", False),
])
def test_body_cli_selects_measured_backbone_options(name: str, precision: str, compiled: bool) -> None:
    import tyro

    from posekit.models import BodyPose3dConfig
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    config = tyro.cli(BodyPose3dConfig, args=[name])
    assert isinstance(config, Sam3dBodyFastConfig)
    assert config.backbone_precision == precision
    assert config.compile_torch_backbone is compiled
    assert (config.backbone_onnx is None) is compiled


def test_multiview_task_accepts_fast_body_config() -> None:
    import tyro
    from simplecv.data.exoego.hocap import HocapConfig

    from posekit.apis.body3d_mv_body import Sam3MVBodyDemoConfig
    from posekit.models.sam3d_body_fast import Sam3dBodyFastConfig

    manifest = Path(__file__).resolve().parents[3] / "pixi.toml"
    task = tomllib.loads(manifest.read_text())["feature"]["posekit-body"]["tasks"]["posekit-body3d-mv-body"]
    args = shlex.split(task["cmd"])[2:] + ["sam3d-body-fast", "--rr-config.headless"]
    config = tyro.cli(Sam3MVBodyDemoConfig, args=args, config=(tyro.conf.ConsolidateSubcommandArgs,))
    assert isinstance(config.body, Sam3dBodyFastConfig)
    assert isinstance(config.dataset, HocapConfig)
    assert config.max_frames == 50
    assert config.dataset.root_directory == Path("data/hocap/sample")
