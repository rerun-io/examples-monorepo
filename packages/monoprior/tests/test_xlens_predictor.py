"""X-Lens rig-depth registration, model outputs, and unprojection contract."""

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import tyro
import yaml
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray

from monopriors.models.rig_depth import (
    AnnotatedRigDepthPredictorUnion,
    BaseRigDepthPredictorConfig,
    RigDepthPrediction,
    XLensConfig,
    rig_depth_predictor_defaults,
    unproject,
)
from monopriors.third_party.xlens.inference.preprocess import assemble_batch
from monopriors.third_party.xlens.models.net import XLensNet


@pytest.fixture(scope="module")
def random_model() -> XLensNet:
    """Build the released ViT-S architecture with deterministic random weights."""
    config_path: Path = Path(__file__).parents[1] / "monopriors/third_party/xlens/xlens_vits.yaml"
    config: dict[str, Any] = yaml.safe_load(config_path.read_text())
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return XLensNet(
            backbone_name=config["backbone"],
            checkpoint_dir="/nonexistent/xlens-test-checkpoints",
            head_features=config["head_features"],
            head_out_channels=tuple(config["head_out_channels"]),
            predict_mask=config["predict_mask"],
            scale_head_mode=config["scale_head_mode"],
            scale_head_num_queries=config["scale_head_num_queries"],
            scale_head_num_heads=config["scale_head_num_heads"],
            n_cam_types=config["n_cam_types"],
            use_calib_tokens=config["use_calib_tokens"],
            calib_tokens_per_type=config["calib_tokens_per_type"],
            calib_inject_types=tuple(config["calib_token_inject_types"]),
            use_distortion_bias=config["use_distortion_bias"],
            distortion_bias_layers=config["distortion_bias_layers"],
            distortion_bias_hidden_dim=config["distortion_bias_hidden_dim"],
            distortion_bias_chunk_size=config["distortion_bias_chunk_size"],
        ).eval()


def test_xlens_config_registered() -> None:
    assert fields(BaseRigDepthPredictorConfig) == ()
    assert set(rig_depth_predictor_defaults) == {"xlens"}
    config: BaseRigDepthPredictorConfig = rig_depth_predictor_defaults["xlens"]
    assert isinstance(config, XLensConfig)
    assert config.amp == "bf16"
    assert isinstance(tyro.cli(AnnotatedRigDepthPredictorUnion, args=[]), XLensConfig)


@pytest.mark.parametrize(("views", "cam_types", "with_poses"), [(2, (1, 1), False), (3, (0, 1, 0), False), (3, (0, 1, 0), True)])
def test_random_model_returns_rig_depth_contract(random_model: XLensNet, views: int, cam_types: tuple[int, ...], with_poses: bool) -> None:
    generator: np.random.Generator = np.random.default_rng(19 + views + int(with_poses))
    images: UInt8[ndarray, "s 28 42 3"] = generator.integers(0, 256, size=(views, 28, 42, 3), dtype=np.uint8)
    rays: Float32[ndarray, "s 28 42 3"] = generator.normal(size=(views, 28, 42, 3)).astype(np.float32)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
    types: Int64[ndarray, "s"] = np.asarray(cam_types, dtype=np.int64)
    cam_T_ref: np.ndarray | None = None
    if with_poses:
        cam_T_ref = np.tile(np.eye(4, dtype=np.float64), (views, 1, 1))
        cam_T_ref[:, 0, 3] = np.arange(views, dtype=np.float64) * 0.1
    batch: dict = assemble_batch(list(images), list(rays), types.tolist(), c2w=cam_T_ref, device=torch.device("cpu"))

    with torch.inference_mode():
        output: dict = random_model(batch["images"], ray_map=batch["ray_map"], d_cam=batch["d_cam"], cam_types=batch["cam_types"])
    for key in ("depth_metric", "depth_conf", "mask"):
        tensor: torch.Tensor = output[key]
        assert tensor.shape == (1, views, 28, 42) and tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    assert torch.all((output["mask"] >= 0.0) & (output["mask"] <= 1.0))
    assert output["metric_scaling_factor"].shape == (1,)
    assert float(output["metric_scaling_factor"][0]) > 0.0


def test_unproject_uses_camera_z_depth() -> None:
    depth: Float32[torch.Tensor, "1 1 2"] = torch.tensor([[[2.0, 4.0]]], dtype=torch.float32)
    rays: Float32[ndarray, "1 1 2 3"] = np.array([[[[0.6, 0.0, 0.8], [0.0, 0.0, 0.0]]]], dtype=np.float32)
    prediction: RigDepthPrediction = RigDepthPrediction(
        depth_m=depth,
        confidence=torch.ones_like(depth),
        mask=torch.ones_like(depth),
        scale=1.0,
    )
    points: Float32[ndarray, "1 1 2 3"] = unproject(prediction, rays)
    np.testing.assert_allclose(points[0, 0, 0], np.array([1.5, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(points[0, 0, 1], np.zeros(3, dtype=np.float32))
