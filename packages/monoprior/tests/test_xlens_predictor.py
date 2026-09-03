"""X-Lens rig-depth registration, model outputs, export graph, and unprojection contract."""

from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import tyro
import yaml
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor, nn

from monopriors.apis.rig_depth_catalog import RigDepthCatalogConfig
from monopriors.models.rig_depth import (
    AnnotatedRigDepthPredictorUnion,
    BaseRigDepthPredictorConfig,
    RigDepthPrediction,
    XLensConfig,
    XLensTrtConfig,
    rig_depth_predictor_defaults,
    unproject,
)
from monopriors.models.rig_depth.xlens import RigKeyMemo, RigTensors, normalize_images, rig_tensors
from monopriors.models.rig_depth.xlens_trt import EngineGeometry, _XLensRigGraph, engine_geometry
from monopriors.third_party.xlens.inference.preprocess import AssembledBatch, assemble_batch, normalize_image
from monopriors.third_party.xlens.models.dinov2.vision_transformer import FrozenRigGeometry
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


def _random_rig(views: int, with_poses: bool, seed: int) -> tuple[UInt8[ndarray, "s 28 42 3"], Float32[ndarray, "s 28 42 3"], Float64[ndarray, "s 4 4"] | None]:
    generator: np.random.Generator = np.random.default_rng(seed)
    images: UInt8[ndarray, "s 28 42 3"] = generator.integers(0, 256, size=(views, 28, 42, 3), dtype=np.uint8)
    rays: Float32[ndarray, "s 28 42 3"] = generator.normal(size=(views, 28, 42, 3)).astype(np.float32)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
    cam_T_ref: Float64[ndarray, "s 4 4"] | None = None
    if with_poses:
        cam_T_ref = np.tile(np.eye(4, dtype=np.float64), (views, 1, 1))
        cam_T_ref[:, 0, 3] = np.arange(views, dtype=np.float64) * 0.1
    return images, rays, cam_T_ref


def test_registry_has_eager_and_tensorrt_configs() -> None:
    assert fields(BaseRigDepthPredictorConfig) == ()
    assert set(rig_depth_predictor_defaults) == {"xlens", "xlens-trt"}
    eager: BaseRigDepthPredictorConfig = rig_depth_predictor_defaults["xlens"]
    assert isinstance(eager, XLensConfig) and eager.amp == "bf16" and eager.freeze_geometry is True
    tensorrt: BaseRigDepthPredictorConfig = rig_depth_predictor_defaults["xlens-trt"]
    assert isinstance(tensorrt, XLensTrtConfig) and tensorrt.use_cuda_graph is True
    assert isinstance(tyro.cli(AnnotatedRigDepthPredictorUnion, args=["xlens"]), XLensConfig)
    parsed: BaseRigDepthPredictorConfig = tyro.cli(AnnotatedRigDepthPredictorUnion, args=["xlens-trt", "--no-use-cuda-graph"])
    assert isinstance(parsed, XLensTrtConfig) and parsed.use_cuda_graph is False
    catalog: RigDepthCatalogConfig = tyro.cli(RigDepthCatalogConfig, args=["--rr-config.headless", "xlens-trt", "--workspace-gib", "4"])
    assert isinstance(catalog.predictor, XLensTrtConfig) and catalog.predictor.workspace_gib == 4.0
    assert isinstance(tyro.cli(RigDepthCatalogConfig, args=["--rr-config.headless"]).predictor, XLensConfig)
    with pytest.raises(ValueError, match="cuda"):
        XLensTrtConfig().setup("cpu")


@pytest.mark.parametrize(("views", "cam_types", "with_poses"), [(2, (1, 1), False), (3, (0, 1, 0), False), (3, (0, 1, 0), True)])
def test_random_model_returns_rig_depth_contract(random_model: XLensNet, views: int, cam_types: tuple[int, ...], with_poses: bool) -> None:
    images, rays, cam_T_ref = _random_rig(views, with_poses, 19 + views + int(with_poses))
    types: Int64[ndarray, "s"] = np.asarray(cam_types, dtype=np.int64)
    batch: AssembledBatch = assemble_batch(list(images), list(rays), types.tolist(), c2w=cam_T_ref, device=torch.device("cpu"))

    with torch.inference_mode():
        output: dict = random_model(batch["images"], ray_map=batch["ray_map"], d_cam=batch["d_cam"], cam_types=batch["cam_types"])
    for key in ("depth_metric", "depth_conf", "mask"):
        tensor: torch.Tensor = output[key]
        assert tensor.shape == (1, views, 28, 42) and tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
    assert torch.all((output["mask"] >= 0.0) & (output["mask"] <= 1.0))
    assert output["metric_scaling_factor"].shape == (1,)
    assert float(output["metric_scaling_factor"][0]) > 0.0


def test_device_preprocessing_matches_upstream_numpy() -> None:
    """GPU-side normalisation and geometry tensors equal ``assemble_batch`` bit for bit."""
    images, rays, cam_T_ref = _random_rig(3, True, 5)
    cam_types: Int64[ndarray, "s"] = np.asarray((0, 1, 0), dtype=np.int64)
    batch: AssembledBatch = assemble_batch(list(images), list(rays), cam_types.tolist(), c2w=cam_T_ref, device=torch.device("cpu"))
    normalized: Float32[Tensor, "1 s 3 h w"] = normalize_images(images, torch.device("cpu"))
    assert torch.equal(normalized, batch["images"])
    assert torch.equal(normalized[0, 1], torch.from_numpy(normalize_image(images[1])))
    tensors: RigTensors = rig_tensors(rays, cam_types, cam_T_ref, torch.device("cpu"))
    assert torch.equal(tensors.d_cam, batch["d_cam"])
    assert batch["ray_map"] is not None and tensors.ray_map is not None and torch.equal(tensors.ray_map, batch["ray_map"])
    assert torch.equal(tensors.cam_types, batch["cam_types"])
    assert rig_tensors(rays, cam_types, None, torch.device("cpu")).ray_map is None


def test_rig_key_memo_follows_content_and_reuses_the_digest() -> None:
    _, rays, cam_T_ref = _random_rig(2, True, 7)
    cam_types: Int64[ndarray, "s"] = np.asarray((0, 0), dtype=np.int64)
    memo = RigKeyMemo()
    key = memo.key(rays, cam_types, cam_T_ref)
    assert memo.key(rays, cam_types, cam_T_ref) == key
    assert memo.key(rays.copy(), cam_types, cam_T_ref) == key
    assert RigKeyMemo().key(rays, cam_types, cam_T_ref) == key
    assert memo.key(rays, np.asarray((0, 1), dtype=np.int64), cam_T_ref) != key
    assert memo.key(rays, cam_types, None) != key
    other: Float32[ndarray, "s h w 3"] = rays.copy()
    other[0, 0, 0, 0] += 1e-3
    assert memo.key(other, cam_types, cam_T_ref) != key
    assert memo.key(rays, cam_types, cam_T_ref) == key


@pytest.mark.parametrize(("cam_types", "expected_biases", "local_layers_masked"), [((0, 1, 0), 3, True), ((0, 0, 0), 1, False)], ids=("mixed", "fisheye"))
def test_export_graph_matches_frozen_forward(random_model: XLensNet, cam_types: tuple[int, ...], expected_biases: int, local_layers_masked: bool) -> None:
    """The ONNX wrapper (manual scale attention, clamped fp masks, slot mapping) reproduces the frozen eager model."""
    images, rays, cam_T_ref = _random_rig(3, True, 11)
    types: Int64[ndarray, "s"] = np.asarray(cam_types, dtype=np.int64)
    tensors: RigTensors = rig_tensors(rays, types, cam_T_ref, torch.device("cpu"))
    image_tensor: Float32[Tensor, "1 s 3 h w"] = normalize_images(images, torch.device("cpu"))
    with torch.inference_mode():
        frozen: FrozenRigGeometry = random_model.freeze_geometry(tensors.d_cam, tensors.cam_types, tensors.ray_map)
        reference: dict = random_model(image_tensor, frozen=frozen)
        geometry: EngineGeometry = engine_geometry(frozen, bias_dtype=torch.float32)
        graph = _XLensRigGraph(random_model, frozen, geometry).eval()
        outputs: tuple[Tensor, Tensor, Tensor, Tensor] = graph(image_tensor, *(geometry.inputs[name] for name in geometry.names))

    assert geometry.names[:5] == ("ray_feat", "pos_embed", "pos_local", "pos_global", "cam_types")
    bias_names: tuple[str, ...] = geometry.names[5:]
    assert len(bias_names) == expected_biases
    assert len(geometry.layer_slots) == 12
    # Global layers (odd from alt_start=4) always carry the combined per-head bias; local layers only when a view is a pinhole placeholder.
    assert all(geometry.layer_slots[layer] is not None for layer in (5, 7, 9, 11))
    assert all((geometry.layer_slots[layer] is not None) == local_layers_masked for layer in (0, 3, 4, 10))
    assert all(geometry.inputs[name].shape[0] == 1 for name in geometry.names)
    for name in bias_names:
        assert float(geometry.inputs[name].min()) >= -1.0e4
        assert geometry.inputs[name].shape == ((1, 3, geometry.inputs[name].shape[2], geometry.inputs[name].shape[2]) if name in geometry.local_slots else (1, 6, geometry.inputs[name].shape[2], geometry.inputs[name].shape[2]))
    # The wrapper never touches the eager network's own scale head.
    assert all(isinstance(layer, nn.MultiheadAttention) for layer in random_model.scale_head.attn_layers)
    for value, key in zip(outputs, ("depth_metric", "depth_conf", "mask", "metric_scaling_factor"), strict=True):
        assert value.shape == reference[key].shape
        assert torch.allclose(value, reference[key], rtol=1e-4, atol=1e-5), key

    # A batch of two framesets of the same rig: geometry stays batch-1 and broadcasts; each frameset matches its own eager run.
    second_images, _, _ = _random_rig(3, True, 12)
    stacked: Float32[Tensor, "2 s 3 h w"] = torch.cat([image_tensor, normalize_images(second_images, torch.device("cpu"))], dim=0)
    with torch.inference_mode():
        batched: tuple[Tensor, Tensor, Tensor, Tensor] = graph(stacked, *(geometry.inputs[name] for name in geometry.names))
        second_reference: dict = random_model(stacked[1:], frozen=frozen)
    for index, expected in enumerate((reference, second_reference)):
        for value, key in zip(batched, ("depth_metric", "depth_conf", "mask", "metric_scaling_factor"), strict=True):
            assert torch.allclose(value[index : index + 1], expected[key], rtol=1e-4, atol=1e-5), (index, key)


def test_engine_geometry_requires_float_rope_positions(random_model: XLensNet) -> None:
    with torch.inference_mode():
        frozen: FrozenRigGeometry = random_model.backbone.pretrained.freeze_geometry(
            ray_feat=None, d_cam=None, cam_types=None, batch_size=1, n_views=2, image_hw=(28, 42), device=torch.device("cpu")
        )
    assert frozen.pos_local is not None and not frozen.pos_local.is_floating_point()
    with pytest.raises(ValueError, match="FishRoPE"):
        engine_geometry(frozen)


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
