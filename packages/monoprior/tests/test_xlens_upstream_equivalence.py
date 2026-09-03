"""Prove that the owned X-Lens inference fork stays bit-identical to upstream."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from jaxtyping import Float32, UInt8
from numpy import ndarray
from torch import Tensor, nn

from monopriors.third_party.xlens.inference import preprocess as owned_preprocess
from monopriors.third_party.xlens.models.net import XLensNet

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "xlens"
UPSTREAM_PACKAGE: str = "xlens_upstream"


def _package(name: str) -> ModuleType:
    package = ModuleType(name)
    package.__path__ = [str(REFERENCE_DIR)]
    package.__package__ = name
    sys.modules[name] = package
    return package


def _load_module(name: str, fixture: str) -> ModuleType:
    path: Path = REFERENCE_DIR / f"upstream_{fixture}.py"
    spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream fixture {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _exec_package_init(package: ModuleType, fixture: str) -> None:
    path: Path = REFERENCE_DIR / f"upstream_{fixture}.py"
    package.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), package.__dict__)


def _load_upstream_package() -> tuple[type[nn.Module], ModuleType]:
    """Load mutually importing pristine fixtures under a synthetic package."""
    _package(UPSTREAM_PACKAGE)
    models = _package(f"{UPSTREAM_PACKAGE}.models")
    _package(f"{UPSTREAM_PACKAGE}.models.utils")
    dinov2 = _package(f"{UPSTREAM_PACKAGE}.models.dinov2")
    layers = _package(f"{UPSTREAM_PACKAGE}.models.dinov2.layers")
    _package(f"{UPSTREAM_PACKAGE}.inference")

    _load_module(f"{UPSTREAM_PACKAGE}.models.utils.head_utils", "head_utils")
    for module_name in ("attention", "drop_path", "layer_scale", "mlp", "patch_embed", "rope", "swiglu_ffn", "calib_distortion", "block"):
        _load_module(f"{UPSTREAM_PACKAGE}.models.dinov2.layers.{module_name}", f"layer_{module_name}")
    _exec_package_init(layers, "layers_init")
    _load_module(f"{UPSTREAM_PACKAGE}.models.dinov2.vision_transformer", "vision_transformer")
    _load_module(f"{UPSTREAM_PACKAGE}.models.dinov2.dinov2", "dinov2")
    _exec_package_init(dinov2, "dinov2_init")
    _load_module(f"{UPSTREAM_PACKAGE}.models.dpt_head", "dpt_head")
    _load_module(f"{UPSTREAM_PACKAGE}.models.ray_map_encoder", "ray_map_encoder")
    net = _load_module(f"{UPSTREAM_PACKAGE}.models.net", "net")
    _exec_package_init(models, "models_init")
    preprocess = _load_module(f"{UPSTREAM_PACKAGE}.inference.preprocess", "preprocess")
    return cast(type[nn.Module], net.XLensNet), preprocess


@pytest.fixture(scope="module")
def equal_models() -> tuple[nn.Module, XLensNet, ModuleType]:
    """Build pristine and owned released architectures with one copied state dict."""
    upstream_type, upstream_preprocess = _load_upstream_package()
    config: dict[str, Any] = yaml.safe_load((REFERENCE_DIR / "xlens_vits.yaml").read_text())
    kwargs: dict[str, Any] = {
        "backbone_name": config["backbone"],
        "checkpoint_dir": "/nonexistent/xlens-equivalence-checkpoints",
        "head_features": config["head_features"],
        "head_out_channels": tuple(config["head_out_channels"]),
        "predict_mask": config["predict_mask"],
        "scale_head_mode": config["scale_head_mode"],
        "scale_head_num_queries": config["scale_head_num_queries"],
        "scale_head_num_heads": config["scale_head_num_heads"],
        "n_cam_types": config["n_cam_types"],
        "use_calib_tokens": config["use_calib_tokens"],
        "calib_tokens_per_type": config["calib_tokens_per_type"],
        "calib_inject_types": tuple(config["calib_token_inject_types"]),
        "use_distortion_bias": config["use_distortion_bias"],
        "distortion_bias_layers": config["distortion_bias_layers"],
        "distortion_bias_hidden_dim": config["distortion_bias_hidden_dim"],
        "distortion_bias_chunk_size": config["distortion_bias_chunk_size"],
    }
    with torch.random.fork_rng():
        torch.manual_seed(23)
        upstream_model: nn.Module = upstream_type(**kwargs).eval()
        torch.manual_seed(29)
        owned_model: XLensNet = XLensNet(**kwargs).eval()
    upstream_state: dict[str, Tensor] = dict(upstream_model.state_dict())
    assert list(upstream_state) == list(owned_model.state_dict())
    owned_model.load_state_dict(upstream_state, strict=True)
    return upstream_model, owned_model, upstream_preprocess


@pytest.mark.parametrize(
    ("cam_types", "with_poses"),
    [((1, 1), False), ((0, 0), False), ((0, 1, 0), False), ((0, 1, 0), True)],
    ids=("pinhole", "fisheye", "mixed-no-poses", "mixed-poses"),
)
def test_owned_model_is_bit_identical(
    equal_models: tuple[nn.Module, XLensNet, ModuleType],
    cam_types: tuple[int, ...],
    with_poses: bool,
) -> None:
    """Owned fp32 outputs equal pristine outputs for every supported rig mode."""
    upstream_model, owned_model, upstream_preprocess = equal_models
    views: int = len(cam_types)
    generator: np.random.Generator = np.random.default_rng(31 + views + int(with_poses))
    images: UInt8[ndarray, "s 28 42 3"] = generator.integers(0, 256, size=(views, 28, 42, 3), dtype=np.uint8)
    rays: Float32[ndarray, "s 28 42 3"] = generator.normal(size=(views, 28, 42, 3)).astype(np.float32)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
    c2w: Float32[ndarray, "s 4 4"] | None = None
    if with_poses:
        c2w = np.tile(np.eye(4, dtype=np.float32), (views, 1, 1))
        c2w[:, 0, 3] = np.arange(views, dtype=np.float32) * 0.2

    upstream_batch: dict[str, Tensor | None] = upstream_preprocess.assemble_batch(
        list(images), list(rays), list(cam_types), c2w=c2w, device=torch.device("cpu")
    )
    owned_batch: dict[str, Tensor | None] = owned_preprocess.assemble_batch(
        list(images), list(rays), list(cam_types), c2w=c2w, device=torch.device("cpu")
    )
    for key in upstream_batch:
        upstream_value: Tensor | None = upstream_batch[key]
        owned_value: Tensor | None = owned_batch[key]
        assert upstream_value is None and owned_value is None or torch.equal(cast(Tensor, upstream_value), cast(Tensor, owned_value))

    with torch.inference_mode():
        upstream_output: dict[str, Tensor] = upstream_model(
            upstream_batch["images"], ray_map=upstream_batch["ray_map"], d_cam=upstream_batch["d_cam"], cam_types=upstream_batch["cam_types"]
        )
        owned_output: dict[str, Tensor] = owned_model(
            owned_batch["images"], ray_map=owned_batch["ray_map"], d_cam=owned_batch["d_cam"], cam_types=owned_batch["cam_types"]
        )
    output_keys: tuple[str, ...] = ("depth", "depth_metric", "depth_conf", "metric_scaling_factor", "mask_logits", "mask")
    assert all(torch.equal(upstream_output[key], owned_output[key]) for key in output_keys)
