"""One dynamic-profile ONNX graph of the random ViT-S serves different view counts, resolutions, and batches.

The export runs in a clean interpreter (``PIXI_DEV_MODE=0``): symbolic dims are
``torch.SymInt`` values, which beartype's dev-mode ``int`` checks reject. ORT
then evaluates the same graph in-process at shapes the example never saw.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from jaxtyping import Float32, Float64, UInt8
from numpy import ndarray
from torch import Tensor

from monopriors.models.rig_depth.xlens import RigTensors, normalize_framesets, rig_tensors
from monopriors.models.rig_depth.xlens_trt import EngineGeometry, engine_geometry
from monopriors.third_party.xlens.models.dinov2.vision_transformer import FrozenRigGeometry
from monopriors.third_party.xlens.models.net import XLensNet

pytestmark = pytest.mark.slow

CONFIG_PATH: Path = Path(__file__).parents[1] / "monopriors/third_party/xlens/xlens_vits.yaml"
OUTPUT_NAMES: tuple[str, ...] = ("depth_metric", "depth_conf", "mask", "metric_scaling_factor")


def build_random_model() -> XLensNet:
    """The released ViT-S architecture with seed-17 random weights (identical in every interpreter)."""
    config: dict[str, Any] = yaml.safe_load(CONFIG_PATH.read_text())
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


def random_rig(model: XLensNet, views: int, image_hw: tuple[int, int], seed: int) -> tuple[UInt8[ndarray, "s h w 3"], FrozenRigGeometry]:
    """Random all-fisheye rig with poses, frozen on CPU."""
    generator: np.random.Generator = np.random.default_rng(seed)
    images: UInt8[ndarray, "s h w 3"] = generator.integers(0, 256, size=(views, image_hw[0], image_hw[1], 3), dtype=np.uint8)
    rays: Float32[ndarray, "s h w 3"] = generator.normal(size=(views, image_hw[0], image_hw[1], 3)).astype(np.float32)
    rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-6)
    cam_T_ref: Float64[ndarray, "s 4 4"] = np.tile(np.eye(4, dtype=np.float64), (views, 1, 1))
    cam_T_ref[:, 0, 3] = np.arange(views, dtype=np.float64) * 0.1
    tensors: RigTensors = rig_tensors(rays, np.zeros(views, dtype=np.int64), cam_T_ref, torch.device("cpu"))
    with torch.inference_mode():
        frozen: FrozenRigGeometry = model.freeze_geometry(tensors.d_cam, tensors.cam_types, tensors.ray_map)
    return images, frozen


def export_random_dynamic(onnx_path: Path) -> None:
    """Export the seed-17 model with the dynamic profile (views 2-6, 42-112 x 42-140, batch 1-3); runs in the clean interpreter."""
    from trtkit import export_onnx

    from monopriors.models.rig_depth.xlens_trt import DynamicRanges, _XLensRigGraph, dynamic_dims_spec, slot_token_extras

    model: XLensNet = build_random_model()
    images, frozen = random_rig(model, 3, (56, 84), 1)
    geometry: EngineGeometry = engine_geometry(frozen, bias_dtype=torch.float32)
    kinds = slot_token_extras(model.backbone.pretrained, geometry.layer_slots, frozen.calib_k)
    spec = dynamic_dims_spec(geometry, kinds, DynamicRanges(views=(2, 6), patch_rows=(3, 8), patch_cols=(3, 10), batch=(1, 3)), 1 + frozen.calib_k)
    graph = _XLensRigGraph(model, frozen, geometry).eval()
    example: Float32[Tensor, "2 s 3 h w"] = normalize_framesets(np.stack([images, images[::-1]]), torch.device("cpu"))
    export_onnx(
        graph,
        (example, *(geometry.inputs[name] for name in geometry.names)),
        onnx_path,
        input_names=["images", *geometry.names],
        output_names=list(OUTPUT_NAMES),
        compute_dtype=torch.float16,  # CUDA autocast is inert on CPU tensors; the wrapper gives the nesting torch.export expects
        dynamic_dims=spec,
    )


def test_one_dynamic_graph_matches_eager_at_unseen_shapes(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    onnx_path: Path = tmp_path / "xlens-dynamic.onnx"
    env: dict[str, str] = dict(os.environ)
    env["PIXI_DEV_MODE"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "-c", f"import test_xlens_dynamic_export as m; from pathlib import Path; m.export_random_dynamic(Path({str(onnx_path)!r}))"],
        check=True,
        env=env,
    )
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_dims: dict[str, list[object]] = {node.name: [dim if isinstance(dim, int) else str(dim) for dim in node.shape] for node in session.get_inputs()}
    assert input_dims["images"][:3] == ["batch", "views", 3] and all(isinstance(dim, str) for dim in input_dims["images"][3:])
    assert isinstance(input_dims["attn_bias_0"][2], str), "the cross-view bias token dim must stay symbolic"

    model: XLensNet = build_random_model()
    for views, image_hw, batch, seed in ((2, (42, 70), 1, 2), (4, (70, 98), 3, 3), (6, (42, 140), 1, 4)):
        images, frozen = random_rig(model, views, image_hw, seed)
        geometry: EngineGeometry = engine_geometry(frozen, bias_dtype=torch.float32)
        stacked: UInt8[ndarray, "b s h w 3"] = np.stack([np.roll(images, shift, axis=0) for shift in range(batch)])
        image_tensor: Float32[Tensor, "b s 3 h w"] = normalize_framesets(stacked, torch.device("cpu"))
        outputs = session.run(None, {"images": image_tensor.numpy(), **{name: geometry.inputs[name].numpy() for name in geometry.names}})
        with torch.inference_mode():
            for index in range(batch):
                reference: dict = model(image_tensor[index : index + 1], frozen=frozen)
                for value, key in zip(outputs, OUTPUT_NAMES, strict=True):
                    expected = reference[key].numpy()
                    np.testing.assert_allclose(value[index : index + 1], expected, rtol=1e-4, atol=1e-5, err_msg=f"{views} views {image_hw} batch {batch} {key}")
