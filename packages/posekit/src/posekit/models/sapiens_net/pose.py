# pyrefly: ignore-errors
# Upstream-shaped code, untyped as upstream wrote it (dict metainfo, mixed nn.Module lists); moved with annotation-only changes.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Condensed from facebookresearch/sapiens2 0e51c12 (sapiens/pose/src/models/heads/pose_heatmap_head.py,
# sapiens/pose/src/models/core/pose_topdown_estimator.py, sapiens/pose/src/datasets/utils.py); licensed under
# the Sapiens2 License (LICENSE.md, shipped by recipes/sapiens2-src).
"""Inference-only Sapiens2 pose network: backbone, heatmap head, checkpoint loading, keypoint metadata.

Off the registry, training runners, dense prediction modules, evaluators and
dataset classes. Crops and heatmap decoding run on the GPU in posekit
(``posekit.ops``), so the CPU crop/decode helpers are not vendored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from sapiens2_lite.backbone import Sapiens2


@dataclass(frozen=True)
class ModelSpec:
    arch: str
    embed_dim: int
    deconv_out_channels: tuple[int, ...]
    conv_out_channels: tuple[int, ...]
    image_size: tuple[int, int] = (1024, 768)
    patch_size: int = 16
    input_size: tuple[int, int] = (768, 1024)
    heatmap_size: tuple[int, int] = (192, 256)
    sigma: float = 6.0
    num_keypoints: int = 308


MODEL_SPECS = {
    "0.4B": ModelSpec(
        arch="sapiens2_0.4b",
        embed_dim=1024,
        deconv_out_channels=(1024, 768),
        conv_out_channels=(512, 512, 256),
    ),
    "0.8B": ModelSpec(
        arch="sapiens2_0.8b",
        embed_dim=1280,
        deconv_out_channels=(1024, 768),
        conv_out_channels=(512, 512, 256),
    ),
    "1B": ModelSpec(
        arch="sapiens2_1b",
        embed_dim=1536,
        deconv_out_channels=(1536, 1024),
        conv_out_channels=(768, 512, 256),
    ),
    "5B": ModelSpec(
        arch="sapiens2_5b",
        embed_dim=2432,
        deconv_out_channels=(1024, 768),
        conv_out_channels=(512, 512, 256),
    ),
}


def _load_python_module(path: Path, name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PoseHeatmapHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        deconv_out_channels: tuple[int, ...] = (1024, 768),
        deconv_kernel_sizes: tuple[int, ...] = (4, 4),
        conv_out_channels: tuple[int, ...] = (512, 512, 256),
        conv_kernel_sizes: tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.deconv_layers = self._make_deconv_layers(
            in_channels, deconv_out_channels, deconv_kernel_sizes
        )
        in_channels = deconv_out_channels[-1]
        self.conv_layers = self._make_conv_layers(
            in_channels, conv_out_channels, conv_kernel_sizes
        )
        in_channels = conv_out_channels[-1]
        self.conv_pose = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                weight_dtype = module.weight.dtype
                weight = nn.init.kaiming_normal_(
                    module.weight.float(), mode="fan_out", nonlinearity="relu"
                )
                module.weight.data = weight.to(weight_dtype)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                weight_dtype = module.weight.dtype
                weight = nn.init.kaiming_normal_(
                    module.weight.float(), mode="fan_in", nonlinearity="linear"
                )
                module.weight.data = weight.to(weight_dtype)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                if hasattr(module, "weight"):
                    nn.init.ones_(module.weight)

    def _make_conv_layers(
        self,
        in_channels: int,
        layer_out_channels: tuple[int, ...],
        layer_kernel_sizes: tuple[int, ...],
    ) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes, strict=False):
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_deconv_layers(
        self,
        in_channels: int,
        layer_out_channels: tuple[int, ...],
        layer_kernel_sizes: tuple[int, ...],
    ) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes, strict=False):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f"Unsupported deconv kernel size: {kernel_size}")
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.SiLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        return self.conv_pose(x)


class TopDownPoseModel(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.backbone = Sapiens2(
            arch=spec.arch,
            img_size=spec.image_size,
            patch_size=spec.patch_size,
            final_norm=True,
            use_tokenizer=False,
            with_cls_token=True,
            out_type="featmap",
        )
        self.decode_head = PoseHeatmapHead(
            in_channels=spec.embed_dim,
            out_channels=spec.num_keypoints,
            deconv_out_channels=spec.deconv_out_channels,
            conv_out_channels=spec.conv_out_channels,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.backbone(inputs)[0]
        return self.decode_head(x)


def init_pose_model(size: str, checkpoint: str | Path, device: str = "cuda") -> TopDownPoseModel:
    if size not in MODEL_SPECS:
        raise KeyError(f"Unknown Sapiens2 pose size {size!r}. Valid: {list(MODEL_SPECS)}")

    spec = MODEL_SPECS[size]
    model = TopDownPoseModel(spec)
    state_dict = load_file(str(checkpoint), device="cpu")
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys:
        print(f"Missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        print(f"Unexpected keys: {incompat.unexpected_keys}")
    print(f"\033[96mModel loaded from {checkpoint}\033[0m")

    model.spec = spec
    model.to(device)
    model.eval()
    return model


def parse_pose_metainfo(metainfo: dict) -> dict:
    if "from_file" in metainfo:
        module = _load_python_module(Path(metainfo["from_file"]), "_sapiens2_keypoints308")
        metainfo = module.dataset_info

    assert "dataset_name" in metainfo
    assert "keypoint_info" in metainfo
    assert "skeleton_info" in metainfo

    parsed = dict(
        dataset_name=None,
        num_keypoints=None,
        keypoint_id2name={},
        keypoint_name2id={},
        upper_body_ids=[],
        lower_body_ids=[],
        flip_indices=[],
        flip_pairs=[],
        keypoint_colors=[],
        num_skeleton_links=None,
        skeleton_links=[],
        skeleton_link_colors=[],
        dataset_keypoint_weights=None,
        sigmas=None,
    )

    parsed["dataset_name"] = metainfo["dataset_name"]

    for key in (
        "remove_teeth",
        "min_visible_keypoints",
        "teeth_keypoint_ids",
        "coco_wholebody_to_goliath_mapping",
        "coco_wholebody_to_goliath_keypoint_info",
        "idx_to_original_idx_mapping",
    ):
        if key in metainfo:
            parsed[key] = metainfo[key]

    parsed["num_keypoints"] = len(metainfo["keypoint_info"])
    for kpt_id, kpt in metainfo["keypoint_info"].items():
        kpt_name = kpt["name"]
        parsed["keypoint_id2name"][kpt_id] = kpt_name
        parsed["keypoint_name2id"][kpt_name] = kpt_id
        parsed["keypoint_colors"].append(kpt.get("color", [255, 128, 0]))

        kpt_type = kpt.get("type", "")
        if kpt_type == "upper":
            parsed["upper_body_ids"].append(kpt_id)
        elif kpt_type == "lower":
            parsed["lower_body_ids"].append(kpt_id)

        swap_kpt = kpt.get("swap", "")
        if swap_kpt == kpt_name or swap_kpt == "":
            parsed["flip_indices"].append(kpt_name)
        else:
            parsed["flip_indices"].append(swap_kpt)
            pair = (swap_kpt, kpt_name)
            if pair not in parsed["flip_pairs"]:
                parsed["flip_pairs"].append(pair)

    parsed["num_skeleton_links"] = len(metainfo["skeleton_info"])
    for _, sk in metainfo["skeleton_info"].items():
        parsed["skeleton_links"].append(sk["link"])
        parsed["skeleton_link_colors"].append(sk.get("color", [96, 96, 255]))

    if "joint_weights" in metainfo:
        parsed["dataset_keypoint_weights"] = np.array(
            metainfo["joint_weights"], dtype=np.float32
        )
    if "sigmas" in metainfo:
        parsed["sigmas"] = np.array(metainfo["sigmas"], dtype=np.float32)
    if "stats_info" in metainfo:
        parsed["stats_info"] = {
            name: np.array(val, dtype=np.float32)
            for name, val in metainfo["stats_info"].items()
        }

    def _map(src: Any, mapping: dict):
        if isinstance(src, (list, tuple)):
            cls = type(src)
            return cls(_map(s, mapping) for s in src)
        return mapping[src]

    parsed["flip_pairs"] = _map(parsed["flip_pairs"], parsed["keypoint_name2id"])
    parsed["flip_indices"] = _map(parsed["flip_indices"], parsed["keypoint_name2id"])
    parsed["skeleton_links"] = _map(parsed["skeleton_links"], parsed["keypoint_name2id"])

    parsed["keypoint_colors"] = np.array(parsed["keypoint_colors"], dtype=np.uint8)
    parsed["skeleton_link_colors"] = np.array(
        parsed["skeleton_link_colors"], dtype=np.uint8
    )
    return parsed
