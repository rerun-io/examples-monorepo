"""Small inference-only Sapiens2 pose runtime.

This keeps the Gradio app off the registry, training runners, dense prediction
modules, evaluators, and dataset classes. It intentionally mirrors the old
runtime math so output comparisons can be exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from .backbones import Sapiens2

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    arch: str
    embed_dim: int
    deconv_out_channels: Tuple[int, ...]
    conv_out_channels: Tuple[int, ...]
    image_size: Tuple[int, int] = (1024, 768)
    patch_size: int = 16
    input_size: Tuple[int, int] = (768, 1024)
    heatmap_size: Tuple[int, int] = (192, 256)
    sigma: int = 6
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
        deconv_out_channels: Tuple[int, ...] = (1024, 768),
        deconv_kernel_sizes: Tuple[int, ...] = (4, 4),
        conv_out_channels: Tuple[int, ...] = (512, 512, 256),
        conv_kernel_sizes: Tuple[int, ...] = (1, 1, 1),
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
        layer_out_channels: Tuple[int, ...],
        layer_kernel_sizes: Tuple[int, ...],
    ) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
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
        layer_out_channels: Tuple[int, ...],
        layer_kernel_sizes: Tuple[int, ...],
    ) -> nn.Module:
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
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


class ImagePreprocessor(nn.Module):
    def __init__(
        self,
        mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
        std: Tuple[float, float, float] = (58.395, 57.12, 57.375),
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)

    def forward(self, data: dict) -> dict:
        inputs = data["inputs"].to(self.mean.device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        if inputs.dim() != 4 or inputs.shape[1] != 3:
            raise ValueError(f"Expected CHW or NCHW inputs, got {inputs.shape}")
        inputs = inputs.float()
        inputs = inputs[:, [2, 1, 0], ...]
        data["inputs"] = (inputs - self.mean[None]) / self.std[None]
        data.setdefault("data_samples", None)
        return data


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
    model.codec = UDPHeatmap(
        input_size=spec.input_size,
        heatmap_size=spec.heatmap_size,
        sigma=spec.sigma,
    )
    model.data_preprocessor = ImagePreprocessor()
    model.to(device)
    model.eval()
    return model


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
) -> np.ndarray:
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (
        -0.5 * input_size[0] * math.cos(rot_rad)
        + 0.5 * input_size[1] * math.sin(rot_rad)
        + 0.5 * scale[0]
    )
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (
        -0.5 * input_size[0] * math.sin(rot_rad)
        - 0.5 * input_size[1] * math.cos(rot_rad)
        + 0.5 * scale[1]
    )
    return warp_mat


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def prepare_pose_sample(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    input_size: Tuple[int, int] = (768, 1024),
) -> dict:
    w, h = input_size
    bbox = np.asarray(bbox, dtype=np.float32).reshape(1, 4)
    bbox_center, bbox_scale = bbox_xyxy2cs(bbox, padding=1.25)
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    center = bbox_center[0]
    scale = bbox_scale[0]
    warp_mat = get_udp_warp_matrix(center, scale, rot=0.0, output_size=(w, h))

    sx = np.linalg.norm(warp_mat[0, :2])
    sy = np.linalg.norm(warp_mat[1, :2])
    scale_factor = min(sx, sy)
    interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC

    img = cv2.warpAffine(image_bgr, warp_mat, (int(w), int(h)), flags=interp)
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    if not img.flags.c_contiguous:
        inputs = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    else:
        inputs = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

    return {
        "inputs": inputs,
        "data_samples": {
            "meta": {
                "input_size": (w, h),
                "bbox_center": bbox_center.astype(np.float32),
                "bbox_scale": bbox_scale.astype(np.float32),
                "bbox_score": np.ones(1, dtype=np.float32),
            }
        },
    }


def estimate_pose(
    image_bgr: np.ndarray,
    bboxes: np.ndarray,
    model: TopDownPoseModel,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    inputs_list = []
    samples_list = []
    for bbox in bboxes:
        data = prepare_pose_sample(image_bgr, bbox, input_size=model.spec.input_size)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)
    with torch.no_grad():
        pred = model(inputs)

    pred = pred.cpu().numpy()
    keypoints = []
    scores = []
    for i, sample in enumerate(samples_list):
        kpts_i, scr_i = model.codec.decode(pred[i])
        meta = sample["meta"]
        kpts_i = (
            kpts_i / meta["input_size"] * meta["bbox_scale"]
            + meta["bbox_center"]
            - 0.5 * meta["bbox_scale"]
        )
        keypoints.append(kpts_i[0])
        scores.append(scr_i[0])
    return keypoints, scores


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(heatmaps, np.ndarray), "heatmaps should be numpy.ndarray"
    assert heatmaps.ndim in (3, 4), f"Invalid shape {heatmaps.shape}"

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.0] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)
    return locs, vals


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def refine_keypoints_dark_udp(
    keypoints: np.ndarray, heatmaps: np.ndarray, blur_kernel_size: int
) -> np.ndarray:
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50.0, heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge").flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum("imn,ink->imk", hessian, derivative).squeeze()

    return keypoints


class UDPHeatmap:
    auxiliary_encode_keys = set()

    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        heatmap_type: str = "gaussian",
        sigma: float = 2.0,
        radius_factor: float = 0.0546875,
        blur_kernel_size: int = 11,
    ) -> None:
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.radius_factor = radius_factor
        self.heatmap_type = heatmap_type
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (
            (np.array(input_size) - 1) / (np.array(heatmap_size) - 1)
        ).astype(np.float32)

        if self.heatmap_type != "gaussian":
            raise ValueError("The lightweight pose runtime only supports gaussian UDP heatmaps")

    def decode(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        heatmaps = encoded.copy()
        keypoints, scores = get_heatmap_maximum(heatmaps)
        keypoints = keypoints[None]
        scores = scores[None]
        keypoints = refine_keypoints_dark_udp(
            keypoints, heatmaps, blur_kernel_size=self.blur_kernel_size
        )

        W, H = self.heatmap_size
        keypoints = keypoints / [W - 1, H - 1] * self.input_size
        return keypoints, scores


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


def nms(dets: np.ndarray, thr: float) -> List[int]:
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep
