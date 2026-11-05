"""Regenerate the Sapiens2 nvfp4-768 activation ranges from a directory of RGB person crops."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import orjson
import torch
from serde.json import to_json
from torch import nn

from posekit.models.sapiens_net.calibration import FAST_POSE_INPUT_HW, FastPoseCalibration
from posekit.models.sapiens_net.checkpoints import load_pose_model
from posekit.models.sapiens_net.export import make_fast_pose_exportable
from posekit.ops.crops import IMAGENET_MEAN_255, IMAGENET_STD_255
from sapiens2_lite.backbone import GroupedQueryAttention


@dataclass(frozen=True, slots=True)
class SapiensCalibrateConfig:
    """Calibration input and explicit output; the committed ranges are never overwritten by default."""

    crops_dir: Path
    """Directory containing person crop images with the production bbox padding."""
    output: Path
    """JSON destination for review before replacing the package data."""
    max_crops: int = 256
    """Maximum number of images to process."""


@torch.inference_mode()
def main(config: SapiensCalibrateConfig) -> None:
    """Measure input and post-RoPE attention maxima with bf16 compute."""
    model: nn.Module = make_fast_pose_exportable(load_pose_model("0.4B"))
    input_amax: dict[str, float] = {}
    attention_amax: dict[str, float] = {}
    handles: list[Any] = []
    originals: list[tuple[Any, Callable[..., torch.Tensor]]] = []

    def record(name: str) -> Callable[..., None]:
        def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            input_amax[name] = max(input_amax.get(name, 0.0), args[0].abs().amax().float().item())

        return hook

    def attention(name: str, original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            for tag, value in zip("qkv", (q, k, v), strict=True):
                key: str = f"{name}.{tag}"
                attention_amax[key] = max(attention_amax.get(key, 0.0), value.abs().amax().float().item())
            return original(q, k, v, *args, **kwargs)

        return call

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            handles.append(module.register_forward_pre_hook(record(name)))
        if isinstance(module, GroupedQueryAttention):
            attn: Any = module
            originals.append((attn, attn.attn_op))
            attn.attn_op = attention(name, attn.attn_op)
    paths: list[Path] = sorted(path for path in config.crops_dir.iterdir() if path.suffix.lower() in (".png", ".jpg", ".jpeg"))[: config.max_crops]
    mean: torch.Tensor = torch.tensor(IMAGENET_MEAN_255, device="cuda").view(1, 3, 1, 1)
    std: torch.Tensor = torch.tensor(IMAGENET_STD_255, device="cuda").view(1, 3, 1, 1)
    try:
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read crop {path}")
            image = cv2.cvtColor(cv2.resize(image, FAST_POSE_INPUT_HW[::-1]), cv2.COLOR_BGR2RGB)
            inputs: torch.Tensor = torch.from_numpy(image).to(device="cuda", dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model((inputs - mean) / std)
    finally:
        for handle in handles:
            handle.remove()
        for attn, original in originals:
            attn.attn_op = original
    calibration: FastPoseCalibration = FastPoseCalibration(FAST_POSE_INPUT_HW, len(paths), input_amax, attention_amax)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(to_json(calibration, option=orjson.OPT_INDENT_2) + "\n")
