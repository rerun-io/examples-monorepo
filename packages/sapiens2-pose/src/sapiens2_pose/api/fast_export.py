"""Dynamic-batch export of the calibrated Sapiens2 0.4B pose network."""

import copy
import fcntl
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import nn

from sapiens2_pose.api.fast_calibration import FAST_POSE_INPUT_HW, calibration_path, load_calibration
from sapiens2_pose.api.fast_pose import DeferredHead

FAST_POSE_EXPORT_VERSION: int = 1


def make_fast_pose_exportable(model: nn.Module) -> nn.Module:
    """Select the mapped joints and replace the head with its exact deferred form."""
    from sapiens2_pose.api.metadata import coco133_goliath_pairs
    from sapiens2_pose.api.tensorrt_pose import make_sapiens_pose_onnx_exportable

    model = make_sapiens_pose_onnx_exportable(model)
    indices: list[int] = list(coco133_goliath_pairs()[1])
    head: Any = model.decode_head
    original: nn.Conv2d = head.conv_pose
    sliced: nn.Conv2d = nn.Conv2d(original.in_channels, len(indices), 1).to(original.weight.device, original.weight.dtype)
    sliced.weight.data.copy_(original.weight.data[indices])
    assert sliced.bias is not None and original.bias is not None
    sliced.bias.data.copy_(original.bias.data[indices])
    head.conv_pose = sliced
    model.decode_head = DeferredHead(head)
    return model.eval()


def ensure_fast_pose_onnx(cache_dir: Path, max_batch_size: int = 8) -> Path:
    """Export and quantize once; calibration bytes and export version invalidate caches."""
    from trtkit import export_onnx

    from sapiens2_pose.api.fast_onnx import quantize_fast_pose
    from sapiens2_pose.api.runtime import get_pose_model

    digest: str = hashlib.sha256(calibration_path().read_bytes()).hexdigest()[:16]
    path: Path = cache_dir / f"sapiens2_0.4b_nvfp4-768_dyn{max_batch_size}_{digest}_v{FAST_POSE_EXPORT_VERSION}.onnx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return path
        model: nn.Module = make_fast_pose_exportable(copy.deepcopy(get_pose_model("0.4B", "cuda")))
        base: Path = path.with_name(path.stem + "_bf16.onnx")
        inputs: torch.Tensor = torch.zeros((3, 3, *FAST_POSE_INPUT_HW), dtype=torch.float32, device="cuda")
        export_onnx(
            model, (inputs,), base, input_names=["inputs"], output_names=["heatmaps"], compute_dtype=torch.bfloat16, dynamic_batch_max=max_batch_size
        )
        with TemporaryDirectory(prefix="sapiens2-quantize-", dir=cache_dir) as staging:
            staged: Path = Path(staging) / path.name
            quantize_fast_pose(base, staged, load_calibration())
            os.replace(staged.with_suffix(".onnx.data"), path.with_suffix(".onnx.data"))
            os.replace(staged, path)
        return path
