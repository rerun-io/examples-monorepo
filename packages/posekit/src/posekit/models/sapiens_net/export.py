"""ONNX-exportable Sapiens2 pose networks: the plain export and the calibrated, quantized 0.4B fast export."""

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from posekit.models.sapiens_net.calibration import FAST_POSE_INPUT_HW, calibration_path, load_calibration
from posekit.models.sapiens_net.checkpoints import load_pose_model
from posekit.models.sapiens_net.fast_head import DeferredHead
from posekit.models.sapiens_net.metadata import coco133_goliath_pairs
from sapiens2_lite.backbone import RopePositionEmbedding

FAST_POSE_EXPORT_VERSION: int = 1


class ExportableRMSNorm(nn.Module):
    """RMSNorm implementation composed from ONNX-exportable tensor operations."""

    def __init__(self, normalized_shape: tuple[int, ...], eps: float | None, weight: torch.Tensor | None) -> None:
        """Create an ONNX-friendly RMSNorm layer."""
        super().__init__()
        self.normalized_shape: tuple[int, ...] = normalized_shape
        self.eps: float | None = eps
        if weight is None:
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(weight.detach().clone())

    @classmethod
    def from_rms_norm(cls, module: nn.RMSNorm) -> "ExportableRMSNorm":
        """Create an equivalent exportable module from `torch.nn.RMSNorm`."""
        normalized_shape: tuple[int, ...] = tuple(int(dim) for dim in module.normalized_shape)
        weight: torch.Tensor | None = module.weight
        exportable: ExportableRMSNorm = cls(normalized_shape=normalized_shape, eps=module.eps, weight=weight)
        exportable.training = module.training
        return exportable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization using primitive tensor math."""
        norm_dims: tuple[int, ...] = tuple(range(-len(self.normalized_shape), 0))
        eps: float = self.eps if self.eps is not None else float(torch.finfo(inputs.dtype).eps)
        variance: torch.Tensor = inputs.to(torch.float32).pow(2).mean(dim=norm_dims, keepdim=True)
        output: torch.Tensor = inputs * torch.rsqrt(variance.to(dtype=inputs.dtype) + eps)
        weight: torch.Tensor | None = self.weight
        if weight is not None:
            output = output * weight.to(dtype=output.dtype)
        return output


def make_sapiens_pose_onnx_exportable(model: nn.Module) -> nn.Module:
    """Replace Sapiens2 RMSNorm modules with equivalent ONNX-friendly modules, in place."""
    child_items: list[tuple[str, nn.Module]] = list(model.named_children())
    for child_name, child_module in child_items:
        if isinstance(child_module, nn.RMSNorm):
            setattr(model, child_name, ExportableRMSNorm.from_rms_norm(child_module))
        elif isinstance(child_module, RopePositionEmbedding):
            rope_module: Any = child_module
            rope_module.dtype = torch.float32
            rope_module.periods.data = rope_module.periods.data.to(dtype=torch.float32)
        else:
            make_sapiens_pose_onnx_exportable(child_module)
    return model


def make_fast_pose_exportable(model: nn.Module) -> nn.Module:
    """Select the mapped joints and replace the head with its exact deferred form."""
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
    from trtkit.artifacts import build_once

    from posekit.models.sapiens_net.quantize import quantize_fast_pose

    digest: str = hashlib.sha256(calibration_path().read_bytes()).hexdigest()[:16]
    path: Path = cache_dir / f"sapiens2_0.4b_nvfp4-768_dyn{max_batch_size}_{digest}_v{FAST_POSE_EXPORT_VERSION}.onnx"

    def export() -> None:
        model: nn.Module = make_fast_pose_exportable(copy.deepcopy(load_pose_model("0.4B")))
        base: Path = path.with_name(path.stem + "_bf16.onnx")
        inputs: torch.Tensor = torch.zeros((3, 3, *FAST_POSE_INPUT_HW), dtype=torch.float32, device="cuda")
        export_onnx(
            model, (inputs,), base, input_names=["inputs"], output_names=["heatmaps"], compute_dtype=torch.bfloat16, dynamic_batch_max=max_batch_size
        )
        # The quantized model and its weights are staged and published model-last.
        quantize_fast_pose(base, path, load_calibration())

    return build_once(path, export)
