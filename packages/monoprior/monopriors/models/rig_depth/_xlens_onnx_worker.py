"""Clean-interpreter ONNX export for the X-Lens TensorRT predictor.

Dynamic export dims reach the owned fork as ``torch.SymInt`` values, which the
dev-mode beartype ``int`` checks reject; ``XLensTrtPredictor`` re-enters the
identical export here with ``PIXI_DEV_MODE=0`` (the MoGe v2 pattern).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

from monopriors.models.rig_depth.xlens import load_xlens_model
from monopriors.models.rig_depth.xlens_trt import DynamicRanges, EngineProfile, ExportPlan, export_plan_onnx, plan_export
from monopriors.third_party.xlens.models.net import XLensNet


@dataclass(frozen=True, slots=True)
class WorkerArgs:
    """One export request handed over by ``XLensTrtPredictor``."""

    rig_path: Path
    """``.npz`` with ``rays``, ``cam_types``, and optionally ``cam_T_ref``."""
    onnx_path: Path
    """Destination ONNX path."""
    checkpoint: Path
    """Released X-Lens safetensors."""
    profile: EngineProfile
    """Engine profile."""
    max_batch_size: int
    """``rig`` profile batch maximum."""
    opt_batch_size: int
    """``rig`` profile tuning batch."""
    dynamic_views: tuple[int, int]
    """Dynamic view-count range."""
    dynamic_height: tuple[int, int]
    """Dynamic image-height range."""
    dynamic_width: tuple[int, int]
    """Dynamic image-width range."""
    dynamic_max_batch_size: int
    """Dynamic profile batch maximum."""


def main(args: WorkerArgs) -> None:
    """Rebuild the predictor's export plan and write the ONNX file."""
    with np.load(args.rig_path) as rig:
        rays: Float32[ndarray, "s h w 3"] = rig["rays"]
        cam_types: Int64[ndarray, "s"] = rig["cam_types"]
        poses: Float64[ndarray, "n 4 4"] = rig["cam_T_ref"]
        cam_T_ref: Float64[ndarray, "s 4 4"] | None = None if poses.shape[0] == 0 else poses
    loaded: tuple[XLensNet, Path] = load_xlens_model(args.checkpoint, "cuda")
    plan: ExportPlan = plan_export(
        loaded[0],
        rays,
        cam_types,
        cam_T_ref,
        profile=args.profile,
        max_batch_size=args.max_batch_size,
        opt_batch_size=args.opt_batch_size,
        dynamic_ranges=DynamicRanges(
            views=args.dynamic_views,
            patch_rows=(args.dynamic_height[0] // 14, args.dynamic_height[1] // 14),
            patch_cols=(args.dynamic_width[0] // 14, args.dynamic_width[1] // 14),
            batch=(1, args.dynamic_max_batch_size),
        ),
    )
    export_plan_onnx(loaded[0], plan, args.onnx_path)


if __name__ == "__main__":
    main(tyro.cli(WorkerArgs))
