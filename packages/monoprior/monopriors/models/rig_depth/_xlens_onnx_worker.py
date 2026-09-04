"""Clean-interpreter ONNX export for the X-Lens TensorRT predictor.

Dynamic export dims reach the owned fork as ``torch.SymInt`` values, which the
dev-mode beartype ``int`` checks reject; ``XLensTrtPredictor`` re-enters the
identical export here with ``PIXI_DEV_MODE=0`` (the MoGe v2 pattern).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
    settings_path: Path
    """JSON with ``profile``, ``opt_batch_size``, and the ``DynamicRanges`` fields."""
    onnx_path: Path
    """Destination ONNX path."""
    checkpoint: Path
    """Released X-Lens safetensors."""


def main(args: WorkerArgs) -> None:
    """Rebuild the predictor's export plan and write the ONNX file."""
    with np.load(args.rig_path) as rig:
        rays: Float32[ndarray, "s h w 3"] = rig["rays"]
        cam_types: Int64[ndarray, "s"] = rig["cam_types"]
        poses: Float64[ndarray, "n 4 4"] = rig["cam_T_ref"]
        cam_T_ref: Float64[ndarray, "s 4 4"] | None = None if poses.shape[0] == 0 else poses
    settings: dict[str, Any] = json.loads(args.settings_path.read_text())
    ranges = DynamicRanges(**{name: (int(low), int(high)) for name, (low, high) in settings["ranges"].items()})
    loaded: tuple[XLensNet, Path] = load_xlens_model(args.checkpoint, "cuda")
    plan: ExportPlan = plan_export(
        loaded[0], rays, cam_types, cam_T_ref, profile=cast(EngineProfile, settings["profile"]), opt_batch_size=int(settings["opt_batch_size"]), dynamic_ranges=ranges
    )
    export_plan_onnx(loaded[0], plan, args.onnx_path)


if __name__ == "__main__":
    main(tyro.cli(WorkerArgs))
