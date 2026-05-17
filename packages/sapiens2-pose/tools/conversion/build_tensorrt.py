"""Build a TensorRT engine from a Sapiens2 pose ONNX graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from sapiens2_pose.api.runtime import ModelSize
from sapiens2_pose.api.tensorrt_pose import (
    TensorRtBuildConfig,
    build_tensorrt_engine,
)


@dataclass(frozen=True, slots=True)
class BuildTensorRtCli:
    """CLI arguments for building a TensorRT engine."""

    onnx_path: Path
    """Path to the portable ONNX graph."""
    engine_path: Path
    """Path where the machine-local TensorRT engine should be written."""
    model_size: ModelSize = "0.4B"
    """Sapiens2 model size represented by the ONNX graph."""
    workspace_gib: float = 24.0
    """TensorRT workspace memory limit in GiB."""
    builder_optimization_level: int = 3
    """TensorRT builder optimization level, from 0 through 5."""


def main(args: BuildTensorRtCli) -> None:
    """Run the TensorRT engine build CLI."""
    summary = build_tensorrt_engine(
        TensorRtBuildConfig(
            onnx_path=args.onnx_path,
            engine_path=args.engine_path,
            model_size=args.model_size,
            workspace_gib=args.workspace_gib,
            builder_optimization_level=args.builder_optimization_level,
        )
    )
    print(
        f"built engine={summary.engine_path} manifest={summary.manifest_path} "
        f"precision={summary.precision} batch_profile={summary.batch_profile}"
    )


if __name__ == "__main__":
    main(tyro.cli(BuildTensorRtCli))
