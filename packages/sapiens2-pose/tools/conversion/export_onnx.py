"""Export a Sapiens2 pose model checkpoint to ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from sapiens2_pose.api.image_pose import DeviceChoice, ModelSize
from sapiens2_pose.api.tensorrt_pose import SapiensPoseOnnxExportConfig, export_sapiens_pose_onnx, resolve_sapiens_pose_checkpoint


@dataclass(frozen=True, slots=True)
class ExportOnnxCli:
    """CLI arguments for Sapiens2 pose ONNX export."""

    onnx_path: Path
    """Path where the exported ONNX graph should be written."""
    checkpoint_path: Path | None = None
    """Optional explicit checkpoint path; when omitted the checkpoint is downloaded from Hugging Face."""
    model_size: ModelSize = "0.4B"
    """Sapiens2 pose model size to export."""
    opset_version: int = 17
    """ONNX opset version passed to `torch.onnx.export`."""
    device: DeviceChoice = "cuda"
    """Device used while tracing."""
    dynamo: bool = False
    """Whether to use PyTorch's dynamo ONNX exporter."""


def main(args: ExportOnnxCli) -> None:
    """Run the ONNX export CLI."""
    checkpoint_path: Path = resolve_sapiens_pose_checkpoint(args.model_size, args.checkpoint_path)
    summary = export_sapiens_pose_onnx(
        SapiensPoseOnnxExportConfig(
            checkpoint_path=checkpoint_path,
            onnx_path=args.onnx_path,
            model_size=args.model_size,
            opset_version=args.opset_version,
            device=args.device,
            dynamo=args.dynamo,
        )
    )
    print(
        f"exported onnx={summary.onnx_path} checkpoint={summary.checkpoint_path} "
        f"input_shape={summary.input_shape} output_shape={summary.output_shape}"
    )


if __name__ == "__main__":
    main(tyro.cli(ExportOnnxCli))
