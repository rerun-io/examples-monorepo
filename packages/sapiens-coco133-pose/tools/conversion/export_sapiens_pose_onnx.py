from __future__ import annotations

import tyro
from sapiens2_pose.api.coco133_tensorrt_conversion import export_sapiens_coco133_pose_onnx


def main() -> None:
    """Run the Sapiens COCO-133 ONNX export CLI."""
    print(tyro.cli(export_sapiens_coco133_pose_onnx))


if __name__ == "__main__":
    main()
