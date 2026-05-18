"""Run single-image Sapiens2 pose inference with a TensorRT heatmap backend."""

from __future__ import annotations

import tyro

from sapiens2_pose.api.tensorrt_pose import TensorRtImagePoseConfig, run_tensorrt_image_pose


def main(config: TensorRtImagePoseConfig) -> None:
    """Run TensorRT-backed image pose inference."""
    summary = run_tensorrt_image_pose(config)
    print(f"wrote rrd={summary.rrd_path} artifact={summary.artifact_path} persons={summary.person_count}")


if __name__ == "__main__":
    main(tyro.cli(TensorRtImagePoseConfig))
