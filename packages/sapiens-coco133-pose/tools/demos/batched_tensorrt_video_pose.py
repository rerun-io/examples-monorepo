from __future__ import annotations

import tyro

from sapiens_coco133_pose.api.batched_tensorrt import BatchedTensorRtVideoConfig, run_batched_tensorrt_video_coco133


def main() -> None:
    """Run the batched TensorRT video pose demo."""
    print(run_batched_tensorrt_video_coco133(tyro.cli(BatchedTensorRtVideoConfig)))


if __name__ == "__main__":
    main()
