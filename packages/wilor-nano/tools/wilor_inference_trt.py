"""CLI entrypoint for the optimized batched TensorRT WiLor video path."""

import tyro

from wilor_nano.api.wilor_inference_trt import BatchedTensorRtVideoConfig, run_batched_tensorrt_video

if __name__ == "__main__":
    run_batched_tensorrt_video(tyro.cli(BatchedTensorRtVideoConfig))
