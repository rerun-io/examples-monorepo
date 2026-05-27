"""Build WiLoR TensorRT engines from exported ONNX graphs."""

import tyro

from wilor_nano.api.tensorrt_conversion import TensorRtBuildConfig, build_wilor_tensorrt_engine

if __name__ == "__main__":
    summary = build_wilor_tensorrt_engine(tyro.cli(TensorRtBuildConfig))
    print(
        f"built target={summary.target} engine={summary.engine_path} manifest={summary.manifest_path} "
        f"precision={summary.precision} static_batch={summary.static_batch_size}"
    )
