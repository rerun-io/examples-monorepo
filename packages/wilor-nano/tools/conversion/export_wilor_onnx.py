"""Export WiLoR TensorRT deployment ONNX graphs."""

import tyro

from wilor_nano.api.tensorrt_conversion import WiLorOnnxExportConfig, export_wilor_onnx

if __name__ == "__main__":
    summary = export_wilor_onnx(tyro.cli(WiLorOnnxExportConfig))
    print(
        f"exported target={summary.target} onnx={summary.onnx_path} "
        f"input={summary.input_name}{summary.input_shape} outputs={summary.output_names}"
    )
