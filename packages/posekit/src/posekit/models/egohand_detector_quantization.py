"""FP8 encoder input projections for the released RT-DETR hand detector.

The three encoder input projections run in FP8 with calibrated activation ranges;
every other convolution runs in FP16 and the decoder in FP32.
"""

import hashlib
import math
from pathlib import Path

import onnx
from onnx import TensorProto
from trtkit.artifacts import build_once
from trtkit.onnx_precision import fp16_conv_trunk
from trtkit.onnx_qdq import Fp8Conv, KeepPrecision, QdqPlan, quantize_qdq, save_with_external_data
from trtkit.trt_builder import onnx_content_hash

from posekit.models.calibration_rrd import CalibrationRecording, read_calibration_rrd

DETECTOR_CALIBRATION_MODEL: str = "egohand-detector"
"""Recording model identifier for detector activation ranges."""
FP8_PROJECTIONS: str = "/encoder/input_proj."
"""Node-name prefix of the three encoder input projections that run in FP8."""


def quantize_detector(source: onnx.ModelProto, ranges: dict[str, float]) -> onnx.ModelProto:
    """Use trtkit QDQ for the encoder input projections; retain FP32 decode and original graph I/O.

    The upstream export folds Conv weights into unnamed initializers. Give each
    quantized convolution a stable module name before applying trtkit's explicit plan.
    """
    model: onnx.ModelProto = fp16_conv_trunk(onnx.version_converter.convert_version(source, 21))
    weights: dict[str, TensorProto] = {tensor.name: tensor for tensor in model.graph.initializer}
    modules: dict[str, Fp8Conv | KeepPrecision] = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv" or FP8_PROJECTIONS not in node.name:
            continue
        amax: float | None = ranges.get(node.name)
        if amax is None or not math.isfinite(amax) or amax <= 0.0:
            raise ValueError(f"Missing or invalid detector activation range for {node.name}.")
        module: str = f"detector.conv_{index}"
        tensor: TensorProto = TensorProto()
        tensor.CopyFrom(weights[node.input[1]])
        tensor.name = f"{module}.weight"
        model.graph.initializer.append(tensor)
        node.input[1] = tensor.name
        modules[module] = Fp8Conv(amax * 2.0)
    if not modules:
        raise ValueError("No detector encoder input projections found.")
    # Some exports preserve named linear weights. They must be explicitly kept.
    for tensor in model.graph.initializer:
        if tensor.name.endswith(".weight"):
            modules.setdefault(tensor.name.removeprefix("inner.").removesuffix(".weight"), KeepPrecision())
    # Gemm weights are outside trtkit's MatMul/Conv rewrite; omit unused keep entries.
    used: set[str] = {node.input[1] for node in model.graph.node if node.op_type in ("MatMul", "Conv")}
    modules = {
        name: precision
        for name, precision in modules.items()
        if isinstance(precision, Fp8Conv) or f"{name}.weight" in used or f"inner.{name}.weight" in used
    }
    return quantize_qdq(model, QdqPlan(modules, (), scale_dtype="float16"))


def quantized_detector_path(source: Path, calibration: Path, cache_dir: Path) -> Path:
    """Build once by source bytes, calibration bytes and rewrite version."""
    record: CalibrationRecording = read_calibration_rrd(calibration, DETECTOR_CALIBRATION_MODEL)
    source_hash: str = onnx_content_hash(source, cache_dir=cache_dir)
    if record.properties.get("source_sha256") != source_hash:
        raise ValueError(f"{calibration}: detector weights differ from the calibrated source.")
    digest: str = hashlib.sha256(calibration.read_bytes()).hexdigest()[:16]
    output: Path = cache_dir / f"{source.stem}-{source_hash[:16]}-fp8conv-projections-{digest}-v2.onnx"

    def build() -> None:
        save_with_external_data(quantize_detector(onnx.load(source), record.ranges("input_amax")), output)

    return build_once(output, build)
