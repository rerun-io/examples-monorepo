"""Calibrated FP8 convolutions for the released RT-DETR hand detector."""

import hashlib
import math
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from trtkit.artifacts import build_once
from trtkit.onnx_precision import fp16_conv_trunk
from trtkit.onnx_qdq import Fp8Conv, KeepPrecision, QdqPlan, quantize_qdq, save_with_external_data
from trtkit.trt_builder import onnx_content_hash

from posekit.models.calibration_rrd import CalibrationRecording, read_calibration_rrd
from posekit.models.egohand import DetectorFp8Scope

DETECTOR_CALIBRATION_MODEL: str = "egohand-detector"
"""Recording model identifier for detector activation ranges."""


def detector_activation_model(source: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
    """Expose only each convolution input's absolute maximum for efficient calibration."""
    model: onnx.ModelProto = onnx.ModelProto()
    model.CopyFrom(source)
    names: list[str] = []
    for node in list(model.graph.node):
        if node.op_type != "Conv":
            continue
        prefix: str = f"calibration_{len(names)}"
        model.graph.node.extend(
            [
                helper.make_node("Abs", [node.input[0]], [prefix + "_abs"]),
                helper.make_node("ReduceMax", [prefix + "_abs"], [prefix], keepdims=0),
            ]
        )
        model.graph.output.append(helper.make_tensor_value_info(prefix, TensorProto.FLOAT, []))
        names.append(node.name)
    return model, names


def quantize_detector(source: onnx.ModelProto, ranges: dict[str, float], *, scope: DetectorFp8Scope = "all") -> onnx.ModelProto:
    """Use trtkit QDQ for selected convolutions; retain FP32 decode and original graph I/O.

    The upstream export folds Conv weights into unnamed initializers. Give each
    convolution a stable module name before applying trtkit's explicit plan.
    """
    model: onnx.ModelProto = fp16_conv_trunk(onnx.version_converter.convert_version(source, 21))
    weights: dict[str, TensorProto] = {tensor.name: tensor for tensor in model.graph.initializer}
    modules: dict[str, Fp8Conv | KeepPrecision] = {}
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Conv":
            continue
        if scope == "backbone" and ("/backbone/" not in node.name or "/backbone/conv1/" in node.name):
            continue
        if scope == "late_backbone" and "/backbone/res_layers.3/" not in node.name:
            continue
        if scope == "projections" and "/encoder/input_proj." not in node.name:
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
        raise ValueError(f"No detector convolutions matched FP8 scope {scope}.")
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


def quantized_detector_path(source: Path, calibration: Path, cache_dir: Path, *, scope: DetectorFp8Scope = "all") -> Path:
    """Build once by source bytes, calibration bytes and rewrite version."""
    record: CalibrationRecording = read_calibration_rrd(calibration, DETECTOR_CALIBRATION_MODEL)
    source_hash: str = onnx_content_hash(source, cache_dir=cache_dir)
    if record.properties.get("source_sha256") != source_hash:
        raise ValueError(f"{calibration}: detector weights differ from the calibrated source.")
    digest: str = hashlib.sha256(calibration.read_bytes()).hexdigest()[:16]
    output: Path = cache_dir / f"{source.stem}-{source_hash[:16]}-fp8conv-{scope}-{digest}-v2.onnx"

    def build() -> None:
        save_with_external_data(quantize_detector(onnx.load(source), record.ranges("input_amax"), scope=scope), output)

    return build_once(output, build)
