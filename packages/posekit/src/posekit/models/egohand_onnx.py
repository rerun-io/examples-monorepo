"""EgoHand DINOv3 NVFP4 linears and FP8 native-attention ONNX export."""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from trtkit.nvfp4 import bf16_round_up, quant_nvfp4_mse

from posekit.models.egohand_quantization import EgoHandCalibration

BF16: int = TensorProto.BFLOAT16


def bf16_scalar(name: str, v: float) -> onnx.TensorProto:
    t: torch.Tensor = torch.tensor([v], dtype=torch.bfloat16)
    return helper.make_tensor(name, BF16, [], vals=t.view(torch.uint16).numpy().tobytes(), raw=True)


def raw_tensor(name: str, dtype: int, dims: list[int], data: torch.Tensor) -> onnx.TensorProto:
    return helper.make_tensor(name, dtype, dims, vals=data.contiguous().cpu().view(torch.uint8).numpy().tobytes(), raw=True)


def quantize_backbone(input_path: Path, output_path: Path, calibration: EgoHandCalibration) -> None:
    """Quantize the released 160 linears and 32 native attention blocks."""
    cal: dict[str, float] = calibration.input_amax
    m: onnx.ModelProto = onnx.load(str(input_path))
    g: onnx.GraphProto = m.graph
    prod: dict[str, onnx.NodeProto] = {o: n for n in g.node for o in n.output}
    inits: dict[str, onnx.TensorProto] = {i.name: i for i in g.initializer}
    new_inits: list[onnx.TensorProto] = []
    new_nodes_before: dict[int, list[onnx.NodeProto]] = {}  # node index -> nodes to insert before it
    act_q: dict[tuple[str, str], str] = {}  # activation tensor -> dequantised name (shared by Q/K/V)

    def weight_of(node: onnx.NodeProto) -> tuple[str, list[str]]:
        x: str = node.input[1]
        chain: list[str] = []
        while x not in inits and x in prod:
            chain.append(prod[x].op_type)
            x = prod[x].input[0]
        return x, chain

    for idx, node in enumerate(list(g.node)):
        if node.op_type != "MatMul":
            continue
        weight_source: tuple[str, list[str]] = weight_of(node)
        wname: str = weight_source[0]
        chain: list[str] = weight_source[1]
        if wname not in inits or not wname.endswith(".weight"):
            continue
        mod: str = wname.removeprefix("inner.").removesuffix(".weight")
        ins: list[onnx.NodeProto] = []
        assert chain == ["Transpose", "Cast"], (wname, chain)
        w: torch.Tensor = torch.from_numpy(numpy_helper.to_array(inits[wname]).copy())  # [N, K] fp32
        w = w.cuda()
        n_out: int = w.shape[0]
        k: int = w.shape[1]
        x = node.input[0]
        amax: float = cal[mod] * 2.0
        s2: float = bf16_round_up(w.abs().amax().item() / (6.0 * 256.0))
        quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = quant_nvfp4_mse(w, tensor_div=1.0 / s2 * w.abs().amax().item())
        wq: torch.Tensor = quantized[0]
        wsc: torch.Tensor = quantized[1]
        new_inits += [
            raw_tensor(f"{mod}.w_fp4", TensorProto.FLOAT4E2M1, [n_out, k], wq),
            raw_tensor(f"{mod}.w_sc", TensorProto.FLOAT8E4M3FN, [n_out, k // 16], wsc),
            bf16_scalar(f"{mod}.w_s2", s2),
        ]
        ins += [
            helper.make_node("DequantizeLinear", [f"{mod}.w_sc", f"{mod}.w_s2"], [f"{mod}.w_scdq"]),
            helper.make_node("DequantizeLinear", [f"{mod}.w_fp4", f"{mod}.w_scdq"], [f"{mod}.w_dq"], axis=-1, block_size=16),
        ]
        key: tuple[str, str] = (x, "fp4")
        if key not in act_q:
            sa: str = f"{mod}.a_s"
            new_inits.append(bf16_scalar(sa, amax / (6.0 * 448.0)))
            ins += [
                helper.make_node(
                    "TRT_FP4DynamicQuantize",
                    [x, sa],
                    [f"{mod}.a_q", f"{mod}.a_sc"],
                    domain="trt.plugins",
                    axis=-1,
                    block_size=16,
                    scale_type=TensorProto.FLOAT8E4M3FN,
                ),
                helper.make_node("DequantizeLinear", [f"{mod}.a_sc", sa], [f"{mod}.a_scdq"]),
                helper.make_node("DequantizeLinear", [f"{mod}.a_q", f"{mod}.a_scdq"], [f"{mod}.a_dq"], axis=-1, block_size=16),
            ]
            act_q[key] = f"{mod}.a_dq"

        ins.append(helper.make_node("Transpose", [f"{mod}.w_dq"], [f"{mod}.w_t"], perm=[1, 0]))
        node.input[0] = act_q[key]
        node.input[1] = f"{mod}.w_t"
        new_nodes_before[idx] = ins

    attention_index: int = 0
    for idx, node in enumerate(list(g.node)):
        if node.op_type != "Attention":
            continue
        ins = new_nodes_before.setdefault(idx, [])
        for j, tag in enumerate("qkv"):
            name: str = f"encoder.blocks.{attention_index}.attn.{tag}"
            scale: str = f"{name}.a_s8"
            zp: str = f"{name}.a_zp"
            new_inits.extend(
                [
                    bf16_scalar(scale, calibration.attention_amax[name] / 448.0),
                    raw_tensor(zp, TensorProto.FLOAT8E4M3FN, [], torch.zeros((), dtype=torch.float8_e4m3fn)),
                ]
            )
            ins.extend(
                [
                    helper.make_node("QuantizeLinear", [node.input[j], scale, zp], [f"{name}.a_q8"], saturate=1),
                    helper.make_node("DequantizeLinear", [f"{name}.a_q8", scale], [f"{name}.a_dq8"]),
                ]
            )
            node.input[j] = f"{name}.a_dq8"
        attention_index += 1
    fp4_weights: int = sum(tensor.data_type == TensorProto.FLOAT4E2M1 for tensor in new_inits)
    if fp4_weights != 160:
        raise ValueError(f"Expected 160 backbone FP4 weights, got {fp4_weights}.")
    if attention_index != 32:
        raise ValueError(f"Expected 32 native Attention ops, got {attention_index}.")
    nodes: list[onnx.NodeProto] = []
    for idx, node in enumerate(g.node):
        nodes.extend(new_nodes_before.get(idx, []))
        nodes.append(node)
    # drop dead nodes / initializers
    g.ClearField("node")
    g.node.extend(nodes)
    g.initializer.extend(new_inits)
    live: set[str] = {o.name for o in g.output}
    keep: list[onnx.NodeProto] = []
    for node in reversed(list(g.node)):
        if any(o in live for o in node.output):
            keep.append(node)
            live.update(i for i in node.input if i)
    keep.reverse()
    g.ClearField("node")
    g.node.extend(keep)
    kept_inits: list[onnx.TensorProto] = [i for i in g.initializer if i.name in live]
    g.ClearField("initializer")
    g.initializer.extend(kept_inits)
    if not any(o.domain == "trt.plugins" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("trt.plugins", 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_name: str = output_path.name + ".data"
    # Publish the model last: its presence means both the graph and weights are complete.
    with TemporaryDirectory(prefix="egohand-quantize-", dir=output_path.parent) as directory:
        staged: Path = Path(directory) / output_path.name
        onnx.save(m, str(staged), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
        (staged.parent / data_name).replace(output_path.parent / data_name)
        staged.replace(output_path)
