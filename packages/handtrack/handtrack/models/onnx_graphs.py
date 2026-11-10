"""The ONNX graphs robocap-live loads, built from DetNet-F and KeyNet-F, and their check against PyTorch: shared by
``apis.export_detnet`` and ``apis.export_keynet``.

- ``DetNetGraph(pooled=True)``: input ``image`` [1, 1, 120, 160], the 640 x 480 BarLetterbox frame average-pooled 4 x 4 on the CPU
  (u8, rounded) and divided by 255; the graph the RKNN conversion takes (the full-frame graph's 4 x 4 pool costs ~7 ms on the
  RK3588 NPU, the pooled graph ~1 ms; 2026-09-29 study). ``pooled=False``: input [b, 1, 480, 640], the pool inside the graph,
  exactly what handtrack's ``DetNetDetector`` computes, for ONNX Runtime on hosts.
- ``KeyNetGraph``: inputs ``crop`` [b, 1, 96, 96] in [0, 1] and ``keypoints`` [b, 63]; outputs ``heatmaps`` [b, 21, 18, 18],
  ``distance`` [b, 21, 18], ``presence_logit`` [b, 1] and ``pinch_logit`` [b, 1]. Its pooled feature vector is ``ReduceMean
  (keepdims=1)`` + ``Flatten`` instead of ``mean(dim=(2, 3))``: RKNN's constant folding rejects the rank-2 ``ReduceMean`` output
  feeding a ``Gemm`` (2026-09-29 study). The math is the same.
"""

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray
from serde import serde
from torch import Tensor, nn

from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.models.detnet import DetNetF
from handtrack.models.keynet import KeyNetF


def read_checked_state(path: Path) -> tuple[dict[str, Tensor], str]:
    """A state dict and its sha256, checked against the ``<file>.sha256`` sidecar.

    ``handtrack.pipeline.read_state`` does the same but imports the tracker, which needs handfit's compiled core.
    """
    payload: bytes = path.read_bytes()
    digest: str = hashlib.sha256(payload).hexdigest()
    expected: str = Path(f"{path}.sha256").read_text().split()[0]
    if digest != expected:
        raise ValueError(f"{path}: sha256 {digest} does not match its sidecar {expected}")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True), digest


class DetNetGraph(nn.Module):
    """DetNet-F as three output tensors, on the pooled 120 x 160 input or the full 480 x 640 frame."""

    def __init__(self, model: DetNetF, pooled: bool) -> None:
        super().__init__()
        self.model: DetNetF = model
        self.pooled: bool = pooled

    def forward(self, image: Float32[Tensor, "b 1 h w"]) -> tuple[Float32[Tensor, "b 2 2"], Float32[Tensor, "b 2"], Float32[Tensor, "b 2"]]:
        """center (cx / 640, cy / 480), radius / 640 and presence logits, left slot first."""
        output = self.model.forward_pooled(image) if self.pooled else self.model(image)
        return output.center, output.radius, output.presence_logit


class KeyNetGraph(nn.Module):
    """KeyNet-F with the pinch head as four output tensors; the pooled vector is a keepdims mean + flatten (RKNN rank bug)."""

    def __init__(self, model: KeyNetF) -> None:
        super().__init__()
        if model.pinch_head is None or model.visibility_head is not None:
            raise ValueError("robocap-live's KeyNet graph expects a pinch head and no visibility head")
        self.model: KeyNetF = model
        self.pinch_head: nn.Linear = model.pinch_head

    def forward(
        self, crop: Float32[Tensor, "b 1 96 96"], keypoints: Float32[Tensor, "b 63"]
    ) -> tuple[Float32[Tensor, "b 21 18 18"], Float32[Tensor, "b 21 18"], Float32[Tensor, "b 1"], Float32[Tensor, "b 1"]]:
        """The same layers as ``KeyNetF.forward``; presence and pinch logits keep a trailing axis of one."""
        model: KeyNetF = self.model
        image_features: Float32[Tensor, "b 64 12 12"] = model.image(crop)
        keypoint_features: Float32[Tensor, "b 32 12 12"] = model.keypoints(keypoints).unflatten(1, (32, 12, 12))
        fused: Float32[Tensor, "b 160 6 6"] = model.fused(torch.cat((image_features, keypoint_features), dim=1))
        pooled: Float32[Tensor, "b 160"] = fused.mean(dim=(2, 3), keepdim=True).flatten(1)
        return (
            model.heatmap_head(fused),
            model.distance_head(fused).flatten(1).unflatten(1, (21, 18)),
            model.presence_head(pooled),
            self.pinch_head(pooled),
        )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class OutputCheck:
    """ONNX Runtime CPU vs PyTorch FP32 for one output of one graph over a held-out set."""

    name: str
    """Output name."""
    max_abs_diff: float
    """Largest absolute difference over every element of every sample."""
    max_abs_value: float
    """Largest absolute PyTorch value, for scale."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class GraphRecord:
    """One exported ONNX file."""

    file: str
    """File name inside the models directory."""
    sha256: str
    inputs: list[str]
    """``name[shape]`` per input; ``b`` marks a dynamic batch axis."""
    outputs: list[str]
    """``name[shape]`` per output."""
    checked_samples: int
    """Held-out samples run through ONNX Runtime and PyTorch."""
    checks: list[OutputCheck]


def torch_detnet(model: DetNetF, pooled: UInt8[ndarray, "n 120 160"]) -> tuple[Float32[ndarray, "n 2 2"], Float32[ndarray, "n 2"], Float32[ndarray, "n 2"]]:
    """PyTorch FP32 DetNet outputs on u8 pooled frames, in chunks of 64."""
    centers: list[Float32[ndarray, "c 2 2"]] = []
    radii: list[Float32[ndarray, "c 2"]] = []
    logits: list[Float32[ndarray, "c 2"]] = []
    with torch.inference_mode():
        for start in range(0, len(pooled), 64):
            output = model.forward_pooled(torch.from_numpy(pooled[start : start + 64].astype(np.float32) / 255.0)[:, None])
            centers.append(output.center.numpy())
            radii.append(output.radius.numpy())
            logits.append(output.presence_logit.numpy())
    return np.concatenate(centers), np.concatenate(radii), np.concatenate(logits)


def torch_keynet(model: KeyNetF, crops: UInt8[ndarray, "n 96 96"], keypoints: Float32[ndarray, "n 63"]) -> dict[str, ndarray]:
    """PyTorch FP32 KeyNet outputs (raw and decoded) on u8 crops, in chunks of 64; the reference everything is scored against."""
    parts: dict[str, list[ndarray]] = {name: [] for name in ("heatmaps", "distance", "presence_logit", "pinch_logit", "points_crop", "peak", "d_rel_mm")}
    with torch.inference_mode():
        for start in range(0, len(crops), 64):
            crop: Float32[Tensor, "c 1 96 96"] = torch.from_numpy(crops[start : start + 64].astype(np.float32) / 255.0)[:, None]
            output = model(crop, torch.from_numpy(keypoints[start : start + 64]))
            if output.pinch_logit is None:
                raise ValueError("KeyNet has no pinch head")
            decoded: tuple[Float32[Tensor, "c 21 2"], Float32[Tensor, "c 21"]] = decode_heatmaps(output.heatmaps)
            parts["heatmaps"].append(output.heatmaps.numpy())
            parts["distance"].append(output.distance.numpy())
            parts["presence_logit"].append(output.presence_logit.numpy())
            parts["pinch_logit"].append(output.pinch_logit.numpy())
            parts["points_crop"].append(decoded[0].numpy())
            parts["peak"].append(decoded[1].numpy())
            parts["d_rel_mm"].append(decode_distance(output.distance).numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def export_graph(
    graph: nn.Module, example: tuple[Tensor, ...], path: Path, inputs: list[str], outputs: list[str], dynamic_batch: bool, opset: int
) -> None:
    """TorchScript-path ONNX export (the RKNN converter's tested route), then ``onnx.checker``."""
    axes: dict[str, dict[int, str]] | None = {name: {0: "b"} for name in inputs + outputs} if dynamic_batch else None
    graph.eval()  # the exporter restores the wrapper's mode afterwards, recursively: a fresh wrapper would leave the net in training mode
    with torch.inference_mode():
        torch.onnx.export(
            graph, example, str(path), input_names=inputs, output_names=outputs, opset_version=opset, dynamo=False, dynamic_axes=axes,
            external_data=False,
        )
    model: onnx.ModelProto = onnx.load(str(path))
    onnx.checker.check_model(model)


def describe(values: Sequence[ort.NodeArg]) -> list[str]:
    """``name[d0,d1,...]`` for ONNX Runtime inputs or outputs; symbolic axes print their name."""
    return [f"{value.name}[{','.join(str(axis) for axis in value.shape)}]" for value in values]


def check_graph(
    path: Path, feeds: list[ndarray], references: list[ndarray], batch: int | None, names: list[str], threads: int
) -> GraphRecord:
    """Run ``path`` on every held-out sample (static batches wrap the last one) and compare each output with PyTorch."""
    options: ort.SessionOptions = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session: ort.InferenceSession = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    count: int = len(feeds[0])
    step: int = batch if batch is not None else 64
    collected: list[list[ndarray]] = [[] for _ in names]
    for start in range(0, count, step):
        indices: ndarray = np.arange(start, start + step) % count if batch is not None else np.arange(start, min(start + step, count))
        values: list[ndarray] = [np.asarray(value) for value in session.run(names, {item.name: feed[indices] for item, feed in zip(session.get_inputs(), feeds, strict=True)})]
        kept: int = min(step, count - start)
        for slot, value in enumerate(values):
            collected[slot].append(value[:kept])
    checks: list[OutputCheck] = []
    for name, parts, reference in zip(names, collected, references, strict=True):
        result: Float32[ndarray, "..."] = np.concatenate(parts).reshape(reference.shape)
        checks.append(OutputCheck(name, float(np.abs(result - reference).max()), float(np.abs(reference).max())))
    return GraphRecord(path.name, hashlib.sha256(path.read_bytes()).hexdigest(), describe(session.get_inputs()), describe(session.get_outputs()), count, checks)


def pick_golden(present: Bool[ndarray, "n"], count: int) -> ndarray:
    """``count`` indices spread evenly over the samples where ``present`` holds."""
    candidates: ndarray = np.flatnonzero(present)
    return candidates[np.linspace(0, len(candidates) - 1, count).round().astype(np.int64)]
