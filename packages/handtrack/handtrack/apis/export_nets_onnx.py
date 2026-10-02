"""Export DetNet-F and KeyNet-F (with its pinch head) to ONNX for robocap-live's NPU and ONNX Runtime backends.

Files written to ``out_dir`` (``packages/robocap-live/models/``, gitignored):

- ``detnet_pooled_b1.onnx``: input ``image`` [1, 1, 120, 160], the 640 x 480 BarLetterbox frame average-pooled 4 x 4 on the
  CPU (u8, rounded, as the training cache stores it) and divided by 255. This is the graph the RKNN conversion takes: the
  full-frame graph's 4 x 4 pool costs ~7 ms on the RK3588 NPU, the pooled graph ~1 ms (2026-09-29 study), and the CPU pool of
  307,200 bytes is far cheaper than that difference.
- ``detnet_full.onnx``: input ``image`` [b, 1, 480, 640] (dynamic batch), the frame / 255 with the pool inside the graph: exactly
  what handtrack's ``DetNetDetector`` computes, for ONNX Runtime on hosts.
- ``keynet_b{1,4}.onnx`` (static, for RKNN) and ``keynet.onnx`` (dynamic batch, for ONNX Runtime): inputs ``crop``
  [b, 1, 96, 96] in [0, 1] and ``keypoints`` [b, 63]; outputs ``heatmaps`` [b, 21, 18, 18], ``distance`` [b, 21, 18],
  ``presence_logit`` [b, 1] and ``pinch_logit`` [b, 1].

KeyNet's pooled feature vector is computed as ``ReduceMean(keepdims=1)`` + ``Flatten`` instead of ``mean(dim=(2, 3))``: RKNN's
constant folding rejects the rank-2 ``ReduceMean`` output feeding a ``Gemm`` (2026-09-29 study). The math is the same.

Every graph is checked with ONNX Runtime (CPU) against PyTorch FP32 on the held-out caches (``detnet-test-samples``,
``keynet-test-samples-perspective``); the max abs difference per output goes to ``export_report.json``. The held-out inputs,
their PyTorch outputs and decoded keypoints are saved to ``work_dir`` for ``packages/robocap-live/tools/rknn_convert.py``, and a
few samples to the Rust crate's golden test data.
"""

import hashlib
import io
import json
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
from serde.json import to_json
from torch import Tensor, nn

from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.models.detnet import DetNetF
from handtrack.models.keynet import KeyNetF, keynet_for_state

PACKAGES: Path = Path(__file__).resolve().parents[3]
"""The monorepo's ``packages/`` directory."""


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


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ExportReport:
    """What ``export_nets_onnx`` wrote and how closely ONNX Runtime reproduces PyTorch."""

    detnet_weights: str
    detnet_sha256: str
    keynet_weights: str
    keynet_sha256: str
    torch_version: str
    onnxruntime_version: str
    opset: int
    graphs: list[GraphRecord]


@dataclass(kw_only=True)
class Config:
    """Export DetNet-F and KeyNet-F (pinch head) to ONNX and check them with ONNX Runtime against PyTorch on held-out data."""

    detnet_weights: Path
    """DetNet-F state dict with its ``.sha256`` sidecar."""
    keynet_weights: Path
    """KeyNet-F state dict (pinch head) with its ``.sha256`` sidecar."""
    out_dir: Path = PACKAGES / "robocap-live/models"
    """Where the ``.onnx`` files and ``export_report.json`` go."""
    work_dir: Path
    """Held-out inputs + PyTorch outputs (``*_heldout.npz``) for ``rknn_convert.py``."""
    golden_dir: Path = PACKAGES / "robocap-live/crates/robocap-live/tests/data/nets"
    """Small golden files for the Rust backends' tests."""
    detnet_cache: Path
    """Held-out DetNet samples: ``pooled.npy`` u8 [n, 120, 160]."""
    keynet_cache: Path
    """Held-out KeyNet samples: ``crops.npy`` u8 [n, 96, 96], ``keypoints.npy`` [n, 63], ``kind.npy``."""
    keynet_batches: tuple[int, ...] = (1, 4)
    """Static KeyNet batch sizes for the NPU."""
    golden_samples: int = 4
    """Samples per net in the golden files."""
    opset: int = 17
    threads: int = 8
    """CPU threads for PyTorch and ONNX Runtime."""


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


def write_golden(config: Config, detnet_index: ndarray, keynet_index: ndarray, pooled: UInt8[ndarray, "n 120 160"],
                 detnet_reference: tuple[ndarray, ndarray, ndarray], crops: UInt8[ndarray, "m 96 96"], keypoints: Float32[ndarray, "m 63"],
                 keynet_reference: dict[str, ndarray], report: ExportReport) -> None:
    """Little-endian raw files + ``manifest.json``: what the Rust tests feed each backend and the PyTorch outputs they expect.

    DetNet frames are stored pooled (u8 [k, 120, 160]); the Rust test expands each pixel to 4 x 4 to make the 640 x 480 frame
    the ``HandNets`` trait takes, whose 4 x 4 average pool gives back exactly the stored pooled frame.
    """
    config.golden_dir.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(pooled[detnet_index]).tofile(config.golden_dir / "detnet_pooled_u8.bin")
    detnet_out: Float32[ndarray, "k 8"] = np.concatenate([part[detnet_index].reshape(len(detnet_index), -1) for part in detnet_reference], axis=1)
    detnet_out.astype("<f4").tofile(config.golden_dir / "detnet_out_f32.bin")
    np.ascontiguousarray(crops[keynet_index]).tofile(config.golden_dir / "keynet_crops_u8.bin")
    keypoints[keynet_index].astype("<f4").tofile(config.golden_dir / "keynet_keypoints_f32.bin")
    keynet_out: Float32[ndarray, "k 7184"] = np.concatenate(
        [keynet_reference[name][keynet_index].reshape(len(keynet_index), -1) for name in ("heatmaps", "distance", "presence_logit", "pinch_logit")], axis=1
    )
    keynet_out.astype("<f4").tofile(config.golden_dir / "keynet_out_f32.bin")
    keynet_reference["points_crop"][keynet_index].astype("<f4").tofile(config.golden_dir / "keynet_points_crop_f32.bin")
    manifest: dict = {
        "format": "robocap-live-nets-golden/1",
        "detnet_weights_sha256": report.detnet_sha256,
        "keynet_weights_sha256": report.keynet_sha256,
        "detnet": {"samples": len(detnet_index), "source_indices": detnet_index.tolist(), "pooled_u8": "detnet_pooled_u8.bin [k,120,160]",
                   "out_f32": "detnet_out_f32.bin [k,8] = center (l.x, l.y, r.x, r.y), radius (l, r), presence_logit (l, r)"},
        "keynet": {"samples": len(keynet_index), "source_indices": keynet_index.tolist(), "crops_u8": "keynet_crops_u8.bin [k,96,96] (crop = u8 / 255)",
                   "keypoints_f32": "keynet_keypoints_f32.bin [k,63]",
                   "out_f32": "keynet_out_f32.bin [k,7184] = heatmaps 21*18*18, distance 21*18, presence_logit, pinch_logit",
                   "points_crop_f32": "keynet_points_crop_f32.bin [k,21,2] = decode_heatmaps(heatmaps) in crop px"},
    }
    (config.golden_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(config: Config) -> ExportReport:
    """Export, check against PyTorch, save the held-out references and the golden files; print and return the report."""
    torch.set_num_threads(config.threads)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    detnet_state, detnet_sha256 = read_checked_state(config.detnet_weights)
    detnet: DetNetF = DetNetF()
    detnet.load_state_dict(detnet_state)
    detnet = detnet.eval()
    state, keynet_sha256 = read_checked_state(config.keynet_weights)
    keynet: KeyNetF = keynet_for_state(state)
    keynet.load_state_dict(state)
    keynet = keynet.eval()

    pooled: UInt8[ndarray, "n 120 160"] = np.load(config.detnet_cache / "pooled.npy")
    crops: UInt8[ndarray, "m 96 96"] = np.load(config.keynet_cache / "crops.npy")
    keypoints: Float32[ndarray, "m 63"] = np.load(config.keynet_cache / "keypoints.npy")
    detnet_reference: tuple[Float32[ndarray, "n 2 2"], Float32[ndarray, "n 2"], Float32[ndarray, "n 2"]] = torch_detnet(detnet, pooled)
    keynet_reference: dict[str, ndarray] = torch_keynet(keynet, crops, keypoints)
    with torch.inference_mode():
        graph_heatmaps: Float32[Tensor, "8 21 18 18"] = KeyNetGraph(keynet)(
            torch.from_numpy(crops[:8].astype(np.float32) / 255.0)[:, None], torch.from_numpy(keypoints[:8])
        )[0]
    if not torch.allclose(graph_heatmaps, torch.from_numpy(keynet_reference["heatmaps"][:8]), atol=1e-6):
        raise ValueError("KeyNetGraph does not reproduce KeyNetF.forward")

    np.savez(config.work_dir / "detnet_heldout.npz", pooled_u8=pooled, center=detnet_reference[0], radius=detnet_reference[1],
             presence_logit=detnet_reference[2], dataset=np.load(config.detnet_cache / "dataset.npy"))
    np.savez(config.work_dir / "keynet_heldout.npz", crops_u8=crops, keypoints=keypoints, kind=np.load(config.keynet_cache / "kind.npy"),
             dataset=np.load(config.keynet_cache / "dataset.npy"), **keynet_reference)  # pyrefly: ignore  # bad-argument-type: array kwargs, not allow_pickle

    pooled_feed: Float32[ndarray, "n 1 120 160"] = (pooled.astype(np.float32) / 255.0)[:, None]
    full_feed: Float32[ndarray, "n 1 480 640"] = np.repeat(np.repeat(pooled_feed, 4, axis=2), 4, axis=3)
    crop_feed: Float32[ndarray, "m 1 96 96"] = (crops.astype(np.float32) / 255.0)[:, None]
    detnet_names: list[str] = ["center", "radius", "presence_logit"]
    keynet_names: list[str] = ["heatmaps", "distance", "presence_logit", "pinch_logit"]
    keynet_targets: list[ndarray] = [keynet_reference["heatmaps"], keynet_reference["distance"], keynet_reference["presence_logit"][:, None],
                                     keynet_reference["pinch_logit"][:, None]]
    records: list[GraphRecord] = []

    path: Path = config.out_dir / "detnet_pooled_b1.onnx"
    export_graph(DetNetGraph(detnet, pooled=True), (torch.from_numpy(pooled_feed[:1]),), path, ["image"], detnet_names, False, config.opset)
    records.append(check_graph(path, [pooled_feed], list(detnet_reference), 1, detnet_names, config.threads))
    path = config.out_dir / "detnet_full.onnx"
    export_graph(DetNetGraph(detnet, pooled=False), (torch.from_numpy(full_feed[:2]),), path, ["image"], detnet_names, True, config.opset)
    records.append(check_graph(path, [full_feed], list(detnet_reference), None, detnet_names, config.threads))
    del full_feed
    example: tuple[Tensor, Tensor] = (torch.from_numpy(crop_feed[:4]), torch.from_numpy(keypoints[:4]))
    for batch in config.keynet_batches:
        path = config.out_dir / f"keynet_b{batch}.onnx"
        export_graph(KeyNetGraph(keynet), (example[0][:batch], example[1][:batch]), path, ["crop", "keypoints"], keynet_names, False, config.opset)
        records.append(check_graph(path, [crop_feed, keypoints], keynet_targets, batch, keynet_names, config.threads))
    path = config.out_dir / "keynet.onnx"
    export_graph(KeyNetGraph(keynet), example, path, ["crop", "keypoints"], keynet_names, True, config.opset)
    records.append(check_graph(path, [crop_feed, keypoints], keynet_targets, None, keynet_names, config.threads))

    report: ExportReport = ExportReport(str(config.detnet_weights), detnet_sha256, str(config.keynet_weights), keynet_sha256, torch.__version__,
                                        ort.__version__, config.opset, records)
    (config.out_dir / "export_report.json").write_text(to_json(report) + "\n")
    detnet_golden: ndarray = pick_golden((detnet_reference[2] > 0).all(axis=1), config.golden_samples)
    keynet_golden: ndarray = pick_golden(np.load(config.keynet_cache / "kind.npy") == 0, config.golden_samples)
    write_golden(config, detnet_golden, keynet_golden, pooled, detnet_reference, crops, keypoints, keynet_reference, report)
    for record in records:
        print(record.file, record.sha256[:8], record.inputs, "->", record.outputs)
        for check in record.checks:
            print(f"  {check.name:15s} max|ort - torch| = {check.max_abs_diff:.3g} (max |torch| {check.max_abs_value:.3g}) over {record.checked_samples}")
    return report
