"""Export KeyNet-F alone (pinch head, no visibility head) to the three ONNX graphs robocap-live loads, and check them against PyTorch
on crops from the frozen KeyNet holdout (``data.keynet_holdout``).

Graphs (``export_nets_onnx.KeyNetGraph``, TorchScript exporter): ``keynet.onnx`` (dynamic batch ``b``, for ONNX Runtime) and
``keynet_b1.onnx`` / ``keynet_b4.onnx`` (static, the RKNN converter's sources). Inputs ``crop`` f32 [b, 1, 96, 96] in [0, 1] and
``keypoints`` f32 [b, 63]; outputs ``heatmaps`` [b, 21, 18, 18], ``distance`` [b, 21, 18], ``presence_logit`` [b, 1] and
``pinch_logit`` [b, 1].

The reference rows are a seeded draw of the holdout's deployed variant (variant 0), balanced over its datasets; about half of each
dataset's rows get a zero prior (acquisition, as after a DetNet detection), the rest the stored clean prior (tracked). They go out
beside the graphs as ``keynet_heldout.parquet`` (``data.array_table``): the inputs as fed, the labels (``points_crop`` is the ground
truth), the PyTorch FP32 outputs and their decoding (``decoded_points_crop``, ``peak``, ``d_rel_mm``), and
``keynet_heldout.datasets.txt`` names the ``dataset`` indices. ``robocap-live/tools/rknn_convert.py`` scores the RKNN builds on the
even rows and calibrates INT8 on the odd ones; the priors follow stored, zero, zero, stored, ... within a dataset so that both
halves hold both priors.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json
from torch import Tensor

from handtrack.apis.export_nets_onnx import GraphRecord, KeyNetGraph, check_graph, export_graph, read_checked_state, torch_keynet
from handtrack.data.array_table import write_arrays
from handtrack.data.batches import CropKind
from handtrack.data.keynet_holdout import KeyNetHoldout, read_holdout
from handtrack.data.stream import KeyNetSamples, select_samples
from handtrack.models.keynet import KeyNetF, keynet_for_state

INPUTS: list[str] = ["crop", "keypoints"]
OUTPUTS: list[str] = ["heatmaps", "distance", "presence_logit", "pinch_logit"]
STATIC_BATCHES: tuple[int, ...] = (1, 4)
"""``keynet_b<batch>.onnx`` for the NPU."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DatasetRows:
    """One dataset's share of the reference rows and PyTorch's ground-truth error on them."""

    name: str
    rows: int
    positives: int
    """Rows of ``CropKind.POSITIVE``: the keypoint errors are taken over these."""
    tracked_error_crop_px: float | None
    """Mean |decode(PyTorch) - ground truth| over positives with the stored prior, in 96-px crop pixels; None without any."""
    zero_error_crop_px: float | None
    """The same over positives with a zero prior."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class KeyNetExportReport:
    """What ``export_keynet`` wrote and how closely ONNX Runtime (CPU) reproduces PyTorch FP32 on the reference rows."""

    weights: str
    weights_sha256: str
    holdout: str
    seed: int
    """Seed of the reference draw."""
    torch_version: str
    onnxruntime_version: str
    opset: int
    tolerance: float
    """An output fails when max |ort - torch| exceeds tolerance x max(1, max |torch|)."""
    datasets: list[DatasetRows]
    graphs: list[GraphRecord]


@dataclass(frozen=True, slots=True)
class Config:
    """Export one KeyNet-F weights file (pinch head, no visibility head) to robocap-live's ONNX graphs and check them on the holdout."""

    weights: Path
    """``*.weights.pt`` state dict with its ``.sha256`` sidecar."""
    holdout: Path
    """The frozen KeyNet holdout Parquet (``tools/keynet_holdout.py``), with its ``.datasets.txt``."""
    out_dir: Path
    """Receives the three ``.onnx`` graphs, ``keynet_heldout.parquet`` (+ ``.datasets.txt``) and ``export_report.json``."""
    opset: int = 17
    reference_crops: int = 512
    """Rows drawn from the holdout's deployed variant, balanced over its datasets."""
    seed: int = 0
    threads: int = 8
    """CPU threads for PyTorch and ONNX Runtime."""
    tolerance: float = 1e-3
    """See ``KeyNetExportReport.tolerance``."""


def sample_rows(holdout: KeyNetHoldout, count: int, seed: int) -> tuple[Int64[ndarray, "m"], Bool[ndarray, "m"]]:
    """Holdout row indices of a seeded draw of ``count`` deployed-variant crops, balanced over datasets, and which of them get a zero prior.

    Each dataset with deployed rows gets an equal share; one that has fewer gives them all and the others share the rest. Rows are
    grouped by dataset, in holdout order within one; the priors follow stored, zero, zero, stored, ... within each dataset.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    deployed: Bool[ndarray, "n"] = holdout.variant.numpy() == 0
    dataset: Int64[ndarray, "n"] = holdout.samples.dataset.numpy()
    candidates: dict[int, Int64[ndarray, "c"]] = {d: np.flatnonzero(deployed & (dataset == d)) for d in range(len(holdout.dataset_names))}
    candidates = {d: rows for d, rows in candidates.items() if len(rows)}
    if not candidates:
        raise ValueError("the holdout has no deployed-variant (variant 0) crops")
    remaining: int = count
    picked: dict[int, Int64[ndarray, "k"]] = {}
    for left, d in zip(range(len(candidates), 0, -1), sorted(candidates, key=lambda d: len(candidates[d])), strict=True):
        take: int = min(len(candidates[d]), remaining // left)
        picked[d] = np.sort(rng.permutation(candidates[d])[:take])
        remaining -= take
    rows: Int64[ndarray, "m"] = np.concatenate([picked[d] for d in sorted(picked)])
    zero: Bool[ndarray, "m"] = np.concatenate([np.isin(np.arange(len(picked[d])) % 4, (1, 2)) for d in sorted(picked)])
    return rows, zero


def dataset_rows(names: tuple[str, ...], arrays: dict[str, ndarray]) -> list[DatasetRows]:
    """Per dataset with rows: counts and PyTorch's mean ground-truth keypoint error, stored prior vs zero prior."""
    error: Float32[ndarray, "n"] = np.linalg.norm(arrays["decoded_points_crop"] - arrays["points_crop"], axis=-1).mean(axis=-1)
    positive: Bool[ndarray, "n"] = arrays["kind"] == int(CropKind.POSITIVE)
    out: list[DatasetRows] = []
    for d, name in enumerate(names):
        chosen: Bool[ndarray, "n"] = arrays["dataset"] == d
        if not chosen.any():
            continue
        tracked: Bool[ndarray, "n"] = chosen & positive & ~arrays["zero_prior"]
        zero: Bool[ndarray, "n"] = chosen & positive & arrays["zero_prior"]
        out.append(DatasetRows(name, int(chosen.sum()), int((chosen & positive).sum()), float(error[tracked].mean()) if tracked.any() else None,
                               float(error[zero].mean()) if zero.any() else None))
    return out


def main(config: Config) -> KeyNetExportReport:
    """Export the three graphs, write the reference rows and the report, print a summary; raise if ONNX Runtime drifts from PyTorch."""
    torch.set_num_threads(config.threads)
    loaded: tuple[dict[str, Tensor], str] = read_checked_state(config.weights)
    state: dict[str, Tensor] = loaded[0]
    sha256: str = loaded[1]
    model: KeyNetF = keynet_for_state(state)
    if model.pinch_head is None or model.visibility_head is not None:
        raise ValueError(f"{config.weights}: robocap-live's KeyNet graph needs a pinch head and no visibility head "
                         f"(these weights: pinch head {model.pinch_head is not None}, visibility head {model.visibility_head is not None})")
    model.load_state_dict(state)
    model = model.eval()
    holdout: KeyNetHoldout = read_holdout(config.holdout)
    picked: tuple[Int64[ndarray, "n"], Bool[ndarray, "n"]] = sample_rows(holdout, config.reference_crops, config.seed)
    index: Int64[Tensor, "n"] = torch.from_numpy(picked[0])
    samples: KeyNetSamples = select_samples(holdout.samples, index)
    crops: UInt8[ndarray, "n 96 96"] = samples.crops.numpy()
    keypoints: Float32[ndarray, "n 63"] = samples.keypoints.numpy().copy()
    keypoints[picked[1]] = 0.0
    reference: dict[str, ndarray] = torch_keynet(model, crops, keypoints)
    arrays: dict[str, ndarray] = {
        "crops_u8": crops, "keypoints": keypoints, "zero_prior": picked[1], "dataset": samples.dataset.numpy(), "camera": holdout.camera[index].numpy(),
        "kind": samples.kind.numpy(), "points_crop": samples.points_crop.numpy(), "crop_from_net": samples.crop_from_net.numpy(),
        **{name: reference[name] for name in OUTPUTS}, "decoded_points_crop": reference["points_crop"], "peak": reference["peak"], "d_rel_mm": reference["d_rel_mm"],
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_arrays(config.out_dir / "keynet_heldout.parquet", arrays)
    (config.out_dir / "keynet_heldout.datasets.txt").write_text("\n".join(holdout.dataset_names) + "\n")

    crop_feed: Float32[ndarray, "n 1 96 96"] = (crops.astype(np.float32) / 255.0)[:, None]
    targets: list[ndarray] = [reference["heatmaps"], reference["distance"], reference["presence_logit"][:, None], reference["pinch_logit"][:, None]]
    wrap: Int64[ndarray, "4"] = np.arange(4) % len(crops)
    example: tuple[Tensor, Tensor] = (torch.from_numpy(crop_feed[wrap]), torch.from_numpy(keypoints[wrap]))
    graphs: list[GraphRecord] = []
    for batch in STATIC_BATCHES:
        path: Path = config.out_dir / f"keynet_b{batch}.onnx"
        export_graph(KeyNetGraph(model), (example[0][:batch], example[1][:batch]), path, INPUTS, OUTPUTS, False, config.opset)
        graphs.append(check_graph(path, [crop_feed, keypoints], targets, batch, OUTPUTS, config.threads))
    path = config.out_dir / "keynet.onnx"
    export_graph(KeyNetGraph(model), example, path, INPUTS, OUTPUTS, True, config.opset)
    graphs.append(check_graph(path, [crop_feed, keypoints], targets, None, OUTPUTS, config.threads))

    report: KeyNetExportReport = KeyNetExportReport(str(config.weights), sha256, str(config.holdout), config.seed, torch.__version__, ort.__version__,
                                                    config.opset, config.tolerance, dataset_rows(holdout.dataset_names, arrays), graphs)
    (config.out_dir / "export_report.json").write_text(to_json(report) + "\n")
    print(f"{config.weights} ({sha256[:8]}): {len(crops)} reference rows, {int(picked[1].sum())} with a zero prior -> {config.out_dir}")
    for row in report.datasets:
        errors: str = " / ".join("-" if value is None else f"{value:.2f}" for value in (row.tracked_error_crop_px, row.zero_error_crop_px))
        print(f"  {row.name:32s} {row.rows:4d} rows, {row.positives:4d} positives, PyTorch GT error tracked / zero prior {errors} crop px")
    failures: list[str] = []
    for record in graphs:
        print(record.file, record.sha256[:8], record.inputs, "->", record.outputs)
        for check in record.checks:
            print(f"  {check.name:15s} max|ort - torch| = {check.max_abs_diff:.3g} (max |torch| {check.max_abs_value:.3g}) over {record.checked_samples}")
            if check.max_abs_diff > config.tolerance * max(1.0, check.max_abs_value):
                failures.append(f"{record.file} {check.name}: {check.max_abs_diff:.3g}")
    if failures:
        raise ValueError(f"ONNX Runtime differs from PyTorch beyond the tolerance {config.tolerance}: {', '.join(failures)}")
    return report
