"""Export KeyNet-F alone (pinch head, no visibility head) to the three ONNX graphs robocap-live loads, checked against PyTorch on
crops from the frozen KeyNet holdout (``data.keynet_holdout``).

Graphs (``export_nets_onnx.KeyNetGraph``): ``keynet.onnx`` (dynamic batch, ONNX Runtime) and ``keynet_b1.onnx`` / ``keynet_b4.onnx``
(static, the RKNN converter's sources). The reference rows, a seeded draw of the deployed variant balanced over datasets, go out as
``keynet_heldout.parquet``: inputs as fed (about half with a zero prior, following stored, zero, zero, stored, ... within a dataset so
that ``rknn_convert``'s even rows (scored) and odd rows (INT8 calibration) both hold both priors), labels (``points_crop`` is the
ground truth), and PyTorch's FP32 outputs with their decoding (``decoded_points_crop``, ``peak``, ``d_rel_mm``).
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
from handtrack.data.keynet_holdout import DATASETS_KEY, KeyNetHoldout, read_holdout
from handtrack.data.stream import KeyNetSamples, select_samples
from handtrack.models.keynet import KeyNetF, keynet_from_state

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
    tracked_error_crop_px: float | None
    """Mean |decode(PyTorch) - ground truth| over positives fed a prior, in 96-px crop pixels; None without any."""
    zero_error_crop_px: float | None
    """The same over positives fed a zero prior."""


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
    """The frozen KeyNet holdout Parquet (``tools/keynet_holdout.py``)."""
    out_dir: Path
    opset: int = 17
    reference_crops: int = 512
    """Rows drawn from the holdout's deployed variant, balanced over its datasets."""
    seed: int = 0
    threads: int = 8
    """CPU threads for PyTorch and ONNX Runtime."""
    tolerance: float = 1e-3
    """See ``KeyNetExportReport.tolerance``."""


def sample_rows(holdout: KeyNetHoldout, count: int, seed: int) -> tuple[Int64[ndarray, "m"], Bool[ndarray, "m"]]:
    """Holdout row indices of a seeded draw of ``count`` deployed-variant crops, balanced over datasets (a dataset with fewer gives
    them all and the others share the rest), grouped by dataset, and which of them get a zero prior (stored, zero, zero, stored, ...)."""
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
    state, sha256 = read_checked_state(config.weights)
    model: KeyNetF = keynet_from_state(state).eval()
    graph: KeyNetGraph = KeyNetGraph(model)  # raises unless the weights have a pinch head and no visibility head
    holdout: KeyNetHoldout = read_holdout(config.holdout)
    rows, zeroed = sample_rows(holdout, config.reference_crops, config.seed)
    index: Int64[Tensor, "n"] = torch.from_numpy(rows)
    samples: KeyNetSamples = select_samples(holdout.samples, index)
    crops: UInt8[ndarray, "n 96 96"] = samples.crops.numpy()
    keypoints: Float32[ndarray, "n 63"] = samples.keypoints.numpy().copy()
    keypoints[zeroed] = 0.0
    # A row is a zero-prior row when the prior it is fed is zero: the zeroed ones and those the holdout stores without a prior.
    zero_prior: Bool[ndarray, "n"] = ~keypoints.any(axis=-1)
    reference: dict[str, ndarray] = torch_keynet(model, crops, keypoints)
    arrays: dict[str, ndarray] = {
        "crops_u8": crops, "keypoints": keypoints, "zero_prior": zero_prior, "dataset": samples.dataset.numpy(), "camera": holdout.camera[index].numpy(),
        "kind": samples.kind.numpy(), "points_crop": samples.points_crop.numpy(), "crop_from_net": samples.crop_from_net.numpy(),
        **{name: reference[name] for name in OUTPUTS}, "decoded_points_crop": reference["points_crop"], "peak": reference["peak"], "d_rel_mm": reference["d_rel_mm"],
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_arrays(config.out_dir / "keynet_heldout.parquet", arrays, {DATASETS_KEY: "\n".join(holdout.dataset_names).encode()})

    crop_feed: Float32[ndarray, "n 1 96 96"] = (crops.astype(np.float32) / 255.0)[:, None]
    targets: list[ndarray] = [reference["heatmaps"], reference["distance"], reference["presence_logit"][:, None], reference["pinch_logit"][:, None]]
    wrap: Int64[ndarray, "4"] = np.arange(4) % len(crops)
    example: tuple[Tensor, Tensor] = (torch.from_numpy(crop_feed[wrap]), torch.from_numpy(keypoints[wrap]))
    graphs: list[GraphRecord] = []
    for batch in STATIC_BATCHES:
        path: Path = config.out_dir / f"keynet_b{batch}.onnx"
        export_graph(graph, (example[0][:batch], example[1][:batch]), path, INPUTS, OUTPUTS, False, config.opset)
        graphs.append(check_graph(path, [crop_feed, keypoints], targets, batch, OUTPUTS, config.threads))
    path = config.out_dir / "keynet.onnx"
    export_graph(graph, example, path, INPUTS, OUTPUTS, True, config.opset)
    graphs.append(check_graph(path, [crop_feed, keypoints], targets, None, OUTPUTS, config.threads))

    report: KeyNetExportReport = KeyNetExportReport(str(config.weights), sha256, str(config.holdout), config.seed, torch.__version__, ort.__version__,
                                                    config.opset, config.tolerance, dataset_rows(holdout.dataset_names, arrays), graphs)
    (config.out_dir / "export_report.json").write_text(to_json(report) + "\n")
    print(f"{config.weights} ({sha256[:8]}): {len(crops)} reference rows, {int(zero_prior.sum())} with a zero prior -> {config.out_dir}")
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
