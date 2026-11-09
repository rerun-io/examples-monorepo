"""Export DetNet-F alone to ONNX (full frame, dynamic batch; pooled input, batch 1 for the RKNN converter) and check both graphs
against PyTorch on held-out frames.

Held-out frames come from ``sample_heldout`` (the detector's frozen holdout episodes, decoded as the training stream decodes them)
or any Parquet of ``pooled_u8`` + ``dataset`` arrays (``handtrack.data.array_table``). The PyTorch references go out beside the
graphs as ``detnet_heldout.parquet``, which ``robocap-live/tools/rknn_convert.py`` scores the RKNN builds against.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from jaxtyping import Float32, Int64, UInt8
from numpy import ndarray
from serde import serde
from serde.json import to_json

from handtrack.apis.export_nets_onnx import DetNetGraph, GraphRecord, check_graph, export_graph, read_checked_state, torch_detnet
from handtrack.data.array_table import read_arrays, write_arrays
from handtrack.models.detnet import DetNetF

OUTPUTS: list[str] = ["center", "radius", "presence_logit"]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetNetExportReport:
    weights: str
    weights_sha256: str
    torch_version: str
    onnxruntime_version: str
    opset: int
    graphs: list[GraphRecord]


@dataclass(frozen=True, slots=True)
class ExportDetNetConfig:
    """Export one DetNet-F weights file (``*.weights.pt`` with its ``.sha256`` sidecar)."""

    weights: Path
    heldout: Path
    """Parquet with ``pooled_u8`` [n, 120, 160] and ``dataset`` [n] arrays."""
    out_dir: Path
    """Receives the two ``.onnx`` graphs, ``detnet_heldout.parquet`` and ``export_report.json``."""
    opset: int = 17
    threads: int = 8
    """CPU threads for PyTorch and ONNX Runtime."""
    sample_per_dataset: int = 64
    """When ``heldout`` does not exist yet: frames drawn per dataset (``sample_heldout``, needs the catalog and a GPU)."""
    upright: bool = True
    """Sampled frames are upright (``stream.upright``); False for a model trained on the legacy input."""
    catalog_url: str = ""
    """Catalog for sampling; empty uses handtrack's default."""


def sample_heldout(path: Path, per_dataset: int, upright: bool, catalog_url: str, seed: int = 0) -> None:
    """Draw ``per_dataset`` frames from each dataset's evaluation segments (``detector_eval.eval_segments``), decoded, luma'd and
    letterboxed exactly as the training stream does, and write them as ``pooled_u8`` + ``dataset``."""
    import json

    from serde.json import from_json

    from handtrack.apis.detector_eval import DetectorEvalConfig, eval_segments
    from handtrack.apis.train import DetectorHoldout
    from handtrack.data.catalog import CATALOG_URL
    from handtrack.data.rerun_source import nvdec_producer, source_spec

    defaults: DetectorEvalConfig = DetectorEvalConfig(names=("x",), weights=(path,), out_dir=path.parent)
    url: str = catalog_url or CATALOG_URL
    holdout: DetectorHoldout = from_json(DetectorHoldout, defaults.holdout.read_text())
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    pooled: list[ndarray] = []
    labels: list[ndarray] = []
    for index, dataset in enumerate(defaults.datasets):
        segments: list[str] = eval_segments(dataset, holdout, url, 8, seed)  # a few segments are plenty for an export check
        kept: list[ndarray] = []
        for samples in nvdec_producer(source_spec(dataset, index, upright), segments, catalog_url=url, device=torch.device("cuda"))(seed):
            kept.append(samples.pooled.cpu().numpy())
            if sum(len(k) for k in kept) >= 8 * per_dataset:
                break
        frames: UInt8[ndarray, "m 120 160"] = np.concatenate(kept)
        picked: ndarray = torch.randperm(len(frames), generator=generator)[:per_dataset].numpy()
        pooled.append(frames[np.sort(picked)])
        labels.append(np.full(len(picked), index, dtype=np.int64))
        print(f"{dataset}: {len(picked)} frames from {json.dumps(segments[:2])[:80]}...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_arrays(path, {"pooled_u8": np.concatenate(pooled), "dataset": np.concatenate(labels)})


def main(config: ExportDetNetConfig) -> DetNetExportReport:
    torch.set_num_threads(config.threads)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if not config.heldout.exists():
        sample_heldout(config.heldout, config.sample_per_dataset, config.upright, config.catalog_url)
    state, sha256 = read_checked_state(config.weights)
    model: DetNetF = DetNetF()
    model.load_state_dict(state)
    model = model.eval()
    frames: dict[str, ndarray] = read_arrays(config.heldout)
    pooled: UInt8[ndarray, "n 120 160"] = frames["pooled_u8"]
    dataset: Int64[ndarray, "n"] = frames["dataset"]
    references: tuple[Float32[ndarray, "n 2 2"], Float32[ndarray, "n 2"], Float32[ndarray, "n 2"]] = torch_detnet(model, pooled)
    write_arrays(config.out_dir / "detnet_heldout.parquet", {"pooled_u8": pooled, "dataset": dataset, **dict(zip(OUTPUTS, references, strict=True))})
    pooled_feed: Float32[ndarray, "n 1 120 160"] = (pooled.astype(np.float32) / 255.0)[:, None]
    full_feed: Float32[ndarray, "n 1 480 640"] = np.repeat(np.repeat(pooled_feed, 4, axis=2), 4, axis=3)
    graphs: list[GraphRecord] = []
    path: Path = config.out_dir / "detnet_pooled_b1.onnx"
    export_graph(DetNetGraph(model, pooled=True), (torch.from_numpy(pooled_feed[:1]),), path, ["image"], OUTPUTS, False, config.opset)
    graphs.append(check_graph(path, [pooled_feed], list(references), 1, OUTPUTS, config.threads))
    path = config.out_dir / "detnet_full.onnx"
    export_graph(DetNetGraph(model, pooled=False), (torch.from_numpy(full_feed[:2]),), path, ["image"], OUTPUTS, True, config.opset)
    graphs.append(check_graph(path, [full_feed], list(references), None, OUTPUTS, config.threads))
    report: DetNetExportReport = DetNetExportReport(str(config.weights), sha256, torch.__version__, ort.__version__, config.opset, graphs)
    (config.out_dir / "export_report.json").write_text(to_json(report) + "\n")
    for record in graphs:
        print(record.file, record.sha256[:8], record.inputs, "->", record.outputs)
        for check in record.checks:
            print(f"  {check.name:15s} max|ort - torch| = {check.max_abs_diff:.3g} (max |torch| {check.max_abs_value:.3g}) over {record.checked_samples}")
    return report
