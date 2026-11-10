"""DetNet-F alone to ONNX: both graphs match PyTorch on held-out frames, and the references go out as Parquet, not npz."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime", reason="needs onnxruntime")
import torch  # noqa: E402

from handtrack.apis.export_detnet import ExportDetNetConfig, export  # noqa: E402
from handtrack.data.array_table import read_arrays  # noqa: E402
from handtrack.models.detnet import DetNetF  # noqa: E402
from handtrack.train.checkpoint import export_weights  # noqa: E402


def test_both_graphs_match_pytorch_and_the_references_are_parquet(tmp_path: Path) -> None:
    torch.manual_seed(0)
    export_weights(tmp_path / "detnet.weights.pt", DetNetF())
    rng = np.random.default_rng(0)
    pooled, dataset = rng.integers(0, 256, (6, 120, 160), dtype=np.uint8), np.arange(6, dtype=np.int64)
    report = export(ExportDetNetConfig(weights=tmp_path / "detnet.weights.pt", out_dir=tmp_path / "models", threads=2), pooled, dataset)
    assert {record.file for record in report.graphs} == {"detnet_pooled_b1.onnx", "detnet_full.onnx"}
    for record in report.graphs:
        assert record.checked_samples == 6
        assert all(check.max_abs_diff < 1e-4 for check in record.checks)
    references = read_arrays(tmp_path / "models" / "detnet_heldout.parquet")
    assert set(references) == {"pooled_u8", "dataset", "center", "radius", "presence_logit"}
    assert references["center"].shape == (6, 2, 2) and references["presence_logit"].shape == (6, 2)
    assert not list((tmp_path / "models").glob("*.np*"))
