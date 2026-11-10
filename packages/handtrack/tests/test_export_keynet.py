"""KeyNet-F alone to ONNX: three graphs that match PyTorch on a balanced draw from the frozen holdout, references as Parquet, not npz."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from jaxtyping import Float32, Int64
from numpy import ndarray
from torch import Tensor

pytest.importorskip("onnxruntime", reason="needs onnxruntime")

from handtrack.apis.export_keynet import OUTPUTS, Config, main, sample_rows  # noqa: E402
from handtrack.data.array_table import read_arrays  # noqa: E402
from handtrack.data.keynet_holdout import DATASETS_KEY, KeyNetHoldout, write_holdout  # noqa: E402
from handtrack.data.stream import KeyNetSamples, empty_keynet_samples  # noqa: E402
from handtrack.models.keynet import KeyNetF  # noqa: E402
from handtrack.train.checkpoint import export_weights  # noqa: E402

DATASETS: tuple[str, ...] = ("dataforge-umetrack", "gsi-six-view")


def synthetic_holdout(rows: int, dataset: Int64[Tensor, "n"], variant: Int64[Tensor, "n"]) -> KeyNetHoldout:
    """Random crops and priors, three positives in four, a scale-and-shift crop affine per row."""
    generator: torch.Generator = torch.Generator().manual_seed(1)
    samples: KeyNetSamples = empty_keynet_samples(rows, torch.device("cpu"))
    samples.crops.copy_(torch.randint(0, 256, (rows, 96, 96), generator=generator, dtype=torch.uint8))
    samples.points_crop.copy_(torch.rand(rows, 21, 2, generator=generator) * 96.0)
    samples.keypoints.copy_(torch.rand(rows, 63, generator=generator) * 96.0)
    samples.kind.copy_(torch.tensor([0, 0, 0, 1]).repeat(rows // 4 + 1)[:rows])
    samples.presence.copy_((samples.kind == 0).float())
    samples.dataset.copy_(dataset)
    affine: Float32[Tensor, "n 3 3"] = torch.eye(3).repeat(rows, 1, 1)
    affine[:, 0, 0] = affine[:, 1, 1] = 0.25 + torch.rand(rows, generator=generator)
    affine[:, :2, 2] = torch.rand(rows, 2, generator=generator) * -100.0
    samples.crop_from_net.copy_(affine)
    return KeyNetHoldout(samples, DATASETS, torch.arange(rows) % 3, variant)


def write_inputs(tmp_path: Path, model: KeyNetF) -> tuple[Path, Path, KeyNetHoldout]:
    rows: int = 16
    holdout: KeyNetHoldout = synthetic_holdout(rows, torch.arange(rows) % 2, (torch.arange(rows) // 2) % 2)
    write_holdout(tmp_path / "holdout" / "keynet_holdout.parquet", holdout)
    export_weights(tmp_path / "keynet.weights.pt", model.eval())
    return tmp_path / "keynet.weights.pt", tmp_path / "holdout" / "keynet_holdout.parquet", holdout


def test_three_graphs_match_pytorch_on_a_balanced_parquet_draw(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model: KeyNetF = KeyNetF(pinch_head=True).eval()
    weights, holdout_path, holdout = write_inputs(tmp_path, model)
    out_dir: Path = tmp_path / "models"
    report = main(Config(weights=weights, holdout=holdout_path, out_dir=out_dir, reference_crops=8, threads=2))

    assert {record.file for record in report.graphs} == {"keynet_b1.onnx", "keynet_b4.onnx", "keynet.onnx"}
    assert all((out_dir / name).exists() for name in ("keynet_b1.onnx", "keynet_b4.onnx", "keynet.onnx"))
    for record in report.graphs:
        assert record.checked_samples == 8
        assert [check.name for check in record.checks] == OUTPUTS
        assert all(check.max_abs_diff < 1e-4 for check in record.checks)
    assert [(row.name, row.rows) for row in report.datasets] == [(DATASETS[0], 4), (DATASETS[1], 4)]

    references: dict[str, ndarray] = read_arrays(out_dir / "keynet_heldout.parquet")
    shapes: dict[str, tuple[int, ...]] = {
        "crops_u8": (8, 96, 96), "keypoints": (8, 63), "zero_prior": (8,), "dataset": (8,), "camera": (8,), "kind": (8,), "points_crop": (8, 21, 2),
        "crop_from_net": (8, 3, 3), "heatmaps": (8, 21, 18, 18), "distance": (8, 21, 18), "presence_logit": (8,), "pinch_logit": (8,),
        "decoded_points_crop": (8, 21, 2), "peak": (8, 21), "d_rel_mm": (8, 21),
    }
    assert {name: value.shape for name, value in references.items()} == shapes
    assert references["crops_u8"].dtype == np.uint8 and references["keypoints"].dtype == np.float32
    assert pq.read_schema(out_dir / "keynet_heldout.parquet").metadata[DATASETS_KEY].decode().split() == list(DATASETS)

    # Every row is a deployed-variant crop with its own labels; half of each dataset's rows get a zero prior, the rest the stored one.
    deployed: ndarray = np.flatnonzero(holdout.variant.numpy() == 0)
    by_bytes: dict[bytes, int] = {holdout.samples.crops[i].numpy().tobytes(): int(i) for i in deployed}
    source: list[int] = [by_bytes[crop.tobytes()] for crop in references["crops_u8"]]
    np.testing.assert_array_equal(references["points_crop"], holdout.samples.points_crop.numpy()[source])
    np.testing.assert_array_equal(references["crop_from_net"], holdout.samples.crop_from_net.numpy()[source])
    np.testing.assert_array_equal(references["dataset"], holdout.samples.dataset.numpy()[source])
    np.testing.assert_array_equal(references["camera"], holdout.camera.numpy()[source])
    zero: ndarray = references["zero_prior"]
    assert [int(zero[references["dataset"] == d].sum()) for d in (0, 1)] == [2, 2]
    assert not references["keypoints"][zero].any()
    np.testing.assert_array_equal(references["keypoints"][~zero], holdout.samples.keypoints.numpy()[source][~zero])

    crop: Float32[ndarray, "8 1 96 96"] = (references["crops_u8"].astype(np.float32) / 255.0)[:, None]
    with torch.inference_mode():
        expected = model(torch.from_numpy(crop), torch.from_numpy(references["keypoints"]))
    assert expected.pinch_logit is not None
    np.testing.assert_allclose(references["heatmaps"], expected.heatmaps.numpy(), atol=1e-6)
    np.testing.assert_allclose(references["pinch_logit"], expected.pinch_logit.numpy(), atol=1e-6)

    assert not [path for path in tmp_path.rglob("*") if path.suffix in (".npy", ".npz")]


@pytest.mark.parametrize("heads", [{"pinch_head": False}, {"pinch_head": True, "visibility_head": True}])
def test_weights_without_the_deployed_heads_are_refused(tmp_path: Path, heads: dict[str, bool]) -> None:
    weights, holdout_path, _ = write_inputs(tmp_path, KeyNetF(**heads))
    with pytest.raises(ValueError, match="pinch head"):
        main(Config(weights=weights, holdout=holdout_path, out_dir=tmp_path / "models", threads=2))
    assert not (tmp_path / "models").exists()


def test_sample_fills_from_larger_datasets_when_one_runs_short() -> None:
    dataset: Int64[Tensor, "n"] = torch.tensor([0] * 12 + [1] * 2)
    holdout: KeyNetHoldout = synthetic_holdout(14, dataset, torch.zeros(14, dtype=torch.int64))
    rows, zero = sample_rows(holdout, 8, seed=3)
    assert len(rows) == 8 and len(set(rows.tolist())) == 8
    assert [int((dataset.numpy()[rows] == d).sum()) for d in (0, 1)] == [6, 2]
    again, _ = sample_rows(holdout, 8, seed=3)
    np.testing.assert_array_equal(rows, again)
    assert int(zero.sum()) == 4
