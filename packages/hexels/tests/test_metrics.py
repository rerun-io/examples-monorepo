import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hexels_bench.layout import read_json
from hexels_bench.metrics import (
    MetricsFile,
    ViewMetrics,
    psnr_from_sse,
    score_views,
    ssim,
    summarize,
    view_sse,
    write_metrics_file,
)
from hexels_bench.png16 import from_unit_float, read_png16, to_unit_float, write_png16


def test_png16_round_trip_is_exact(tmp_path: Path) -> None:
    rgb: np.ndarray = np.random.default_rng(0).integers(0, 65536, size=(5, 7, 3), dtype=np.uint16)
    write_png16(tmp_path / "a.png", rgb)
    np.testing.assert_array_equal(read_png16(tmp_path / "a.png"), rgb)


def test_unit_float_quantization_clamps_once() -> None:
    values: np.ndarray = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float32)
    np.testing.assert_array_equal(from_unit_float(values), np.array([[[0, 32768, 65535]]], dtype=np.uint16))
    np.testing.assert_allclose(to_unit_float(from_unit_float(values)), np.clip(values, 0, 1), atol=1.0 / 65535)


def test_psnr_matches_upstream_formula() -> None:
    prediction: np.ndarray = np.full((4, 4, 3), 0.5, dtype=np.float32)
    target: np.ndarray = np.full((4, 4, 3), 0.6, dtype=np.float32)
    sse: float = view_sse(prediction, target)
    assert sse == pytest.approx(48 * (np.float64(np.float32(0.6)) - np.float64(np.float32(0.5))) ** 2)
    assert psnr_from_sse(sse, prediction.size) == pytest.approx(-10.0 * math.log10(sse / 48))
    with pytest.raises(ValueError):
        psnr_from_sse(0.0, 48)


def test_ssim_of_identical_images_is_one_and_drops_with_noise() -> None:
    image: torch.Tensor = torch.rand(3, 32, 32, generator=torch.Generator().manual_seed(0))
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    noisy: torch.Tensor = (image + 0.2 * torch.randn(3, 32, 32, generator=torch.Generator().manual_seed(1))).clamp(0, 1)
    assert ssim(noisy, image) < 0.9


def test_score_and_metrics_file_round_trip(tmp_path: Path) -> None:
    rng: np.random.Generator = np.random.default_rng(3)
    for name in ("0000", "0008"):
        target: np.ndarray = rng.integers(0, 65536, size=(16, 16, 3), dtype=np.uint16)
        write_png16(tmp_path / "gt" / f"{name}.png", target)
        write_png16(tmp_path / "renders" / f"{name}.png", np.clip(target.astype(np.int64) + 300, 0, 65535).astype(np.uint16))
    rows: list[ViewMetrics] = score_views(tmp_path / "renders", tmp_path / "gt", ["0000", "0008"], torch.device("cpu"), with_lpips=False)
    assert [row.name for row in rows] == ["0000", "0008"]
    assert all(row.psnr > 40.0 and row.lpips_vgg is None for row in rows)
    metrics: MetricsFile = summarize(
        rows, method="hexels-v1-7k", dataset="dtu", scene="scan24", scorer="shared-png16", fps=None, deploy_mb=1.0, ply_mb=None
    )
    assert metrics.lpips_vgg is None
    assert metrics.psnr == pytest.approx((rows[0].psnr + rows[1].psnr) / 2)
    write_metrics_file(tmp_path / "metrics.json", metrics)
    assert '"fps":null' in (tmp_path / "metrics.json").read_text().replace(" ", "")
    assert read_json(MetricsFile, tmp_path / "metrics.json") == metrics
