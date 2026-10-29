"""Comparison rows: TABLE.json cells, the WGSL FPS fallback, probe and speed timing, published numbers, and the logged table."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from gsplat_rust_renderer.hexels_metrics import (
    METRIC_LABEL,
    METRICS_ENTITY,
    BenchTable,
    MetricRow,
    SceneCells,
    log_metric_table,
    paper_row,
    read_json,
    run_row,
    scene_cells,
)

PUBLISHED: dict = {"model": "v1-fixed", "views": 24, "psnr": 24.71, "ssim": 0.728, "lpips_vgg": 0.337, "deploy_mb": 11.10128, "primitives": 138766, "train_min": 20.775}


def row(method: str, **cells: object) -> dict:
    """A TABLE.json scene row as hexels-bench-table writes it (unscored cells null)."""
    empty: dict = dict.fromkeys(("reason", "views", "train_min", "total_min", "vram_gib", "fps", "psnr", "ssim", "lpips_vgg", "deploy_mb", "ply_mb", "primitives"))
    return {"dataset": "mipnerf360", "scene": "garden", "method": method, "status": "complete", "sfm_points": 138766, "paper": None, "delta_psnr": None} | empty | cells


def write(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def table(tmp_path: Path) -> BenchTable:
    rows: list[dict] = [
        row("brush-7k", psnr=24.25, ssim=0.7, lpips_vgg=0.38, train_min=0.5, vram_gib=0.46, deploy_mb=0.236),
        row("brush-7k-wgsl", psnr=24.25, ssim=0.7, lpips_vgg=0.38, fps=554.6, deploy_mb=0.236),
        row("hexels-v1-30k-pub-wgsl", psnr=24.7121, ssim=0.73, lpips_vgg=0.33, fps=900.0, deploy_mb=11.10128, paper=PUBLISHED),
        row("hexels-v1-7k", status="incomplete", reason="no metrics.json", paper=PUBLISHED),
    ]
    document: dict = {"format": "hexels-bench-table-3", "published_hardware": "RTX A5000 24 GB", "scenes": rows, "means": [], "flags": []}
    return read_json(BenchTable, write(tmp_path / "TABLE.json", document))


def test_a_table_written_before_the_hardware_field_still_reads(tmp_path: Path) -> None:
    old: BenchTable = read_json(BenchTable, write(tmp_path / "TABLE.json", {"format": "hexels-bench-table-2", "scenes": [row("brush-7k")]}))
    assert old.published_hardware is None and old.scenes[0].method == "brush-7k"


def cells(table: BenchTable, method: str) -> SceneCells:
    found: SceneCells | None = scene_cells(table, "mipnerf360", "garden", method)
    assert found is not None
    return found


def test_brush_row_takes_table_cells_and_the_wgsl_rerender_fps(table: BenchTable, tmp_path: Path) -> None:
    backends: list[dict] = [
        {"implementation": "brush", "adapter": "GPU R", "median_of_repeat_medians_ms": 9.0},
        {"implementation": "ours", "adapter": "GPU R", "median_of_repeat_medians_ms": 2.5},
    ]
    speed: Path = write(tmp_path / "speed.json", {"complete": True, "splats": 1000, "backends": backends})
    runs: Path = tmp_path / "runs"
    write(runs / "mipnerf360/garden/brush-7k/train.json", {"method": "brush-7k", "gpu": "GPU T", "host": "h"})
    write(runs / "mipnerf360/garden/brush-7k-wgsl/fps.json", {"median_fps": 554.6, "adapter": "GPU F", "backend": "Vulkan"})
    brush: MetricRow = run_row(table, cells(table, "brush-7k"), "Brush 3DGS 7k", 1000, None, speed, runs)
    assert (brush.psnr, brush.ssim, brush.lpips_vgg, brush.source) == (24.25, 0.7, 0.38, "ours, shared-png16")
    assert (brush.fps, brush.train_min, brush.train_vram_gib, brush.deploy_mb, brush.primitives) == (554.6, 0.5, 0.46, 0.236, 1000)
    assert brush.renderer_ms_median == 2.5 and brush.viewer_ms_median is None
    assert (brush.train_gpu, brush.fps_gpu, brush.renderer_gpu, brush.viewer_gpu) == ("GPU T", "GPU F", "GPU R", None), "each GPU from its own record"


def test_unscored_run_shows_empty_cells_not_another_scorers_numbers(table: BenchTable, tmp_path: Path) -> None:
    unscored: MetricRow = run_row(table, cells(table, "hexels-v1-7k"), "Hexels V1 7k", 31205, None, None, tmp_path / "runs")
    assert unscored.source == "unscored" and unscored.psnr is None and unscored.fps is None and unscored.primitives == 31205


def test_probe_statistics_and_the_published_row(table: BenchTable, tmp_path: Path) -> None:
    frames: list[dict] = [{"ms": float(ms), "camera": {}} for ms in range(1, 301)]
    probe: Path = write(tmp_path / "probe.json", {"adapter": "RTX 5090", "frames": frames, "gpu_bytes_scene": 2_000_000, "gpu_bytes_views": 500_000})
    hexels: MetricRow = run_row(table, cells(table, "hexels-v1-30k-pub-wgsl"), "Hexels V1 30k published", 138_766, probe, None, tmp_path / "runs")
    assert (hexels.psnr, hexels.fps, hexels.train_min) == (24.7121, 900.0, None)
    assert (hexels.viewer_gpu, hexels.fps_gpu, hexels.train_gpu) == ("RTX 5090", None, None), "no fps.json, no train.json: no claim"
    assert hexels.viewer_ms_median == pytest.approx(150.5)
    assert hexels.viewer_ms_p95 == pytest.approx(float(np.percentile(np.arange(1, 301), 95.0)))
    assert hexels.viewer_gpu_mb == pytest.approx(2.5)
    published: MetricRow = paper_row(cells(table, "hexels-v1-30k-pub-wgsl"), "Paper Hexels V1 30k")
    assert (published.psnr, published.primitives, published.train_min, published.deploy_mb) == (24.71, 138766, 20.775, 11.10128)
    assert published.source == "published" and published.train_vram_gib is None
    with pytest.raises(ValueError, match="no published checkpoint"):
        paper_row(cells(table, "brush-7k"), "x")


def test_malformed_records_name_their_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TABLE.json"):
        read_json(BenchTable, write(tmp_path / "TABLE.json", {"scenes": [{"method": 3}]}))


def test_table_has_one_row_per_metric_and_one_column_per_model(tmp_path: Path) -> None:
    recording: rr.RecordingStream = rr.RecordingStream("metrics-table-contract")
    output: Path = tmp_path / "table.rrd"
    recording.save(output)
    log_metric_table([MetricRow("Brush", "ours", psnr=24.0), MetricRow("Hexels", "paper", psnr=24.7, primitives=5)], recording=recording)
    recording.flush(timeout_sec=30.0)
    recording.disconnect()
    batches: list[pa.RecordBatch] = [chunk.to_record_batch() for chunk in RrdReader(output).stream().filter(content=METRICS_ENTITY).to_chunks()]
    rows: list[dict] = []
    for batch in batches:
        for index in range(batch.num_rows):
            rows.append({name: batch.column(name)[index].as_py() for name in batch.schema.names})
    rows.sort(key=lambda row: row["row"])
    assert [row[METRIC_LABEL] for row in rows] == [["PSNR dB"], ["primitives"]]
    assert rows[0]["Brush · ours"] == [24.0] and rows[0]["Hexels · paper"] == [24.7]
    assert rows[1]["Hexels · paper"] == [5.0] and rows[1].get("Brush · ours") in (None, [])
