"""Comparison rows: source precedence, probe statistics, published numbers, and the logged table."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from rerun.chunk import RrdReader

from gsplat_rust_renderer.hexels_metrics import METRIC_LABEL, METRICS_ENTITY, MetricRow, log_metric_table, paper_row, run_row


def write(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_brush_run_uses_its_own_eval_until_the_shared_scorer_exists(tmp_path: Path) -> None:
    run: Path = tmp_path / "brush-7k"
    write(run / "train.json", {"loop_seconds": 30.0, "peak_vram_gib": 0.46, "primitives": 1000, "steps": 7000})
    write(run / "brush-summary.json", {"brush_eval_psnr": 24.2, "brush_eval_ssim": 0.69, "steps": 7000})
    row: MetricRow = run_row(run, "Brush 3DGS 7k", "brush", 1000)
    assert (row.psnr, row.ssim, row.lpips_vgg) == (24.2, 0.69, None)
    assert row.source == "ours, Brush eval 8-bit" and row.train_min == 0.5 and row.train_vram_gib == 0.46
    assert row.deploy_mb == pytest.approx(0.236)
    write(run / "metrics.json", {"psnr": 24.25, "ssim": 0.7, "lpips_vgg": 0.38, "scorer": "shared-png16", "fps": 500.0, "views": 24})
    shared: MetricRow = run_row(run, "Brush 3DGS 7k", "brush", 1000)
    assert (shared.psnr, shared.lpips_vgg, shared.fps, shared.source) == (24.25, 0.38, 500.0, "ours, shared-png16")


def test_published_wgsl_run_reports_parity_psnr_and_probe_statistics(tmp_path: Path) -> None:
    run: Path = tmp_path / "hexels-v1-30k-pub-wgsl"
    write(run / "parity.json", {"summary": {"views": 24, "mean_psnr_wgsl_gt": 24.7121, "max_abs_delta_db": 0.0002}})
    frames: list[dict] = [{"ms": float(ms), "camera": {}} for ms in range(1, 301)]
    probe: Path = write(tmp_path / "probe.json", {"adapter": "RTX 5090", "frames": frames, "gpu_bytes_scene": 2_000_000, "gpu_bytes_views": 500_000})
    row: MetricRow = run_row(run, "Hexels V1 30k", "v1", 138_766, probe=probe)
    assert row.psnr == 24.7121 and row.ssim is None and row.train_min is None
    assert row.deploy_mb == pytest.approx(11.10128)
    assert row.viewer_ms_median == pytest.approx(150.5)
    assert row.viewer_ms_p95 == pytest.approx(float(np.percentile(np.arange(1, 301), 95.0)))
    assert row.viewer_gpu_mb == pytest.approx(2.5)


def test_paper_row_reads_the_published_checkpoint(tmp_path: Path) -> None:
    scene: dict = {"psnr": 24.71, "ssim": 0.728, "lpips_vgg": 0.337, "primitives": 138766, "learned_model_bytes": 11101280,
                   "train_seconds_summary": 1246.5, "peak_gpu_used_gb_legacy": 0.687, "views": 24}
    numbers: Path = write(tmp_path / "paper-numbers.json", {"hf_nvs_per_scene": {"mipnerf360": {"garden": {"v1-fixed": scene}}}, "table1_2d": {}})
    row: MetricRow = paper_row(numbers, "mipnerf360", "garden", "v1-fixed", "Hexels V1 30k (paper)")
    assert (row.psnr, row.primitives, row.train_vram_gib) == (24.71, 138766, 0.687)
    assert row.train_min == pytest.approx(20.775) and row.deploy_mb == pytest.approx(11.10128)


def test_malformed_records_name_their_file(tmp_path: Path) -> None:
    run: Path = tmp_path / "run"
    write(run / "train.json", {"loop_seconds": "slow"})
    with pytest.raises(ValueError, match="train.json"):
        run_row(run, "x", "v1", 1)


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
