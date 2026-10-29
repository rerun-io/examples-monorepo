"""Table builder math: per-scene rows, dataset means, paper values beside ours, flags, and the metrics recording."""

import json
from pathlib import Path

import pytest

from gsplat_rust_renderer.bench_table import BenchTable, PaperNumbers, build_table, mean_columns, scene_columns, write_table_rrd
from gsplat_rust_renderer.nvs_protocol import read_json


def paper_numbers(path: Path) -> Path:
    """A minimal paper-numbers.json with the keys the builder reads (values are made up)."""
    hf: dict = {
        "views": 2, "psnr": 26.0, "ssim": 0.9, "lpips_vgg": 0.2, "primitives": 100, "learned_model_bytes": 8000,
        "train_seconds_summary": 600.0, "peak_gpu_used_gb_legacy": 0.5,
    }  # fmt: skip
    data: dict = {
        "hf_nvs_per_scene": {"dtu": {"scan24": {"v1-fixed": hf}, "scan37": {"v1-fixed": {**hf, "psnr": 27.0, "primitives": 120}}}},
        "sfm_points": {"dtu": {"scan24": 100, "scan37": 120}},
        "table3_matched_primitives_sfm_count": {"dtu": {}},
    }
    path.write_text(json.dumps(data))
    return path


def write_run(runs: Path, scene: str, method: str, psnr: float, fps: float | None, train: tuple[float, float, float] | None, vertices: int) -> None:
    run_dir: Path = runs / "dtu" / scene / method
    (run_dir / "renders" / "test").mkdir(parents=True)
    names: list[str] = ["0000", "0008"]
    (run_dir / "cameras.json").write_text(json.dumps({"cameras": [{"name": name} for name in names]}))
    for name in names:
        (run_dir / "renders" / "test" / f"{name}.png").write_bytes(b"png")
    per_view: list[dict] = [{"name": name, "psnr": psnr, "ssim": 0.85, "lpips_vgg": 0.25} for name in names]
    metrics: dict = {
        "method": method, "dataset": "dtu", "scene": scene, "views": 2, "psnr": psnr, "ssim": 0.85, "lpips_vgg": 0.25, "fps": fps,
        "deploy_mb": vertices * 80 / 1e6, "ply_mb": 0.01, "scorer": "shared-png16", "per_view": per_view,
    }  # fmt: skip
    (run_dir / "model.ply").write_bytes(f"ply\nformat binary_little_endian 1.0\nelement vertex {vertices}\nend_header\n".encode())
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    if train is not None:
        loop, total, vram = train
        record: dict = {
            "method": method, "dataset": "dtu", "scene": scene, "steps": 7000, "loop_seconds": loop, "total_seconds": total,
            "peak_vram_gib": vram, "primitives": vertices, "sfm_points": vertices, "gpu": "GPU", "host": "host", "source_sha": "abc", "command": "x",
        }  # fmt: skip
        (run_dir / "train.json").write_text(json.dumps(record))
    if method.endswith("pub-cuda"):
        (run_dir / "upstream_float_metrics.json").write_text("{}")


@pytest.fixture
def table(tmp_path: Path) -> BenchTable:
    runs: Path = tmp_path / "runs"
    write_run(runs, "scan24", "hexels-v1-7k", 24.0, 900.0, (60.0, 90.0, 0.5), 100)
    write_run(runs, "scan37", "hexels-v1-7k", 26.0, 1100.0, (120.0, 150.0, 0.7), 120)
    write_run(runs, "scan24", "hexels-v1-30k-pub-wgsl", 26.03, 1500.0, None, 100)
    write_run(runs, "scan37", "hexels-v1-30k-pub-wgsl", 26.2, 1300.0, None, 120)
    (runs / "dtu" / "scan37" / "brush-7k").mkdir(parents=True)  # started, never scored
    two: dict = {"dtu": ("scan24", "scan37")}
    return build_table(runs, read_json(PaperNumbers, paper_numbers(tmp_path / "paper.json")), datasets=("dtu",), scenes=two, mean_scenes=two)


def test_scene_rows_carry_our_numbers_and_the_published_scene_beside_them(table: BenchTable) -> None:
    row = next(r for r in table.scenes if r.scene == "scan24" and r.method == "hexels-v1-7k")
    assert row.status == "complete"
    assert (row.train_min, row.total_min, row.vram_gib, row.fps, row.psnr) == (1.0, 1.5, 0.5, 900.0, 24.0)
    assert (row.primitives, row.sfm_points, row.views) == (100, 100, 2)
    assert row.paper is not None and row.paper.psnr == 26.0 and row.paper.train_min == 10.0 and row.paper.deploy_mb == 0.008
    assert row.delta_psnr == pytest.approx(-2.0)
    pub = next(r for r in table.scenes if r.scene == "scan24" and r.method == "hexels-v1-30k-pub-wgsl")
    assert pub.primitives == 100 and pub.train_min is None
    brush = next(r for r in table.scenes if r.scene == "scan37" and r.method == "brush-7k")
    assert brush.status == "incomplete" and brush.psnr is None and brush.paper is None
    missing = next(r for r in table.scenes if r.scene == "scan24" and r.method == "brush-7k")
    assert missing.status == "missing"


def test_dataset_means_average_the_scenes_beside_the_published_scenes_mean(table: BenchTable) -> None:
    mean = next(m for m in table.means if m.dataset == "dtu" and m.method == "hexels-v1-7k")
    assert (mean.scenes, mean.total, mean.complete) == (2, 2, True)
    assert mean.psnr == pytest.approx(25.0)
    assert mean.train_min == pytest.approx(1.5)
    assert mean.vram_gib == pytest.approx(0.6)
    assert mean.fps == pytest.approx(1000.0)
    assert mean.primitives_k == pytest.approx(0.11)
    assert mean.paper is not None and mean.paper.model == "v1-fixed" and mean.paper.scenes == 2
    assert mean.paper.psnr == pytest.approx(26.5) and mean.paper.primitives_k == pytest.approx(0.11)
    assert mean.delta_psnr == pytest.approx(-1.5)
    brush = next(m for m in table.means if m.method == "brush-7k")
    assert (brush.scenes, brush.complete, brush.psnr, brush.paper) == (0, False, None, None)
    v3 = next(m for m in table.means if m.method == "hexels-v3-7k")
    assert v3.paper is None, "no published v3 numbers in this fixture"


def test_published_checkpoints_off_the_paper_by_more_than_0_05_db_are_flagged(table: BenchTable) -> None:
    flagged: list[str] = [flag for flag in table.flags if "hexels-v1-30k-pub-wgsl" in flag]
    assert len(flagged) == 1, table.flags
    assert "dtu/scan37" in flagged[0] and "-0.800 dB" in flagged[0]


def test_paper_style_columns_put_the_published_checkpoint_beside_ours(table: BenchTable) -> None:
    columns = scene_columns([row for row in table.scenes if row.scene == "scan24"])
    assert set(columns) == {"hexels-v1-7k", "hexels-v1-30k-pub-wgsl", "published v1-fixed 30k"}
    assert columns["hexels-v1-7k"]["psnr"] == 24.0 and columns["hexels-v1-7k"]["primitives_k"] == 0.1
    assert columns["published v1-fixed 30k"]["psnr"] == 26.0 and columns["published v1-fixed 30k"]["train_min"] == 10.0
    means = mean_columns([entry for entry in table.means if entry.dataset == "dtu"])
    assert "brush-7k" not in means, "no complete scene, no column"
    assert means["hexels-v1-7k"]["scenes"] == 2.0 and means["published v1-fixed 30k"]["psnr"] == pytest.approx(26.5)


def test_metrics_recording_holds_per_scene_paper_and_mean_entities(table: BenchTable, tmp_path: Path) -> None:
    from rerun.chunk import RrdReader

    path: Path = tmp_path / "table.rrd"
    write_table_rrd(table, path)
    reader = RrdReader(path)
    entities: set[str] = {str(entity) for entity in reader.store().schema().entity_paths()}
    assert {"/metrics/dtu/hexels-v1-7k", "/table/dtu/mean", "/table/dtu/scan24", "/table/dtu/scan37"} <= entities
    assert reader.blueprints(), "the recording ships a default blueprint"
