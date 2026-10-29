import re
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from hexels_bench.apis.score import score_run
from hexels_bench.apis.train import apply_schedule, scaled_schedule, sfm_point_count
from hexels_bench.cameras import CamerasFile, ContractCamera, write_cameras_file
from hexels_bench.layout import Roots
from hexels_bench.metrics import MetricsFile, load_metrics_file
from hexels_bench.png16 import write_png16
from hexels_bench.upstream import UPSTREAM_HEXELS_REV

RECIPE: Path = Path(__file__).resolve().parents[2] / "hexels-src" / "recipe.yaml"


def test_schedule_is_identity_at_30k_and_scales_to_7k() -> None:
    assert scaled_schedule(30000) == ([15001, 20001, 25001], 30000)
    assert scaled_schedule(7000) == ([3501, 4668, 5834], 7000)


def test_apply_schedule_replaces_upstream_values() -> None:
    command: list[str] = ["python", "-m", "hexels.train", "--iterations", "7000", "--lr_decay_milestones", "15001", "20001", "25001", "--seed", "0"]
    result: list[str] = apply_schedule(command, 7000)
    assert result == [
        "python",
        "-m",
        "hexels.train",
        "--iterations",
        "7000",
        "--lr_decay_milestones",
        "3501",
        "4668",
        "5834",
        "--seed",
        "0",
        "--lr_means3d_max_steps",
        "7000",
    ]
    assert apply_schedule(["--lr_means3d_max_steps", "30000"], 7000)[:2] == ["--lr_means3d_max_steps", "7000"]


def test_sfm_point_count_reads_colmap_binary_and_text(tmp_path: Path) -> None:
    sparse: Path = tmp_path / "a" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 31205) + b"\x00" * 16)
    assert sfm_point_count(tmp_path / "a") == 31205
    text: Path = tmp_path / "b" / "sparse" / "0"
    text.mkdir(parents=True)
    (text / "points3D.txt").write_text("# header\n1 0 0 0 1 2 3 0.1\n2 0 0 0 1 2 3 0.1\n")
    assert sfm_point_count(tmp_path / "b") == 2


def test_upstream_revision_matches_recipe() -> None:
    revisions: list[str] = re.findall(r"rev: ([0-9a-f]{40})", RECIPE.read_text())
    assert revisions[0] == UPSTREAM_HEXELS_REV


def test_score_run_writes_metrics_beside_renders(tmp_path: Path) -> None:
    roots = Roots(data=tmp_path / "data", runs=tmp_path / "runs")
    run_dir: Path = roots.run_dir("dtu", "scanX", "hexels-v1-7k")
    camera = ContractCamera(
        name="0000", width=8, height=6, fov_x=0.5, fov_y=0.4, world_view_transform=np.eye(4).tolist(), camera_center=[0.0, 0.0, 0.0]
    )
    write_cameras_file(
        run_dir / "cameras.json",
        CamerasFile(format="hexels-cameras-1", dataset="dtu", scene="scanX", split="test", background=[0.0, 0.0, 0.0], cameras=[camera]),
    )
    rgb: np.ndarray = np.random.default_rng(0).integers(0, 65536, size=(6, 8, 3), dtype=np.uint16)
    write_png16(roots.gt_dir("dtu", "scanX") / "0000.png", rgb)
    write_png16(run_dir / "renders" / "test" / "0000.png", rgb // 2)
    metrics: MetricsFile = score_run(roots, "dtu", "scanX", "hexels-v1-7k", 123.0, torch.device("cpu"), with_lpips=False)
    assert metrics.views == 1 and metrics.fps == 123.0 and metrics.deploy_mb is None
    assert load_metrics_file(run_dir / "metrics.json") == metrics
    assert pytest.approx(metrics.psnr, abs=1e-9) == metrics.per_view[0].psnr
