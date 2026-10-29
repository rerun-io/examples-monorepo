import re
from pathlib import Path

import numpy as np
import pytest
import torch

from hexels_bench.apis.fit_image import fit_command
from hexels_bench.apis.score import score_run
from hexels_bench.apis.train import apply_schedule, scaled_schedule
from hexels_bench.cameras import CamerasFile, ContractCamera, write_cameras_file
from hexels_bench.layout import Roots, read_json
from hexels_bench.metrics import MetricsFile
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


def test_upstream_revision_matches_recipe() -> None:
    revisions: list[str] = re.findall(r"rev: ([0-9a-f]{40})", RECIPE.read_text())
    assert revisions[0] == UPSTREAM_HEXELS_REV


def test_fit_command_runs_upstreams_paper_recipes(tmp_path: Path) -> None:
    """All three 2D methods go through upstream `scripts.fit_image` with its paper defaults (specs/twod.md D6)."""
    roots = Roots(data=tmp_path / "data", runs=tmp_path / "runs")
    command: list[str] = fit_command(roots, "anime-1_2k", "imagegs-2d", tmp_path / "out", None)
    assert command[1:3] == ["-m", "scripts.fit_image"]
    assert command[-2:] == ["--variant", "imagegs"]
    assert command[command.index("--image") + 1] == str(tmp_path / "data" / "imagegs" / "images" / "anime-1_2k.png")
    assert fit_command(roots, "anime-1_2k", "hexels-2d-flat", tmp_path / "out", 500)[-4:] == ["--variant", "flat", "--steps", "500"]


def test_score_run_writes_metrics_beside_renders(tmp_path: Path) -> None:
    roots = Roots(data=tmp_path / "data", runs=tmp_path / "runs")
    run_dir: Path = roots.run_dir("dtu", "scanX", "hexels-v1-7k")
    camera = ContractCamera(name="0000", width=8, height=6, fov_x=0.5, fov_y=0.4, world_view_transform=np.eye(4), camera_center=np.zeros(3))
    write_cameras_file(
        run_dir / "cameras.json",
        CamerasFile(format="hexels-cameras-1", dataset="dtu", scene="scanX", split="test", background=[0.0, 0.0, 0.0], cameras=[camera]),
    )
    rgb: np.ndarray = np.random.default_rng(0).integers(0, 65536, size=(6, 8, 3), dtype=np.uint16)
    write_png16(roots.gt_dir("dtu", "scanX") / "0000.png", rgb)
    write_png16(run_dir / "renders" / "test" / "0000.png", rgb // 2)
    metrics: MetricsFile = score_run(roots, "dtu", "scanX", "hexels-v1-7k", 80, 123.0, torch.device("cpu"), with_lpips=False)
    assert metrics.views == 1 and metrics.fps == 123.0 and metrics.deploy_mb is None
    assert read_json(MetricsFile, run_dir / "metrics.json") == metrics
    assert pytest.approx(metrics.psnr, abs=1e-9) == metrics.per_view[0].psnr
