"""The protocol runner writes the run-layout directory around one Brush training."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pytest

from hexels_bench.apis.train_brush import Config, main
from hexels_bench.layout import GSPLAT_RELEASE, Roots

SceneFactory: TypeAlias = Callable[..., tuple[Path, Path]]
CONTRACT: str = "data/hexels-pretrained/cameras/nvs/dtu/{scene}/contract.json"

FAKE_TRAINER: str = """#!/usr/bin/env python3
import json, pathlib, sys
args = sys.argv[1:]
pathlib.Path(__file__).with_name("arguments").write_text("\\n".join(args))
value = lambda flag: args[args.index(flag) + 1]
pathlib.Path(value("--export-path"), "model.ply").write_bytes(b"ply\\nformat binary_little_endian 1.0\\nelement vertex 2\\nend_header\\n")
renders = pathlib.Path(value("--test-renders"))
renders.mkdir(parents=True)
for name in ("0000", "0008"):
    (renders / f"{name}.png").write_bytes(b"png")
pathlib.Path(value("--summary")).write_text(json.dumps({"steps": 70, "loop_seconds": 1.5, "wall_seconds": 2.0, "training_end_unix_seconds": 0.0,
    "splats": 2, "brush_eval_psnr": 20.0, "brush_eval_ssim": 0.5, "renderer": "fake"}))
print("trained")
"""


def config(root: Path, binary: Path, steps: int = 70, scenes: tuple[str, ...] = ("scan1",)) -> Config:
    roots = Roots(data=root / "data", runs=root / "runs", brush_prepared=root / "prepared")
    return Config(dataset="dtu", scenes=scenes, steps=steps, method="brush-test", roots=roots, binary=binary, gpu_lock=root / "gpu.lock")


def test_run_directory_follows_the_run_layout_contract(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    protocol_scene(tmp_path / "data/dtu/scan1", tmp_path / CONTRACT.format(scene="scan1"), ["0000", "0008"])
    binary: Path = tmp_path / "bin/gsplat-train"
    binary.parent.mkdir()
    binary.write_text(FAKE_TRAINER)
    binary.chmod(0o755)
    main(config(tmp_path, binary), ["--", "--seed", "7"])
    run: Path = tmp_path / "runs/dtu/scan1/brush-test"
    arguments: list[str] = (tmp_path / "bin/arguments").read_text().splitlines()
    assert arguments == [
        str((tmp_path / "prepared/dtu/scan1").resolve()),
        "--total-train-iters",
        "70",
        "--max-splats",
        "3",
        "--eval-every",
        "70",
        "--export-every",
        "70",
        "--export-path",
        str(run.resolve()),
        "--export-name",
        "model.ply",
        "--test-renders",
        str((run / "renders/test").resolve()),
        "--summary",
        str((run / "brush-summary.json").resolve()),
        "--seed",
        "7",
    ]
    report: dict = json.loads((run / "train.json").read_text())
    assert set(report) == {
        "method",
        "dataset",
        "scene",
        "steps",
        "loop_seconds",
        "total_seconds",
        "peak_vram_gib",
        "primitives",
        "sfm_points",
        "gpu",
        "host",
        "source_sha",
        "command",
    }
    assert (report["method"], report["dataset"], report["scene"], report["steps"]) == ("brush-test", "dtu", "scan1", 70)
    assert (report["loop_seconds"], report["primitives"], report["sfm_points"]) == (1.5, 2, 3)
    assert report["total_seconds"] > 0.0
    assert report["command"].endswith("--seed 7")
    assert (run / "log.txt").read_text().splitlines()[1:] == ["trained"], "the command line, then the trainer's output"
    cameras: dict = json.loads((run / "cameras.json").read_text())
    assert (cameras["format"], cameras["split"], cameras["background"]) == ("hexels-cameras-1", "test", [0.0, 0.0, 0.0])
    assert [camera["name"] for camera in cameras["cameras"]] == ["0000", "0008"]
    assert cameras["cameras"][1]["world_view_transform"][3] == [0.8, 0.0, 3.0, 1.0]
    # A finished run is left alone.
    (tmp_path / "bin/arguments").unlink()
    main(config(tmp_path, binary), [])
    assert not (tmp_path / "bin/arguments").exists()


def test_failed_training_leaves_no_report(tmp_path: Path, protocol_scene: SceneFactory, capsys: pytest.CaptureFixture[str]) -> None:
    protocol_scene(tmp_path / "data/dtu/scan1", tmp_path / CONTRACT.format(scene="scan1"), ["0000", "0008"])
    binary: Path = tmp_path / "gsplat-train"
    binary.write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
    binary.chmod(0o755)
    with pytest.raises(RuntimeError, match="1 of 1 scenes failed: scan1"):
        main(config(tmp_path, binary), [])
    assert "FAILED dtu/scan1: gsplat-train exited with 3" in capsys.readouterr().out
    assert not (tmp_path / "runs/dtu/scan1/brush-test/train.json").exists()
    assert "broken" in (tmp_path / "runs/dtu/scan1/brush-test/log.txt").read_text()


def test_a_failing_scene_does_not_stop_the_batch(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    protocol_scene(tmp_path / "data/dtu/scan2", tmp_path / CONTRACT.format(scene="scan2"), ["0000", "0008"])
    binary: Path = tmp_path / "bin/gsplat-train"
    binary.parent.mkdir()
    binary.write_text(FAKE_TRAINER)
    binary.chmod(0o755)
    with pytest.raises(RuntimeError, match="1 of 2 scenes failed: scan1"):
        main(config(tmp_path, binary, scenes=("scan1", "scan2")), [])
    assert (tmp_path / "runs/dtu/scan2/brush-test/train.json").is_file()


@pytest.mark.integration
def test_tiny_brush_training_through_the_protocol_path(tmp_path: Path, protocol_scene: SceneFactory) -> None:
    """Real gsplat-train (release build) on a synthetic 10-view scene: PLY, 16-bit renders, report."""
    cv2 = pytest.importorskip("cv2")
    binary: Path = GSPLAT_RELEASE / "gsplat-train"
    if not binary.is_file():
        pytest.skip(f"needs {binary} (pixi run -e gsplat-rust-renderer gsplat-build)")
    source, _ = protocol_scene(
        tmp_path / "data/dtu/scan1", tmp_path / CONTRACT.format(scene="scan1"), ["0000", "0008"], image_size=(64, 48), contract_size=(32, 24)
    )
    rng: np.random.Generator = np.random.default_rng(1)
    points: list[str] = [
        f"{i} {x:.4f} {y:.4f} {z:.4f} {r} {g} {b} 0.1 1 0"
        for i, ((x, y, z), (r, g, b)) in enumerate(zip(rng.uniform(-0.5, 0.5, (64, 3)).tolist(), rng.integers(0, 256, (64, 3)).tolist(), strict=True))
    ]
    (source / "sparse/0/points3D.txt").write_text("\n".join(points) + "\n")
    main(config(tmp_path, binary, steps=30), [])
    run: Path = tmp_path / "runs/dtu/scan1/brush-test"
    report: dict = json.loads((run / "train.json").read_text())
    header: bytes = (run / "model.ply").read_bytes().split(b"end_header")[0]
    assert f"element vertex {report['primitives']}\n".encode() in header
    assert 0 < report["primitives"] <= report["sfm_points"] == 64
    assert 0.0 < report["loop_seconds"] < report["total_seconds"]
    assert report["peak_vram_gib"] > 0.0, "NVML sees the trainer's wgpu memory"
    for name in ("0000", "0008"):
        render = cv2.imread(str(run / f"renders/test/{name}.png"), cv2.IMREAD_UNCHANGED)
        assert (render.dtype, render.shape) == (np.uint16, (24, 32, 3))
    summary: dict = json.loads((run / "brush-summary.json").read_text())
    assert summary["brush_eval_psnr"] is not None and summary["training_end_unix_seconds"] > 0.0
