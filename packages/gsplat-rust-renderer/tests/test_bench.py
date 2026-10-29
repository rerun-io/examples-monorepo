"""Benchmark conductor: queue order, completeness, skip/resume, failure log (fake runner, no GPU)."""

import json
import os
import time
from pathlib import Path

import pytest

from gsplat_rust_renderer.bench import (
    BenchPaths,
    Item,
    Step,
    StepFailed,
    build_queue,
    incomplete_reason,
    load_failures,
    run_queue,
)

NAMES: tuple[str, ...] = ("0000", "0008")


def bench_paths(tmp_path: Path) -> BenchPaths:
    return BenchPaths(
        repo=tmp_path / "repo",
        runs=tmp_path / "runs",
        data=tmp_path / "data",
        brush_prepared=tmp_path / "prepared",
        gpu_lock=tmp_path / "gpu.lock",
    )


def write_cameras(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cameras: list[dict] = [{"name": name, "width": 4, "height": 3} for name in NAMES]
    (run_dir / "cameras.json").write_text(json.dumps({"format": "hexels-cameras-1", "cameras": cameras}))


def write_scored(run_dir: Path, method: str, fps: float | None = None, lpips: float | None = 0.2) -> None:
    """A scored run directory as the shared scorer leaves it."""
    write_cameras(run_dir)
    (run_dir / "renders" / "test").mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        (run_dir / "renders" / "test" / f"{name}.png").write_bytes(b"png")
    per_view: list[dict] = [{"name": name, "psnr": 25.0, "ssim": 0.8, "lpips_vgg": lpips} for name in NAMES]
    metrics: dict = {
        "method": method, "dataset": "dtu", "scene": "scan24", "views": len(NAMES), "psnr": 25.0, "ssim": 0.8,
        "lpips_vgg": lpips, "fps": fps, "deploy_mb": 1.0, "ply_mb": 1.0, "scorer": "shared-png16", "per_view": per_view,
    }  # fmt: skip
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def write_train(run_dir: Path, method: str) -> None:
    record: dict = {
        "method": method, "dataset": "dtu", "scene": "scan24", "steps": 7000, "loop_seconds": 60.0, "total_seconds": 70.0,
        "peak_vram_gib": 0.5, "primitives": 10, "sfm_points": 10, "gpu": "GPU", "host": "host", "source_sha": "abc", "command": "train",
    }  # fmt: skip
    (run_dir / "train.json").write_text(json.dumps(record))


def complete_run(run_dir: Path, method: str) -> None:
    """Every file a finished run of `method` has (the model is written before it is scored)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.ply").write_bytes(b"ply")
    write_scored(run_dir, method, fps=500.0 if "wgsl" in method or "cuda" in method else None)
    if method in ("brush-7k", "hexels-v1-7k", "hexels-v3-7k"):
        write_train(run_dir, method)
    if method.endswith("pub-cuda"):
        (run_dir / "upstream_float_metrics.json").write_text("{}")


class FakeRunner:
    """Records steps and produces the outputs the real tools would write."""

    def __init__(self, paths: BenchPaths, fail_methods: tuple[str, ...] = ()) -> None:
        self.paths: BenchPaths = paths
        self.fail_methods: tuple[str, ...] = fail_methods
        self.steps: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, step: Step, log: Path) -> None:
        self.steps.append((step.env, step.args))
        method: str = log.parent.name
        if method in self.fail_methods:
            raise StepFailed(f"{step.name} exited with 1")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write(f"{step.name}\n")
        if step.name == "brush-train":
            write_cameras(log.parent)
            write_train(log.parent, method)
            (log.parent / "model.ply").write_bytes(b"ply")
        elif step.name == "wgsl-render":
            (log.parent / "fps.json").write_text(json.dumps(fps_report(321.5)))
        elif step.name in ("hexels-train", "hexels-render-published"):
            complete_run(log.parent, method)
        elif step.name == "score":
            fps: float | None = float(step.args[step.args.index("--fps") + 1]) if "--fps" in step.args else None
            write_scored(log.parent, method, fps=fps)


def fps_report(median_fps: float) -> dict:
    trial: dict = {"seconds": 0.1, "frames": 20, "fps": median_fps}
    return {
        "protocol": "hexels-paper", "warmup_passes": 2, "timed_passes": 10, "timing_trials": 4, "views": 2, "width": 4, "height": 3,
        "adapter": "GPU", "backend": "Vulkan", "trials": [trial] * 4, "median_fps": median_fps,
    }  # fmt: skip


def test_queue_runs_datasets_in_order_then_scenes_in_protocol_order_then_methods() -> None:
    queue: list[Item] = build_queue(("dtu", "mipnerf360"), scenes=("garden", "scan37", "scan24"), methods=("brush-7k", "hexels-v1-7k"))
    assert [(item.dataset, item.scene, item.method) for item in queue] == [
        ("dtu", "scan24", "brush-7k"),
        ("dtu", "scan24", "hexels-v1-7k"),
        ("dtu", "scan37", "brush-7k"),
        ("dtu", "scan37", "hexels-v1-7k"),
        ("mipnerf360", "garden", "brush-7k"),
        ("mipnerf360", "garden", "hexels-v1-7k"),
    ]
    assert len(build_queue(("dtu", "mipnerf360", "tnt"))) == 34 * 10
    with pytest.raises(ValueError, match="Garden"):
        build_queue(("mipnerf360",), scenes=("Garden",))


@pytest.mark.parametrize(
    ("method", "breaks", "reason"),
    [
        ("hexels-v1-30k-pub-wgsl", lambda run: (run / "metrics.json").unlink(), "no metrics.json"),
        ("hexels-v1-30k-pub-wgsl", lambda run: (run / "renders/test/0008.png").unlink(), "1 renders"),
        ("brush-7k", lambda run: (run / "train.json").unlink(), "no train.json"),
        ("hexels-v3-30k-pub-cuda", lambda run: (run / "upstream_float_metrics.json").unlink(), "upstream_float_metrics.json"),
        ("brush-7k-wgsl", lambda run: write_scored(run, "brush-7k-wgsl", fps=None), "no FPS"),
        ("hexels-v1-7k", lambda run: write_scored(run, "hexels-v1-7k", fps=1.0, lpips=None), "LPIPS"),
        ("hexels-v1-7k", lambda run: write_scored(run, "brush-7k", fps=1.0), "method brush-7k"),
        ("brush-7k", lambda run: os.utime(run / "model.ply", (time.time() + 60, time.time() + 60)), "older than model.ply"),
    ],
)
def test_incomplete_reason_names_what_is_missing(tmp_path: Path, method: str, breaks, reason: str) -> None:
    run_dir: Path = tmp_path / method
    complete_run(run_dir, method)
    assert incomplete_reason(run_dir, method) is None
    breaks(run_dir)
    found: str | None = incomplete_reason(run_dir, method)
    assert found is not None and reason in found


def test_the_tail_scores_brush_on_every_other_scene_after_the_selected_ones() -> None:
    from gsplat_rust_renderer.apis.bench import Config, queue_for

    queue: list[Item] = queue_for(
        Config(datasets=("dtu",), scenes=("scan24", "scan114"), methods=("brush-7k", "hexels-v1-7k"), tail_methods=("brush-7k",))
    )
    assert [(item.scene, item.method) for item in queue[:4]] == [
        ("scan24", "brush-7k"),
        ("scan24", "hexels-v1-7k"),
        ("scan114", "brush-7k"),
        ("scan114", "hexels-v1-7k"),
    ]
    assert len(queue) == 4 + 13 and all(item.method == "brush-7k" for item in queue[4:])
    assert queue[4].scene == "scan37"


def test_complete_items_are_skipped_and_the_rest_run_through_pixi_envs(tmp_path: Path) -> None:
    paths: BenchPaths = bench_paths(tmp_path)
    scene: Path = paths.runs / "dtu" / "scan24"
    complete_run(scene / "hexels-v1-7k", "hexels-v1-7k")
    write_cameras(scene / "hexels-v2-30k-pub-cuda")
    (scene / "hexels-v2-30k-pub-cuda" / "model.ply").write_bytes(b"ply")
    runner = FakeRunner(paths)
    queue: list[Item] = build_queue(("dtu",), scenes=("scan24",), methods=("brush-7k", "brush-7k-wgsl", "hexels-v1-7k", "hexels-v2-30k-pub-wgsl"))

    summary = run_queue(queue, paths, runner)

    assert [item.method for item in summary.skipped] == ["hexels-v1-7k"]
    assert [item.method for item in summary.done] == ["brush-7k", "brush-7k-wgsl", "hexels-v2-30k-pub-wgsl"]
    assert summary.failed == []
    envs_and_heads: list[tuple[str, str]] = [(env, args[0]) for env, args in runner.steps]
    assert envs_and_heads == [
        ("gsplat-rust-renderer", "python"),  # Brush protocol runner
        ("hexels", "hexels-score"),
        ("gsplat-rust-renderer", str(paths.gsplat_binary)),
        ("hexels", "hexels-score"),
        ("gsplat-rust-renderer", str(paths.gsplat_binary)),
        ("hexels", "hexels-score"),
    ]
    brush_train: tuple[str, ...] = runner.steps[0][1]
    assert "--gpu-lock" not in brush_train, "the conductor holds the lock; a child flock would deadlock"
    assert brush_train[brush_train.index("--scenes") + 1] == "scan24"
    wgsl_score: tuple[str, ...] = runner.steps[-1][1]
    assert wgsl_score[wgsl_score.index("--fps") + 1] == "321.5"
    staged: Path = scene / "hexels-v2-30k-pub-wgsl" / "model.ply"
    assert staged.is_symlink() and staged.resolve() == (scene / "hexels-v2-30k-pub-cuda" / "model.ply").resolve()
    assert (scene / "brush-7k-wgsl" / "model.ply").resolve() == (scene / "brush-7k" / "model.ply").resolve()

    rerun = run_queue(queue, paths, FakeRunner(paths))
    assert rerun.done == [] and len(rerun.skipped) == 4


def test_a_failure_is_logged_and_the_queue_continues(tmp_path: Path) -> None:
    paths: BenchPaths = bench_paths(tmp_path)
    queue: list[Item] = build_queue(("dtu",), scenes=("scan24", "scan37"), methods=("hexels-v3-7k",))
    summary = run_queue(queue, paths, FakeRunner(paths, fail_methods=("hexels-v3-7k",)))
    assert [item.scene for item in summary.failed] == ["scan24", "scan37"]
    failures = load_failures(paths.runs / "FAILURES.jsonl")
    assert [(f.scene, f.method) for f in failures] == [("scan24", "hexels-v3-7k"), ("scan37", "hexels-v3-7k")]
    assert "exited with 1" in failures[0].error

    retry = run_queue(queue[:1], paths, FakeRunner(paths))
    assert [item.scene for item in retry.done] == ["scan24"]


def test_a_wgsl_rerender_waits_for_its_source_run(tmp_path: Path) -> None:
    paths: BenchPaths = bench_paths(tmp_path)
    summary = run_queue(build_queue(("dtu",), scenes=("scan24",), methods=("brush-7k-wgsl",)), paths, FakeRunner(paths))
    assert len(summary.failed) == 1
    assert "brush-7k" in load_failures(paths.runs / "FAILURES.jsonl")[0].error


def test_partial_outputs_move_to_the_attic_before_a_rerun(tmp_path: Path) -> None:
    paths: BenchPaths = bench_paths(tmp_path)
    train_run: Path = paths.runs / "dtu" / "scan24" / "hexels-v1-7k"
    (train_run / "upstream").mkdir(parents=True)
    (train_run / "log.txt").write_text("crashed\n")
    cuda_run: Path = paths.runs / "dtu" / "scan24" / "hexels-v1-30k-pub-cuda"
    write_cameras(cuda_run)
    (cuda_run / "model.ply").write_bytes(b"ply")
    (cuda_run / "upstream-eval").mkdir()
    (cuda_run / "upstream-eval-job").mkdir()

    summary = run_queue(build_queue(("dtu",), scenes=("scan24",), methods=("hexels-v1-7k", "hexels-v1-30k-pub-cuda")), paths, FakeRunner(paths))

    assert len(summary.done) == 2
    attic: list[str] = sorted(path.name for path in (paths.runs / "_attic").iterdir())
    assert len(attic) == 2
    assert attic[0].startswith("dtu-scan24-hexels-v1-30k-pub-cuda-") and attic[1].startswith("dtu-scan24-hexels-v1-7k-")
    assert (paths.runs / "_attic" / attic[1] / "log.txt").read_text() == "crashed\n"
    assert (cuda_run / "model.ply").is_file(), "staged published files stay"
    assert not (cuda_run / "upstream-eval-job").exists()
