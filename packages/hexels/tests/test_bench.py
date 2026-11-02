"""Benchmark conductor: queue order, completeness, skip/resume, failure log (fake runner, no GPU)."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from hexels_bench.bench import Item, Step, StepFailed, build_queue, incomplete_reason, load_failures, pixi_runner, run_queue
from hexels_bench.layout import GSPLAT_RELEASE, Roots
from hexels_bench.methods import Method

NAMES: tuple[str, ...] = ("0000", "0008")
MODEL_PLY: bytes = b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nend_header\n"


def bench_roots(tmp_path: Path) -> Roots:
    return Roots(data=tmp_path / "data", runs=tmp_path / "runs", brush_prepared=tmp_path / "prepared")


def write_cameras(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    camera: dict = {"width": 4, "height": 3, "fov_x": 0.5, "fov_y": 0.4, "world_view_transform": np.eye(4).tolist(), "camera_center": [0.0, 0.0, 0.0]}
    cameras: list[dict] = [{**camera, "name": name} for name in NAMES]
    header: dict = {"format": "hexels-cameras-1", "dataset": "dtu", "scene": "scan24", "split": "test", "background": [0.0, 0.0, 0.0]}
    (run_dir / "cameras.json").write_text(json.dumps({**header, "cameras": cameras}))


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
    (run_dir / "model.ply").write_bytes(MODEL_PLY)
    write_scored(run_dir, method, fps=500.0 if "wgsl" in method or "cuda" in method else None)
    if method in ("brush-7k", "hexels-v1-7k", "hexels-v3-7k"):
        write_train(run_dir, method)
    if method.endswith("pub-cuda"):
        (run_dir / "upstream_float_metrics.json").write_text("{}")


class FakeRunner:
    """Records steps and produces the outputs the real tools would write."""

    def __init__(self, fail_methods: tuple[str, ...] = ()) -> None:
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
            (log.parent / "model.ply").write_bytes(MODEL_PLY)
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


def replace_with_link(path: Path, target: Path) -> None:
    path.unlink()
    path.symlink_to(target)


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
        ("hexels-v1-7k", lambda run: (run / "model.ply").unlink(), "no model.ply"),
        ("brush-7k", lambda run: (run / "model.ply").write_bytes(b"ply\nend_header\n"), "unreadable model.ply"),
        ("hexels-v1-30k-pub-wgsl", lambda run: replace_with_link(run / "model.ply", run / "missing.ply"), "no model.ply"),
    ],
)
def test_incomplete_reason_names_what_is_missing(tmp_path: Path, method: Method, breaks, reason: str) -> None:
    run_dir: Path = tmp_path / method
    complete_run(run_dir, method)
    assert incomplete_reason(run_dir, method) is None
    breaks(run_dir)
    found: str | None = incomplete_reason(run_dir, method)
    assert found is not None and reason in found


def test_the_tail_scores_brush_on_every_other_scene_after_the_selected_ones() -> None:
    from hexels_bench.apis.bench import Config, queue_for

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
    roots: Roots = bench_roots(tmp_path)
    lock: Path = tmp_path / "gpu.lock"
    scene: Path = roots.runs / "dtu" / "scan24"
    complete_run(scene / "hexels-v1-7k", "hexels-v1-7k")
    write_cameras(scene / "hexels-v2-30k-pub-cuda")
    (scene / "hexels-v2-30k-pub-cuda" / "model.ply").write_bytes(MODEL_PLY)
    runner = FakeRunner()
    queue: list[Item] = build_queue(("dtu",), scenes=("scan24",), methods=("brush-7k", "brush-7k-wgsl", "hexels-v1-7k", "hexels-v2-30k-pub-wgsl"))

    summary = run_queue(queue, roots, lock, runner)

    assert [item.method for item in summary.skipped] == ["hexels-v1-7k"]
    assert [item.method for item in summary.done] == ["brush-7k", "brush-7k-wgsl", "hexels-v2-30k-pub-wgsl"]
    assert summary.failed == []
    envs_and_heads: list[tuple[str, str]] = [(env, args[0]) for env, args in runner.steps]
    assert envs_and_heads == [
        ("hexels", "python"),  # Brush protocol runner
        ("hexels", "hexels-score"),
        ("gsplat-rust-renderer", str(GSPLAT_RELEASE / "gsplat")),
        ("hexels", "hexels-score"),
        ("gsplat-rust-renderer", str(GSPLAT_RELEASE / "gsplat")),
        ("hexels", "hexels-score"),
    ]
    hexels_steps: list[tuple[str, ...]] = [args for env, args in runner.steps if env == "hexels"]
    assert all(
        args[args.index(flag) + 1] == str(root)
        for args in hexels_steps
        for flag, root in (("--roots.data", roots.data), ("--roots.runs", roots.runs), ("--roots.brush-prepared", roots.brush_prepared))
    ), "every hexels tool uses the queue's roots"
    brush_train: tuple[str, ...] = runner.steps[0][1]
    assert "--gpu-lock" not in brush_train, "the conductor holds the lock; a child flock would deadlock"
    assert brush_train[brush_train.index("--scenes") + 1] == "scan24"
    wgsl_score: tuple[str, ...] = runner.steps[-1][1]
    assert wgsl_score[wgsl_score.index("--fps") + 1] == "321.5"
    staged: Path = scene / "hexels-v2-30k-pub-wgsl" / "model.ply"
    assert staged.is_symlink() and staged.resolve() == (scene / "hexels-v2-30k-pub-cuda" / "model.ply").resolve()
    assert (scene / "brush-7k-wgsl" / "model.ply").resolve() == (scene / "brush-7k" / "model.ply").resolve()

    rerun = run_queue(queue, roots, lock, FakeRunner())
    assert rerun.done == [] and len(rerun.skipped) == 4


def test_a_failure_is_logged_and_the_queue_continues(tmp_path: Path) -> None:
    roots: Roots = bench_roots(tmp_path)
    lock: Path = tmp_path / "gpu.lock"
    queue: list[Item] = build_queue(("dtu",), scenes=("scan24", "scan37"), methods=("hexels-v3-7k",))
    summary = run_queue(queue, roots, lock, FakeRunner(fail_methods=("hexels-v3-7k",)))
    assert [item.scene for item in summary.failed] == ["scan24", "scan37"]
    failures = load_failures(roots.runs / "FAILURES.jsonl")
    assert [(f.scene, f.method) for f in failures] == [("scan24", "hexels-v3-7k"), ("scan37", "hexels-v3-7k")]
    assert "exited with 1" in failures[0].error

    retry = run_queue(queue[:1], roots, lock, FakeRunner())
    assert [item.scene for item in retry.done] == ["scan24"]


def test_a_wgsl_rerender_waits_for_its_source_run(tmp_path: Path) -> None:
    roots: Roots = bench_roots(tmp_path)
    lock: Path = tmp_path / "gpu.lock"
    summary = run_queue(build_queue(("dtu",), scenes=("scan24",), methods=("brush-7k-wgsl",)), roots, lock, FakeRunner())
    assert len(summary.failed) == 1
    assert "brush-7k" in load_failures(roots.runs / "FAILURES.jsonl")[0].error


def test_partial_outputs_move_to_the_attic_before_a_rerun(tmp_path: Path) -> None:
    roots: Roots = bench_roots(tmp_path)
    lock: Path = tmp_path / "gpu.lock"
    train_run: Path = roots.runs / "dtu" / "scan24" / "hexels-v1-7k"
    (train_run / "upstream").mkdir(parents=True)
    (train_run / "log.txt").write_text("crashed\n")
    cuda_run: Path = roots.runs / "dtu" / "scan24" / "hexels-v1-30k-pub-cuda"
    write_cameras(cuda_run)
    (cuda_run / "model.ply").write_bytes(MODEL_PLY)
    (cuda_run / "upstream-eval").mkdir()
    (cuda_run / "upstream-eval-job").mkdir()

    runner = FakeRunner()
    summary = run_queue(build_queue(("dtu",), scenes=("scan24",), methods=("hexels-v1-7k", "hexels-v1-30k-pub-cuda")), roots, lock, runner)

    assert len(summary.done) == 2
    assert [(args[0], args[args.index("--roots.runs") + 1]) for _, args in runner.steps] == [
        ("hexels-train", str(roots.runs)),
        ("hexels-render-published", str(roots.runs)),
    ]
    assert [args[args.index(flag) + 1] for (_, args), flag in zip(runner.steps, ("--method", "--methods"), strict=True)] == [
        "hexels-v1-7k",
        "hexels-v1-30k-pub-cuda",
    ], "tools take the method itself, never a variant and step count"
    attic: list[str] = sorted(path.name for path in (roots.runs / "_attic").iterdir())
    assert len(attic) == 2
    assert attic[0].startswith("dtu-scan24-hexels-v1-30k-pub-cuda-") and attic[1].startswith("dtu-scan24-hexels-v1-7k-")
    assert (roots.runs / "_attic" / attic[1] / "log.txt").read_text() == "crashed\n"
    assert (cuda_run / "model.ply").is_file(), "staged published files stay"
    assert not (cuda_run / "upstream-eval-job").exists()


def test_roots_default_to_the_environment_then_the_home_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEXELS_WORK", str(tmp_path / "work"))
    monkeypatch.setenv("GSPLAT_MODERN_DATA", str(tmp_path / "data"))
    roots = Roots()
    assert (roots.data, roots.runs, roots.brush_prepared) == (tmp_path / "data", tmp_path / "work/runs", tmp_path / "work/brush-prepared")
    monkeypatch.delenv("HEXELS_WORK")
    monkeypatch.delenv("GSPLAT_MODERN_DATA")
    assert (Roots().data, Roots().runs) == (Path.home() / "gsplat-modern-data", Path.home() / "hexels-work" / "runs")


def test_a_trained_run_missing_only_its_scores_is_scored_not_retrained(tmp_path: Path) -> None:
    roots: Roots = bench_roots(tmp_path)
    run_dir: Path = roots.runs / "dtu" / "scan24" / "hexels-v1-7k"
    complete_run(run_dir, "hexels-v1-7k")
    (run_dir / "metrics.json").unlink()
    runner = FakeRunner()
    summary = run_queue(build_queue(("dtu",), scenes=("scan24",), methods=("hexels-v1-7k",)), roots, tmp_path / "gpu.lock", runner)
    assert [item.method for item in summary.done] == ["hexels-v1-7k"]
    assert [args[0] for _, args in runner.steps] == ["hexels-score"], "7k steps of training are not redone for a missing metrics.json"
    assert not (roots.runs / "_attic").exists()


def test_step_output_reaches_the_log_while_the_step_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long step's log.txt shows its progress before it exits (a fake `pixi` checks its own log mid-run)."""
    log: Path = tmp_path / "run" / "log.txt"
    fake: Path = tmp_path / "bin" / "pixi"
    fake.parent.mkdir()
    fake.write_text(
        f'#!/bin/sh\necho first\nsleep 0.3\nif grep -q first "{log}"; then echo "log had first"; else echo "log lacked first"; fi\nexit 3\n'
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    with pytest.raises(StepFailed, match="exited with 3"):
        pixi_runner(Step(name="probe", env="hexels", args=("true",)), log)
    lines: list[str] = log.read_text().splitlines()
    assert lines[0].startswith("== bench probe: pixi run -e hexels --frozen true")
    assert lines[1:3] == ["first", "log had first"]
    assert lines[3].startswith("== bench probe: exit 3 after")
