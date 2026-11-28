"""Solver timing on in-memory synthetic problems shaped like the harness's COLMAP models (no pycolmap needed).

P1 is `make_problem(PINHOLE, 400 frames, 40,000 points, seed 0, visible 0.15)`: 2,398,861 observations, the harness's
p1 model. The `*-100` problems match p3/p4/p5 (100 frames, 20,000 points, about 300 k observations) and `rig-100`
p6 (two cameras, the second's extrinsic refined). `clip007-shape` is p2 (clip 007): 524 frames of a fixed 6-camera
rig, about 380 observations per image and 4 per point, so warp tiles rather than whole blocks share an image. Every
LM trial count and PCG iteration count is printed next to its time: after convergence float32 noise decides the tail
length, so time per PCG iteration is the fair comparison.
"""

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from caspar_mojo.problem import BaProblem, CameraModel
from caspar_mojo.reference.synthetic import make_problem
from caspar_mojo.solver import solve

PROBLEMS: dict[str, tuple[CameraModel, dict[str, object]]] = {
    "p1": (CameraModel.PINHOLE, {"n_frames": 400, "n_points": 40000, "visible_fraction": 0.15}),
    "pinhole-100": (CameraModel.PINHOLE, {}),
    "simple-radial-100": (CameraModel.SIMPLE_RADIAL, {}),
    "fisheye-100": (CameraModel.OPENCV_FISHEYE, {}),
    "full-opencv-100": (CameraModel.FULL_OPENCV, {}),
    "rig-100": (CameraModel.PINHOLE, {"rig_cameras": 2, "sensor_free": True}),
    "clip007-shape": (
        CameraModel.PINHOLE,
        {"n_frames": 524, "n_points": 300000, "rig_cameras": 6, "visible_fraction": 0.5, "frame_window": 1100},
    ),
}
"""Problem name -> camera model and `make_problem` arguments over the 100-frame defaults."""


@dataclass(frozen=True, slots=True)
class Config:
    """What to time."""

    problems: tuple[str, ...] = tuple(PROBLEMS)
    """Names from `PROBLEMS`."""
    intrinsics_free: tuple[bool, ...] = (False, True)
    """Run each problem with the focal-and-extra block fixed, refined, or both."""
    runs: int = 7
    """Timed solves per case, after one untimed warm-up."""
    profile: bool = False
    """One more solve per case with CASPAR_MOJO_PROF=1 (a sync after every kernel, so never timed)."""
    output: Path | None = None
    """JSON file for the result rows (outside the repository)."""
    lock: Path | None = None
    """File to flock around each case's solves (a GPU shared between jobs), not around the problem building."""


@dataclass(frozen=True, slots=True)
class Timing:
    """One case: the best and median solver time of `Config.runs` solves, with every run's LM and PCG counts."""

    problem: str
    intrinsics_free: bool
    observations: int
    best_s: float
    median_s: float
    lm_trials: list[int]
    pcg_iterations: list[int]
    best_ms_per_pcg: float
    """Solver time of the best run over its PCG iterations, everything per LM trial included."""
    best_ms_per_lm: float
    median_ms_per_pcg: float
    final_cost: float
    exit: str


def build(name: str, intrinsics_free: bool) -> BaProblem:
    """The named problem, observations sorted by camera then frame (the order the scatter pre-reduction wants)."""
    model, arguments = PROBLEMS[name]
    sizes: dict[str, object] = {"n_frames": 100, "n_points": 20000, "visible_fraction": 0.15, **arguments}
    focal_noise = 0.01 if intrinsics_free else 0.0
    problem = make_problem(model, seed=0, intrinsics_free=intrinsics_free, focal_noise=focal_noise, **sizes)[0]  # type: ignore[arg-type]
    order = np.lexsort((problem.observation_frame, problem.observation_camera))
    return dataclasses.replace(
        problem,
        pixels=problem.pixels[order],
        observation_frame=problem.observation_frame[order],
        observation_camera=problem.observation_camera[order],
        observation_point=problem.observation_point[order],
    )


def time_case(name: str, intrinsics_free: bool, config: Config) -> Timing:
    """Best and median solver time of one case, with the LM and PCG counts of every run."""
    problem = build(name, intrinsics_free)
    with open(config.lock, "a") if config.lock else contextlib.nullcontext() as lock:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
        solve(problem)
        results = [solve(problem) for _ in range(config.runs)]
        if config.profile:
            os.environ["CASPAR_MOJO_PROF"] = "1"
            try:
                solve(problem)
            finally:
                del os.environ["CASPAR_MOJO_PROF"]
    seconds = [r.solve_seconds for r in results]
    best = min(range(len(results)), key=lambda i: seconds[i])
    return Timing(
        problem=name,
        intrinsics_free=intrinsics_free,
        observations=len(problem.pixels),
        best_s=seconds[best],
        median_s=statistics.median(seconds),
        lm_trials=[len(r.trace) for r in results],
        pcg_iterations=[r.pcg_iterations for r in results],
        best_ms_per_pcg=1e3 * seconds[best] / results[best].pcg_iterations,
        best_ms_per_lm=1e3 * seconds[best] / len(results[best].trace),
        median_ms_per_pcg=statistics.median(1e3 * r.solve_seconds / r.pcg_iterations for r in results),
        final_cost=results[best].final_cost,
        exit=results[best].exit_reason,
    )


def main(config: Config) -> None:
    """Time every case and print one line each."""
    rows: list[Timing] = []
    for name in config.problems:
        for intrinsics_free in config.intrinsics_free:
            start = time.perf_counter()
            row = time_case(name, intrinsics_free, config)
            rows.append(row)
            print(
                f"{name:18s} intr={'free ' if intrinsics_free else 'fixed'} obs={row.observations:>9,} "
                f"best={1e3 * row.best_s:7.1f} ms median={1e3 * row.median_s:7.1f} ms "
                f"ms/PCG={row.best_ms_per_pcg:.3f} (median {row.median_ms_per_pcg:.3f}) ms/LM={row.best_ms_per_lm:.2f} "
                f"LM={row.lm_trials} PCG={row.pcg_iterations} {row.exit} ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(json.dumps([dataclasses.asdict(row) for row in rows], indent=1))


def parse(argv: list[str] | None = None) -> Config:
    """`Config` from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--intrinsics", nargs="+", choices=["fixed", "free"], default=["fixed", "free"])
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args(argv)
    return Config(tuple(args.problems), tuple(i == "free" for i in args.intrinsics), args.runs, args.profile, args.output, args.lock)
