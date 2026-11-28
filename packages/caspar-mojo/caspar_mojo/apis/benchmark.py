"""Solver timing on in-memory synthetic problems shaped like the harness's COLMAP models (no pycolmap needed).

P1 is `make_problem(PINHOLE, 400 frames, 40,000 points, seed 0, visible 0.15)`: 2,398,861 observations, the harness's
p1 model. The `*-100` problems match p3/p4/p5 (100 frames, 20,000 points, about 300 k observations) and `rig-100`
p6 (two cameras, the second's extrinsic refined). `clip007-shape` is p2 (clip 007): 524 frames of a fixed 6-camera
rig, about 380 observations per image and 4 per point, so warp tiles rather than whole blocks share an image. Every
LM trial count and PCG iteration count is printed next to its time: after convergence float32 noise decides the tail
length, so time per PCG iteration is the fair comparison.
"""

import contextlib
import dataclasses
import fcntl
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from serde import serde
from serde.json import to_json

from caspar_mojo.problem import BaProblem, CameraModel
from caspar_mojo.reference.synthetic import make_problem
from caspar_mojo.solver import solve

ProblemName = Literal["p1", "pinhole-100", "simple-radial-100", "fisheye-100", "full-opencv-100", "rig-100", "clip007-shape"]


@dataclass(frozen=True, slots=True)
class Case:
    """`make_problem` arguments of one benchmark problem (seed 0)."""

    model: CameraModel
    n_frames: int = 100
    n_points: int = 20000
    visible_fraction: float = 0.15
    rig_cameras: int = 1
    sensor_free: bool = False
    frame_window: int | None = None


PROBLEMS: dict[ProblemName, Case] = {
    "p1": Case(CameraModel.PINHOLE, n_frames=400, n_points=40000),
    "pinhole-100": Case(CameraModel.PINHOLE),
    "simple-radial-100": Case(CameraModel.SIMPLE_RADIAL),
    "fisheye-100": Case(CameraModel.OPENCV_FISHEYE),
    "full-opencv-100": Case(CameraModel.FULL_OPENCV),
    "rig-100": Case(CameraModel.PINHOLE, rig_cameras=2, sensor_free=True),
    "clip007-shape": Case(CameraModel.PINHOLE, n_frames=524, n_points=300000, visible_fraction=0.5, rig_cameras=6, frame_window=1100),
}


@dataclass(frozen=True, slots=True)
class Config:
    """What to time."""

    problems: tuple[ProblemName, ...] = tuple(PROBLEMS)
    """Problems to time."""
    intrinsics: tuple[Literal["fixed", "free"], ...] = ("fixed", "free")
    """Run each problem with the focal-and-extra block fixed, refined, or both."""
    runs: int = 7
    """Timed solves per case, after one untimed warm-up."""
    profile: bool = False
    """One more solve per case with CASPAR_MOJO_PROF=1 (a sync after every kernel, so never timed)."""
    output: Path | None = None
    """JSON file for the result rows (outside the repository)."""
    lock: Path | None = None
    """File to flock around each case's solves (a GPU shared between jobs), not around the problem building."""

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError(f"runs must be >= 1, got {self.runs}")


@serde(deny_unknown_fields=True)
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


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TimingTable:
    """The `--output` file."""

    rows: list[Timing]


def build(name: ProblemName, intrinsics_free: bool) -> BaProblem:
    """The named problem, observations sorted by camera then frame (the order the solver would sort them into)."""
    case = PROBLEMS[name]
    problem = make_problem(
        case.model,
        n_frames=case.n_frames,
        n_points=case.n_points,
        seed=0,
        rig_cameras=case.rig_cameras,
        focal_noise=0.01 if intrinsics_free else 0.0,
        intrinsics_free=intrinsics_free,
        sensor_free=case.sensor_free,
        visible_fraction=case.visible_fraction,
        frame_window=case.frame_window,
    )[0]
    order = np.lexsort((problem.observation_frame, problem.observation_camera))
    return dataclasses.replace(
        problem,
        pixels=problem.pixels[order],
        observation_frame=problem.observation_frame[order],
        observation_camera=problem.observation_camera[order],
        observation_point=problem.observation_point[order],
    )


def time_case(name: ProblemName, intrinsics_free: bool, config: Config) -> Timing:
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
        for intrinsics in config.intrinsics:
            start = time.perf_counter()
            row = time_case(name, intrinsics == "free", config)
            rows.append(row)
            print(
                f"{name:18s} intr={intrinsics:5s} obs={row.observations:>9,} "
                f"best={1e3 * row.best_s:7.1f} ms median={1e3 * row.median_s:7.1f} ms "
                f"ms/PCG={row.best_ms_per_pcg:.3f} (median {row.median_ms_per_pcg:.3f}) ms/LM={row.best_ms_per_lm:.2f} "
                f"LM={row.lm_trials} PCG={row.pcg_iterations} {row.exit} ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(to_json(TimingTable(rows)))
