"""2D image fits on the Image-GS benchmark (2048x2048, 10K primitives), timed (specs/twod.md 1.1).

Every method runs upstream `scripts.fit_image` (hexels env), the paper's own harness:

- `hexels-2d`: `--variant rich` (20K -> 10K pruned hexels, 5,100 steps, L1 + 0.1 SSIM, seed 123,
  uint16 straight-through fitting); writes the paper's packed `step_005100_mixed14.hxm`.
- `hexels-2d-flat`: `--variant flat` (constant colour, beta = 10); packed `step_005100_balanced.hxm`.
- `imagegs-2d`: `--variant imagegs` (Image-GS `GaussianSplatting2D` with its native error-guided
  growth 5K -> 10K, 10,000 steps, same loss, seed and uint16 fitting) on hexel-kernel's Image-GS kernels.

Run directory: `runs/imagegs/<stem>/<method>/{upstream/, train.json, log.txt}`; decoding and scoring
are `apis/decode2d.py`. Ground truth: `runs/imagegs/<stem>/gt/test/<stem>.png` (the input widened to 16 bit).
"""

import json
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from PIL import Image
from serde.json import to_json

from hexels_bench.gpu_monitor import PeakVramSampler
from hexels_bench.layout import Roots
from hexels_bench.metrics import TrainRecord
from hexels_bench.png16 import write_png16
from hexels_bench.results2d import FIT_OF_RUN, FitMethod
from hexels_bench.upstream import UPSTREAM_HEXELS_REV

PRIMITIVES: int = 10000


@dataclass
class Config:
    """Fit Image-GS benchmark images with upstream's 2D recipes; one run directory per image and method."""

    method: list[FitMethod]
    """Fit methods, in order; each runs over all stems before the next (`--method imagegs-2d hexels-2d`)."""
    stems: list[str] = field(default_factory=list)
    """Image stems in fitting order, e.g. anime-1_2k; empty means all 45 in sorted order."""
    steps: int | None = None
    """Override the recipe's step count (smoke runs); None keeps 5,100 (Hexels) / 10,000 (Image-GS)."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


def image_path(roots: Roots, stem: str) -> Path:
    return roots.data / "imagegs" / "images" / f"{stem}.png"


def stage_image_gt(roots: Roots, stem: str) -> Path:
    gt: Path = roots.runs / "imagegs" / stem / "gt" / "test" / f"{stem}.png"
    if not gt.is_file():
        with Image.open(image_path(roots, stem)) as image:
            rgb: UInt8[ndarray, "h w 3"] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        write_png16(gt, rgb.astype(np.uint16) * np.uint16(257))
    return gt.parent


def fit_command(roots: Roots, stem: str, method: FitMethod, output: Path, steps: int | None) -> list[str]:
    command: list[str] = [
        sys.executable, "-m", "scripts.fit_image", "--image", str(image_path(roots, stem)),
        "--output", str(output), "--variant", FIT_OF_RUN[method].recipe,
    ]  # fmt: skip
    return command if steps is None else [*command, "--steps", str(steps)]


def run_measured(command: list[str], cwd: Path, log: Path) -> tuple[float, float]:
    """Run `command` as a direct child; returns (wall seconds, peak VRAM GiB)."""
    started: float = time.perf_counter()
    with log.open("a") as stream:
        stream.write(" ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        with PeakVramSampler(pid=process.pid) as sampler:
            return_code: int = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{command[1:4]} exited with {return_code}; see {log}")
    return time.perf_counter() - started, sampler.peak_gib


def fit(roots: Roots, method: FitMethod, stem: str, steps: int | None) -> TrainRecord:
    run_dir: Path = roots.runs / "imagegs" / stem / method
    output: Path = run_dir / "upstream"
    if output.exists():
        shutil.rmtree(output)  # an interrupted fit of ours; upstream needs a fresh output directory
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_image_gt(roots, stem)
    command: list[str] = fit_command(roots, stem, method, output, steps)
    total_seconds, peak_gib = run_measured(command, run_dir, run_dir / "log.txt")
    upstream: dict = json.loads((output / "metrics.json").read_text())
    record = TrainRecord(
        method=method,
        dataset="imagegs",
        scene=stem,
        steps=int(upstream["job"]["max_steps"]),
        loop_seconds=float(upstream["training_seconds"]),
        total_seconds=total_seconds,
        peak_vram_gib=peak_gib,
        primitives=PRIMITIVES,
        sfm_points=None,
        gpu=str(upstream["gpu"]),
        host=socket.gethostname(),
        source_sha=UPSTREAM_HEXELS_REV,
        command=" ".join(command),
    )
    (run_dir / "train.json").write_text(to_json(record) + "\n")
    return record


def is_fitted(roots: Roots, stem: str, method: FitMethod) -> bool:
    run_dir: Path = roots.runs / "imagegs" / stem / method
    return (run_dir / "train.json").is_file() and (run_dir / "upstream" / "metrics.json").is_file()


def main(config: Config) -> None:
    stems: list[str] = config.stems or sorted(path.stem for path in (config.roots.data / "imagegs" / "images").glob("*_2k.png"))
    failures: Path = config.roots.runs / "imagegs" / "FAILURES.jsonl"
    for method in config.method:  # method by method: the missing Image-GS numbers come first
        for stem in stems:
            if is_fitted(config.roots, stem, method):
                print(f"skip {stem} {method}: fitted", flush=True)
                continue
            try:
                record: TrainRecord = fit(config.roots, method, stem, config.steps)
            except RuntimeError as error:  # a failed upstream fit: record it and continue with the queue
                failures.parent.mkdir(parents=True, exist_ok=True)
                with failures.open("a") as stream:
                    stream.write(json.dumps({"stem": stem, "method": method, "host": socket.gethostname(), "error": str(error)}) + "\n")
                print(f"FAILED {stem} {method}: {error}", flush=True)
                continue
            print(
                f"{stem} {method}: loop {record.loop_seconds:.1f} s, total {record.total_seconds:.1f} s, peak {record.peak_vram_gib:.2f} GiB",
                flush=True,
            )
