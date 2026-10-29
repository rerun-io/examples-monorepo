"""Decode 2D fits with upstream CUDA and with our WGSL renderer; score, time and compare (specs/twod.md 3).

For every image and fit, each stored condition (`decoded_hxm` = the paper's packed `.hxm`, `uint16`,
`raw_fp32`) is decoded twice from the same upstream-decoded fields:

- CUDA: upstream `render_image` / `render_parameters_image` (as `scripts.decode_image` does), scored
  in float like `scripts.evaluate_image`, timed with upstream's 10 + 5 x 20 protocol.
- WGSL: the fields as a 2D hexel PLY, `gsplat render --png16 --fps` (the same 10 + 5 x 20 protocol), scored from the 16-bit PNG,
  and compared pixel by pixel with the CUDA render of the same condition.

Image-GS fits have no WGSL decoder (specs/twod.md D5). Results: `metrics.json` (results2d.ImageRun)
in `runs/imagegs/<stem>/<method>/`. Run it in the hexels env under the GPU lock (it times decodes).
"""

import contextlib
import fcntl
import json
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, UInt16
from numpy import ndarray
from serde import SerdeError

from hexels_bench.image2d import HexelImage, hexel_image_from_fields, write_hexel_image_ply
from hexels_bench.layout import GSPLAT_RELEASE, Roots
from hexels_bench.png16 import from_unit_float, read_png16, write_png16
from hexels_bench.results2d import (
    FITS,
    Condition,
    ConditionScores,
    FitKind,
    FitSpec,
    ImageRun,
    image_category,
    load_image_run,
    psnr_between,
    write_image_run,
)
from hexels_bench.score2d import ImageScorer, ImageScores, load_target, time_decoder


@dataclass
class Config:
    """Decode, score and time 2D fits with upstream CUDA and with the WGSL renderer."""

    stems: list[str] = field(default_factory=list)
    """Image stems; empty means all 45 images."""
    kinds: list[FitKind] = field(default_factory=lambda: ["pub-rich", "pub-flat", "rich", "flat", "imagegs"])
    """Which fits: published rich/flat (HF), our rich/flat Hexel fits, our Image-GS fits. Missing fits are skipped."""
    force: bool = False
    """Redo images whose metrics.json files already exist."""
    gsplat: Path = GSPLAT_RELEASE / "gsplat"
    """Release `gsplat` binary (pixi run -e gsplat-rust-renderer gsplat-build)."""
    gpu_lock: Path | None = None
    """Hold this flock file while decoding and timing each fit, so timings do not share the GPU."""
    roots: Roots = field(default_factory=Roots)
    """Data and run roots."""


@dataclass(frozen=True, slots=True)
class FitFiles:
    """The stored files of one fit."""

    spec: FitSpec
    """Which fit, and the run directories its decodes go to."""
    files: dict[Condition, Path]
    """Stored file per condition."""
    primary: Condition
    """The condition the tables report."""
    reference_psnr: dict[Condition, float]
    """Published or upstream-recorded PSNR per condition, where one exists."""


def published_files(roots: Roots, stem: str, spec: FitSpec) -> FitFiles | None:
    assert spec.appearance is not None and spec.packed_profile is not None, "published fits are Hexels fits"
    appearance: str = spec.appearance
    packed_format: str = spec.packed_profile
    folder: Path = roots.published_image_fit(stem, appearance)
    if not folder.is_dir():
        return None
    metadata: dict = json.loads((folder / "metadata.json").read_text())
    by_format: dict[str, float] = {item["format"]: float(item["expected_metrics"]["psnr_db"]) for item in metadata["representations"]}
    return FitFiles(
        spec=spec,
        files={
            "decoded_hxm": folder / f"hexels_{appearance}_{packed_format}.hxm",
            "uint16": folder / f"hexels_{appearance}_uniform-uint16.hxc",
            "raw_fp32": folder / "hexels_raw-fp32.pt",
        },
        primary="decoded_hxm",
        reference_psnr={"decoded_hxm": by_format[packed_format], "uint16": by_format["uniform-uint16"]},
    )


def fitted_files(roots: Roots, stem: str, spec: FitSpec) -> FitFiles | None:
    upstream: Path = roots.runs / "imagegs" / stem / spec.family / "upstream"
    if not (upstream / "metrics.json").is_file():
        return None
    record: dict = json.loads((upstream / "metrics.json").read_text())
    steps: int = int(record["job"]["max_steps"])
    final: list[dict] = [checkpoint for checkpoint in record["checkpoints"] if checkpoint["step"] == steps]
    if spec.packed_profile is None:  # Image-GS stores uint16 and FP32 .hxi files
        by_bits: dict[int, float] = {int(checkpoint["bits"]): float(checkpoint["psnr_db"]) for checkpoint in final}
        return FitFiles(
            spec=spec,
            files={"uint16": upstream / f"step_{steps:06d}_16bit.hxi", "raw_fp32": upstream / f"step_{steps:06d}_32bit.hxi"},
            primary="uint16",
            reference_psnr={"uint16": by_bits[16], "raw_fp32": by_bits[32]},
        )
    profile: str = spec.packed_profile
    return FitFiles(
        spec=spec,
        files={
            "decoded_hxm": upstream / f"step_{steps:06d}_{profile}.hxm",
            "uint16": upstream / f"step_{steps:06d}_16bit.hxc",
            "raw_fp32": upstream / f"render_parameters_{steps:06d}.pt",
        },
        primary="decoded_hxm",
        reference_psnr={"uint16": float(final[0]["psnr_db"])},
    )


@dataclass(frozen=True, slots=True)
class StoredFit:
    """Upstream-decoded fields of one stored file, resident on the GPU."""

    header: dict
    """Upstream header: method, layout, width, height."""
    values: dict[str, torch.Tensor]
    """Decoded fields on the GPU."""
    legacy: bool
    """Upstream's reciprocal-scale codec (Image-GS `.hxi`)."""

    def decode(self) -> torch.Tensor:
        """The image upstream's `render_representation` returns: float32 (3, H, W), unclamped."""
        from scripts.image_compression_stable import render_image
        from scripts.image_compression_training import render_parameters_image

        height: int = int(self.header["height"])
        width: int = int(self.header["width"])
        if self.legacy:
            return render_parameters_image(self.values, self.header["method"], height, width)
        return render_image(self.values, self.header["method"], self.header["layout"], height, width)


def load_stored_fit(path: Path) -> StoredFit:
    from scripts.decode_image import load_representation

    header, arrays, legacy = load_representation(path)
    return StoredFit(header=header, values={name: value.cuda() for name, value in arrays.items()}, legacy=legacy)


def render_path(run_dir: Path, stem: str, condition: Condition, primary: Condition) -> Path:
    return run_dir / "renders" / ("test" if condition == primary else condition) / f"{stem}.png"


def condition_scores(
    condition: Condition,
    files: FitFiles,
    run_dir: Path,
    render: Path,
    scores: ImageScores,
    trial_ms: list[float],
    vs_cuda_psnr: float | None,
) -> ConditionScores:
    return ConditionScores(
        condition=condition,
        source=str(files.files[condition]),
        render=str(render.relative_to(run_dir)),
        psnr=scores.psnr,
        ssim=scores.ssim,
        lpips_alex=scores.lpips_alex,
        fps=1000.0 / float(np.median(trial_ms)),
        trial_ms=trial_ms,
        reference_psnr=files.reference_psnr.get(condition),
        vs_cuda_psnr=vs_cuda_psnr,
    )


def decode_fit(config: Config, scorer: ImageScorer, stem: str, files: FitFiles) -> list[ImageRun]:
    """Decode every condition of one fit with CUDA (and WGSL for Hexels); write both metrics.json files."""
    image_dir: Path = config.roots.runs / "imagegs" / stem
    target: torch.Tensor = load_target(config.roots.data / "imagegs" / "images" / f"{stem}.png", scorer.device)
    cuda_dir: Path = image_dir / files.spec.cuda_method
    cuda_rows: list[ConditionScores] = []
    wgsl_rows: list[ConditionScores] = []
    wgsl_adapter: str = ""
    primitives: int = 0
    for condition, path in files.files.items():
        stored: StoredFit = load_stored_fit(path)
        primitives = int(stored.values["positions"].shape[0])
        with torch.no_grad():
            image: torch.Tensor = stored.decode()
        cuda_render: Path = render_path(cuda_dir, stem, condition, files.primary)
        cuda_pixels: UInt16[ndarray, "h w 3"] = from_unit_float(image.permute(1, 2, 0).contiguous().cpu().numpy())
        write_png16(cuda_render, cuda_pixels)
        cuda_rows.append(condition_scores(condition, files, cuda_dir, cuda_render, scorer.score(image, target), time_decoder(stored.decode), None))
        if files.spec.wgsl_method is None:
            continue
        wgsl_dir: Path = image_dir / files.spec.wgsl_method
        fields: dict[str, ndarray] = {name: value.float().cpu().numpy() for name, value in stored.values.items()}
        hexels: HexelImage = hexel_image_from_fields(int(stored.header["width"]), int(stored.header["height"]), fields)
        ply: Path = wgsl_dir / ("model.ply" if condition == files.primary else f"model-{condition}.ply")
        write_hexel_image_ply(ply, hexels, str(path), int(stored.header.get("step", -1)))
        wgsl_render: Path = render_path(wgsl_dir, stem, condition, files.primary)
        timing_path: Path = wgsl_dir / "timing" / f"{condition}.json"
        wgsl_render.parent.mkdir(parents=True, exist_ok=True)
        command: list[str] = [str(config.gsplat), "render", "--ply", str(ply), "--output", str(wgsl_render), "--png16", "--fps", str(timing_path)]
        with (wgsl_dir / "log.txt").open("a") as log:
            log.write(" ".join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        timing: dict = json.loads(timing_path.read_text())
        wgsl_adapter = f"{timing['adapter']} ({timing['backend']})"
        wgsl_pixels: UInt16[ndarray, "h w 3"] = read_png16(wgsl_render)
        wgsl_image: Float32[ndarray, "h w 3"] = wgsl_pixels.astype(np.float32) / np.float32(65535.0)
        wgsl_tensor: torch.Tensor = torch.from_numpy(np.ascontiguousarray(wgsl_image.transpose(2, 0, 1)))
        wgsl_rows.append(
            condition_scores(
                condition,
                files,
                wgsl_dir,
                wgsl_render,
                scorer.score(wgsl_tensor, target),
                [trial["seconds"] * 1000.0 / trial["frames"] for trial in timing["trials"]],
                psnr_between(wgsl_pixels, cuda_pixels),
            )
        )
    packed: Path = files.files["decoded_hxm" if "decoded_hxm" in files.files else "uint16"]
    common: dict = {
        "image": stem,
        "category": image_category(stem),
        "host": socket.gethostname(),
        "primitives": primitives,
        "bytes_fp32": primitives * files.spec.floats_per_primitive * 4,
        "bytes_packed": packed.stat().st_size,
        "primary": files.primary,
    }
    runs: list[ImageRun] = [ImageRun(method=files.spec.cuda_method, decoder="cuda", gpu=torch.cuda.get_device_name(), conditions=cuda_rows, **common)]
    if files.spec.wgsl_method is not None:
        runs.append(ImageRun(method=files.spec.wgsl_method, decoder="wgsl", gpu=wgsl_adapter, conditions=wgsl_rows, **common))
    for run in runs:
        write_image_run(image_dir / run.method / "metrics.json", run)
    return runs


def is_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        load_image_run(path)
    except (SerdeError, ValueError):  # another schema (the 3D metrics file) or a partial file
        return False
    return True


def main(config: Config) -> None:
    stems: list[str] = config.stems or sorted(path.stem for path in (config.roots.data / "imagegs" / "images").glob("*_2k.png"))
    scorer = ImageScorer(torch.device("cuda"))
    for stem in stems:
        for kind in config.kinds:
            spec: FitSpec = FITS[kind]
            files: FitFiles | None = published_files(config.roots, stem, spec) if spec.published else fitted_files(config.roots, stem, spec)
            if files is None:
                continue
            outputs: list[str] = [method for method in (spec.cuda_method, spec.wgsl_method) if method is not None]
            if not config.force and all(is_done(config.roots.runs / "imagegs" / stem / method / "metrics.json") for method in outputs):
                continue
            with contextlib.ExitStack() as stack:
                if config.gpu_lock is not None:
                    lock = stack.enter_context(config.gpu_lock.open("a"))
                    fcntl.flock(lock, fcntl.LOCK_EX)  # released when the file closes
                runs: list[ImageRun] = decode_fit(config, scorer, stem, files)
            for run in runs:
                scores: ConditionScores = run.scores()
                extra: str = "" if scores.vs_cuda_psnr is None else f" vs CUDA {scores.vs_cuda_psnr:.1f} dB"
                print(f"{stem} {run.method}: {scores.psnr:.4f} dB (ref {scores.reference_psnr}) {scores.fps:.0f} FPS{extra}", flush=True)
