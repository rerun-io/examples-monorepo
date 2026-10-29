"""Results of the 2D Image-GS benchmark: one `metrics.json` per run directory.

A 2D run directory `runs/imagegs/<stem>/<method>/` scores one decoder (upstream CUDA or our WGSL)
on one fit under one or more conditions: the paper's packed file (`decoded_hxm`), the uniform
16-bit file (`uint16`) and the raw FP32 fields (`raw_fp32`). The primary condition's 16-bit render is
`renders/test/<stem>.png`; the others are `renders/<condition>/<stem>.png`.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from jaxtyping import Float32, Float64, UInt16
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from hexels_bench.layout import read_json
from hexels_bench.metrics import TrainRecord
from hexels_bench.paper import PaperNumbers, PublishedImage, Table1Row

Condition: TypeAlias = Literal["decoded_hxm", "uint16", "raw_fp32"]
Decoder: TypeAlias = Literal["cuda", "wgsl"]
Category: TypeAlias = Literal["anime", "art", "painting", "photo", "vector"]
CATEGORIES: tuple[Category, ...] = ("anime", "art", "painting", "photo", "vector")
ERROR_RANGE: float = 0.10
"""Error maps saturate at this per-pixel RGB mean absolute error."""
IDENTICAL_PSNR: float = 200.0
"""PSNR reported for two bit-identical renders (MSE floored at 1e-20)."""
TABLE_METHODS: tuple[tuple[str, Condition | None], ...] = (
    ("imagegs-2d", None),
    ("imagegs-2d", "raw_fp32"),
    ("hexels-2d-flat-pub-cuda", None),
    ("hexels-2d-flat-pub-cuda", "raw_fp32"),
    ("hexels-2d-flat-pub-wgsl", None),
    ("hexels-2d-pub-cuda", None),
    ("hexels-2d-pub-cuda", "raw_fp32"),
    ("hexels-2d-pub-wgsl", None),
    ("hexels-2d-pub-wgsl", "raw_fp32"),
    ("hexels-2d", None),
    ("hexels-2d", "raw_fp32"),
    ("hexels-2d-wgsl", None),
    ("hexels-2d-flat", None),
    ("hexels-2d-flat-wgsl", None),
)
"""Table rows: (method, condition); None is the method's primary condition (the stored file)."""
FitKind: TypeAlias = Literal["pub-rich", "pub-flat", "rich", "flat", "imagegs"]
FitMethod: TypeAlias = Literal["hexels-2d", "hexels-2d-flat", "imagegs-2d"]
PAPER_ROW: dict[FitMethod, str] = {"imagegs-2d": "Image-GS", "hexels-2d-flat": "Hexels (constant)", "hexels-2d": "Hexels"}
"""Paper Table 1 row of each recipe."""


@dataclass(frozen=True, slots=True)
class FitSpec:
    """One 2D fit: a recipe, ours or the published one, and the run directories its decodes go to."""

    family: FitMethod
    """Our fit of the same recipe: its run directory, and the recipe's Table 1 row."""
    published: bool
    """The HF jhu-ccvl/Hexels fit of the paper's recipe, not ours."""
    cuda_method: str
    """Run directory of the upstream CUDA decode (for our fits also the fit itself)."""
    wgsl_method: str | None
    """Run directory of the WGSL decode; None for Image-GS (no WGSL decoder)."""
    recipe: str
    """Upstream `scripts.fit_image --variant`."""
    appearance: str | None
    """HF appearance name of the Hexels recipe (spatial-rgb, flat-rgb); None for Image-GS."""
    packed_profile: str | None
    """Profile of the paper's packed Hexels file (mixed14, balanced); None for Image-GS (uint16 .hxi)."""
    floats_per_primitive: int
    """Stored FP32 floats per primitive."""


FITS: dict[FitKind, FitSpec] = {
    "pub-rich": FitSpec("hexels-2d", True, "hexels-2d-pub-cuda", "hexels-2d-pub-wgsl", "rich", "spatial-rgb", "mixed14", 15),
    "pub-flat": FitSpec("hexels-2d-flat", True, "hexels-2d-flat-pub-cuda", "hexels-2d-flat-pub-wgsl", "flat", "flat-rgb", "balanced", 8),
    "rich": FitSpec("hexels-2d", False, "hexels-2d", "hexels-2d-wgsl", "rich", "spatial-rgb", "mixed14", 15),
    "flat": FitSpec("hexels-2d-flat", False, "hexels-2d-flat", "hexels-2d-flat-wgsl", "flat", "flat-rgb", "balanced", 8),
    "imagegs": FitSpec("imagegs-2d", False, "imagegs-2d", None, "imagegs", None, None, 8),
}
FIT_OF_RUN: dict[str, FitSpec] = {method: spec for spec in FITS.values() for method in (spec.cuda_method, spec.wgsl_method) if method is not None}
"""2D run directory -> the fit it decodes."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ConditionScores:
    """One decoded condition of a fit, scored against the ground truth."""

    condition: Condition
    """Which stored fields were decoded."""
    source: str
    """The decoded file."""
    render: str
    """16-bit PNG of the decode, relative to the run directory."""
    psnr: float
    """PSNR in dB: float64 MSE over all pixels and channels after one clamp to [0, 1]."""
    ssim: float
    """fused_ssim (11x11 Gaussian, sigma 1.5) after the same clamp."""
    lpips_alex: float
    """LPIPS AlexNet on inputs mapped to [-1, 1]."""
    fps: float | None
    """1 / median of 5 trials of 20 decodes after 10 warm-up decodes; null when not timed."""
    trial_ms: list[float]
    """Mean milliseconds per decode of each timing trial; empty when not timed."""
    reference_psnr: float | None
    """Published (HF catalog) or upstream-recorded PSNR of the same file, when one exists."""
    vs_cuda_psnr: float | None
    """WGSL runs: PSNR of this render against the CUDA render of the same condition (200 = identical)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ImageRun:
    """`metrics.json` of one 2D run directory."""

    method: str
    """Run method: hexels-2d, hexels-2d-wgsl, imagegs-2d, hexels-2d-pub-cuda, hexels-2d-pub-wgsl, ..."""
    image: str
    """Image stem, e.g. anime-1_2k."""
    category: Category
    """Image category (the stem's prefix)."""
    decoder: Decoder
    """Upstream CUDA kernels or our WGSL renderer."""
    gpu: str
    """GPU that decoded and timed."""
    host: str
    """Host name."""
    primitives: int
    """Primitives in the fit."""
    bytes_fp32: int
    """FP32 learned parameter bytes: primitives x stored floats x 4."""
    bytes_packed: int | None
    """Size of the paper's packed file (.hxm for Hexels, uint16 .hxi for Image-GS)."""
    primary: Condition
    """The condition the tables report."""
    conditions: list[ConditionScores]
    """Every scored condition; exactly one is `primary`."""

    def __post_init__(self) -> None:
        names: list[Condition] = [scores.condition for scores in self.conditions]
        if len(set(names)) != len(names) or self.primary not in names:
            raise ValueError(f"{self.image}/{self.method}: conditions {names} must be unique and contain {self.primary}")

    def scores(self, condition: Condition | None = None) -> ConditionScores:
        wanted: Condition = self.primary if condition is None else condition
        return next(scores for scores in self.conditions if scores.condition == wanted)


def image_category(stem: str) -> Category:
    prefix: str = stem.split("-", 1)[0]
    for category in CATEGORIES:
        if prefix == category:
            return category
    raise ValueError(f"{stem}: not an Image-GS benchmark stem")


def write_image_run(path: Path, run: ImageRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(run) + "\n")


def load_image_run(path: Path) -> ImageRun:
    return from_json(ImageRun, path.read_text())


def published_scores(item: PublishedImage, condition: Condition) -> tuple[float | None, float | None, float | None]:
    """HF catalog PSNR, SSIM and LPIPS-Alex of the published file matching `condition` (packed .hxm, uniform 16-bit)."""
    if condition == "decoded_hxm":
        return item.psnr_hxm, item.ssim_hxm, item.lpips_alex_hxm
    if condition == "uint16":
        return item.psnr_uint16, None, None
    return None, None, None


def psnr_between(first: UInt16[ndarray, "h w 3"], second: UInt16[ndarray, "h w 3"]) -> float:
    """PSNR of two 16-bit renders in [0, 1] units; IDENTICAL_PSNR when they are bit-identical."""
    difference: Float64[ndarray, "h w 3"] = (first.astype(np.float64) - second.astype(np.float64)) / 65535.0
    mse: float = float(np.mean(np.square(difference)))
    return -10.0 * math.log10(max(mse, 10.0 ** (-IDENTICAL_PSNR / 10.0)))


def mae_map(prediction: Float32[ndarray, "h w 3"], target: Float32[ndarray, "h w 3"]) -> Float32[ndarray, "h w"]:
    """Per-pixel RGB mean absolute error after one clamp of the prediction (the Figure 2 error maps)."""
    return np.mean(np.abs(np.clip(prediction, 0.0, 1.0) - target), axis=2, dtype=np.float32)


def worst_window(gain: Float32[ndarray, "h w"], size: int, stride: int) -> tuple[int, int]:
    """Top-left (x, y) of the `size` x `size` window, on a `stride` grid, with the largest sum of `gain`."""
    integral: Float64[ndarray, "h1 w1"] = np.pad(np.cumsum(np.cumsum(gain.astype(np.float64), axis=0), axis=1), ((1, 0), (1, 0)))
    height: int = gain.shape[0]
    width: int = gain.shape[1]
    ys: ndarray = np.arange(0, height - size + 1, stride)
    xs: ndarray = np.arange(0, width - size + 1, stride)
    sums: Float64[ndarray, "ny nx"] = (
        integral[ys[:, None] + size, xs[None, :] + size]
        - integral[ys[:, None], xs[None, :] + size]
        - integral[ys[:, None] + size, xs[None, :]]
        + integral[ys[:, None], xs[None, :]]
    )
    row, column = divmod(int(np.argmax(sums)), sums.shape[1])
    return int(xs[column]), int(ys[row])


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class TableRow:
    """Scores of one method and condition on one image, or their mean over every image (`group = all`)."""

    method: str
    """Run directory name."""
    condition: Condition
    """Decoded condition."""
    group: str
    """Image stem, or `all`."""
    images: int
    """Images averaged."""
    psnr: float
    """PSNR, dB."""
    ssim: float
    """SSIM."""
    lpips_alex: float
    """LPIPS-Alex."""
    fps: float | None
    """Decode FPS (null when any image is untimed)."""
    train_seconds: float | None
    """Optimisation-loop seconds (our fits only)."""
    peak_vram_gib: float | None
    """Peak training VRAM, GiB (our fits only)."""
    train_gpu: str | None
    """GPU the fits ran on."""
    published_psnr: float | None
    """HF catalog PSNR of the published fit of the same images, method family and condition (Hexels .hxm and uint16)."""
    published_ssim: float | None
    """HF catalog SSIM, .hxm condition only."""
    published_lpips_alex: float | None
    """HF catalog LPIPS-Alex, .hxm condition only."""
    published_fps_a5000: float | None
    """HF catalog uint16 decode FPS on the authors' RTX A5000."""
    table1_psnr: float | None
    """Paper Table 1 (all 45 images) PSNR of the method family; `all` rows only."""
    table1_ssim: float | None
    """Paper Table 1 SSIM; `all` rows only."""
    table1_lpips_alex: float | None
    """Paper Table 1 LPIPS-Alex; `all` rows only."""


def mean_or_none(values: list[float | None]) -> float | None:
    return None if not values or any(value is None for value in values) else float(np.mean([value for value in values if value is not None]))


def table_rows(image_root: Path, stems: list[str], paper: PaperNumbers) -> list[TableRow]:
    """Per-image rows and an all-image mean for every TABLE_METHODS entry found under `image_root` (runs/imagegs).

    Hexels rows carry the HF catalog numbers of the published fit of the same image (compare
    per image, not with dataset means); `all` rows also carry the paper's 45-image Table 1 value.
    """
    rows: list[TableRow] = []
    for method, wanted in TABLE_METHODS:
        runs: list[ImageRun] = [
            load_image_run(image_root / stem / method / "metrics.json") for stem in stems if (image_root / stem / method / "metrics.json").is_file()
        ]
        runs = [run for run in runs if wanted is None or any(scores.condition == wanted for scores in run.conditions)]
        if not runs:
            continue
        condition: Condition = runs[0].primary if wanted is None else wanted
        fit: FitSpec = FIT_OF_RUN[method]
        trains: dict[str, TrainRecord] = {}
        for run in runs:
            train: Path = image_root / run.image / fit.family / "train.json"
            if not fit.published and train.is_file():  # published fits have no training run of ours
                trains[run.image] = read_json(TrainRecord, train)
        table1: Table1Row | None = next((row for row in paper.table1_2d if row.method == PAPER_ROW[fit.family]), None)
        for group in [*(run.image for run in runs), "all"]:
            members: list[ImageRun] = [run for run in runs if group in ("all", run.image)]
            scores: list[ConditionScores] = [run.scores(condition) for run in members]
            catalog: list[PublishedImage] = [paper.hf_images_2d_per_image[run.image][fit.appearance] for run in members if fit.appearance is not None]
            published: list[tuple[float | None, float | None, float | None]] = [published_scores(item, condition) for item in catalog]
            member_trains: list[TrainRecord] = [trains[run.image] for run in members if run.image in trains]
            complete: bool = len(member_trains) == len(members)
            whole: bool = group == "all" and table1 is not None
            rows.append(
                TableRow(
                    method=method,
                    condition=condition,
                    group=group,
                    images=len(members),
                    psnr=float(np.mean([s.psnr for s in scores])),
                    ssim=float(np.mean([s.ssim for s in scores])),
                    lpips_alex=float(np.mean([s.lpips_alex for s in scores])),
                    fps=mean_or_none([s.fps for s in scores]),
                    train_seconds=float(np.mean([t.loop_seconds for t in member_trains])) if complete else None,
                    peak_vram_gib=mean_or_none([t.peak_vram_gib for t in member_trains]) if complete else None,
                    train_gpu=member_trains[0].gpu if complete else None,
                    published_psnr=mean_or_none([item[0] for item in published]),
                    published_ssim=mean_or_none([item[1] for item in published]),
                    published_lpips_alex=mean_or_none([item[2] for item in published]),
                    published_fps_a5000=mean_or_none([item.median_fps_uint16_a5000 for item in catalog]),
                    table1_psnr=table1.psnr if whole and table1 is not None else None,
                    table1_ssim=table1.ssim if whole and table1 is not None else None,
                    table1_lpips_alex=table1.lpips_alex if whole and table1 is not None else None,
                )
            )
    return rows
