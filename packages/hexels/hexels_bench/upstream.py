"""Thin adapters over upstream Hexels (the `hexels` and `scripts` modules from packages/hexels-src).

Upstream imports stay inside the functions: they need the hexels env (hexel-kernel CUDA build), while
the rest of hexels_bench (export, scorer, layout) runs anywhere torch runs.
"""

import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float32
from serde import serde

from hexels_bench.layout import Roots, read_json
from hexels_bench.png16 import from_unit_float, write_png16
from hexels_bench.surface_frames import SurfaceFrame

UPSTREAM_HEXELS_REV: str = "698d1afb0944d3e766074c3beb70362f16c7550d"
"""hexels commit packaged by packages/hexels-src/recipe.yaml (a unit test keeps them equal)."""


@serde
@dataclass(frozen=True, slots=True)
class PublishedView:
    """One per-view row of upstream's direct-float evaluation (read partially)."""

    image_name: str
    """Image stem."""
    psnr_db: float
    """Direct-float PSNR."""
    ssim: float
    """2DGS SSIM."""
    lpips_vgg: float
    """LPIPS-VGG."""


@serde
@dataclass(frozen=True, slots=True)
class UpstreamBenchmark:
    """`benchmark` of upstream's evaluator result: the paper FPS protocol (read partially)."""

    median_fps: float
    """Median of the trial frame rates."""


@serde
@dataclass(frozen=True, slots=True)
class PublishedEvaluation:
    """Upstream's direct-float scores of one checkpoint: HF `evaluation.json` beside each published checkpoint, or the
    result of `scripts.evaluate_nvs_campaign evaluate` (read partially)."""

    per_view: list[PublishedView]
    """Per-view scores in render order."""
    benchmark: UpstreamBenchmark | None = None
    """The evaluator's FPS benchmark; HF evaluation.json has none."""


@serde
@dataclass(frozen=True, slots=True)
class TrainScenePlan:
    """What `scripts.train_scene --prepare` prints: the resolved training command."""

    command: list[str]
    """Trainer command line."""
    cwd: str
    """Directory to run it in."""


@serde
@dataclass(frozen=True, slots=True)
class TrainSummary:
    """`summary.json` of an upstream training run (read partially)."""

    train_seconds: float
    """Wall time of the optimisation loop."""


@runtime_checkable
class UpstreamCamera(Protocol):
    """The part of upstream's 2DGS `Camera` this module reads."""

    image_name: str


@dataclass(frozen=True, slots=True)
class UpstreamRenderer:
    """An upstream model ready to render its test split."""

    render: Callable[[UpstreamCamera], torch.Tensor]
    """camera -> float CHW image, unclamped (upstream `Hexels3DRenderer.render(...)["image"]`)."""
    cameras: list[UpstreamCamera]
    """Upstream 2DGS test cameras in render order (sorted by image name, every llffhold-th)."""


def _load(model_dir: Path, checkpoint: Path, scene_dir: Path, *split: str) -> tuple:
    """`(model, scene, renderer, background)` from upstream's own loader (`hexels/render.py`
    `_load_render_components`), with the dataset read from `scene_dir` instead of the config's relative source_path."""
    from hexels.render import RenderOptions, _load_render_components

    arguments: list[str] = ["--checkpoint", str(checkpoint), "--model_path", str(model_dir), "--source_path", str(scene_dir), *split, "--skip_path"]
    return _load_render_components(RenderOptions().parse_args(arguments))[1:5]


def load_renderer(model_dir: Path, checkpoint: Path, scene_dir: Path) -> UpstreamRenderer:
    """An upstream run ready to render its test split."""
    model, scene, renderer, background = _load(model_dir, checkpoint, scene_dir, "--split", "test")

    def render(camera: UpstreamCamera) -> torch.Tensor:
        return renderer.render(camera, model, background)["image"]

    return UpstreamRenderer(render=render, cameras=list(scene.test_cameras))


@torch.no_grad()
def surface_frames(model_dir: Path, checkpoint: Path, scene_dir: Path) -> Iterator[SurfaceFrame]:
    """Every training view rendered by upstream CUDA with its surface outputs, as upstream's mesh
    extraction renders them (`scripts/extract_hexels_dtu_mesh.py` L.73-80 and L.103-128): all registered
    views for all-view checkpoints, color `clamp(0, 1) * 255` truncated, expected depth
    `surf_depth / max(surf_alpha, 1e-6)`. Masking and the alpha cut happen in our fusion code."""
    model, scene, renderer, background = _load(model_dir, checkpoint, scene_dir, "--split", "train", "--skip_test")
    for camera in scene.train_cameras:
        package: dict[str, torch.Tensor] = renderer.render(camera, model, background, return_surface=True, return_median_depth=True)
        alpha: torch.Tensor = package["surf_alpha"]
        yield SurfaceFrame(
            name=str(camera.image_name),
            rgb=package["image"].clamp(0, 1).mul(255).byte().permute(1, 2, 0).contiguous().cpu().numpy(),
            median_depth=package["surf_median_depth"].squeeze(0).float().contiguous().cpu().numpy(),
            expected_depth=(package["surf_depth"] / alpha.clamp_min(1e-6)).squeeze(0).float().contiguous().cpu().numpy(),
            alpha=alpha.squeeze(0).float().contiguous().cpu().numpy(),
        )


@torch.no_grad()
def write_test_renders(renderer: UpstreamRenderer, names: list[str], out_dir: Path) -> None:
    """Render every test camera and write `<name>.png` as 16-bit RGB (clamped once to [0, 1])."""
    camera_names: list[str] = [str(camera.image_name) for camera in renderer.cameras]
    if camera_names != names:
        raise ValueError(f"upstream test cameras {camera_names} differ from cameras.json {names}")
    for camera, name in zip(renderer.cameras, names, strict=True):
        image: torch.Tensor = renderer.render(camera)
        rgb: Float32[np.ndarray, "h w 3"] = image.detach().float().permute(1, 2, 0).contiguous().cpu().numpy()
        write_png16(out_dir / f"{name}.png", from_unit_float(rgb))


def measure_fps(renderer: UpstreamRenderer) -> float:
    """Paper FPS protocol via upstream's own benchmark (2 warm-up passes, 4 trials x 10 passes,
    synchronized, median): `scripts/evaluate_nvs_campaign.py` `benchmark`."""
    from scripts.evaluate_nvs_campaign import benchmark

    options: dict[str, int] = {"warmup_passes": 2, "timed_passes": 10, "timing_trials": 4}
    return float(benchmark(renderer.render, renderer.cameras, options)["median_fps"])


def upstream_cwd(roots: Roots) -> Path:
    """A directory where upstream's relative paths resolve: `data/<dataset>/<scene>` (its configs and
    camera settings) and `pretrained/Hexels` (its evaluation jobs)."""
    cwd: Path = roots.runs / "_upstream_cwd"
    (cwd / "pretrained").mkdir(parents=True, exist_ok=True)
    for link, target in ((cwd / "data", roots.data), (cwd / "pretrained" / "Hexels", roots.data / "hexels-pretrained")):
        if not link.is_symlink():
            link.symlink_to(target, target_is_directory=True)
    return cwd


def absolute_job(job_path: Path, cwd: Path, out_dir: Path) -> Path:
    """Copy an upstream evaluation job (and its camera settings) into `out_dir` with the job's
    `pretrained/...` paths and the settings' `source_path` resolved against `cwd`.

    Upstream resolves relative paths against its own checkout root, which in our env is the installed
    `share/hexels-src` tree, not a directory that holds `data/` and `pretrained/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    job: dict = json.loads(job_path.read_text())
    for key in ("model_path", "camera_contract", "camera_settings"):
        job[key] = str((cwd / job[key]).resolve())
    settings: dict = json.loads(Path(job["camera_settings"]).read_text())
    settings["source_path"] = str((cwd / settings["source_path"]).resolve())
    settings_copy: Path = out_dir / "settings.json"
    settings_copy.write_text(json.dumps(settings, indent=2))
    job["camera_settings"] = str(settings_copy)
    job_copy: Path = out_dir / "evaluation_job.json"
    job_copy.write_text(json.dumps(job, indent=2))
    return job_copy


def run_upstream_evaluator(roots: Roots, model_dir: Path, result: Path, log: Path) -> PublishedEvaluation:
    """`python -m scripts.evaluate_nvs_campaign evaluate --job <model_dir>/evaluation_job.json` on the
    published checkpoint; returns upstream's result (direct-float metrics, benchmark FPS)."""
    if result.parent.exists():
        raise FileExistsError(f"{result.parent} exists; upstream refuses to overwrite its renders")
    cwd: Path = upstream_cwd(roots)
    job: Path = absolute_job(model_dir / "evaluation_job.json", cwd, result.parent.with_name(result.parent.name + "-job"))
    result.parent.mkdir(parents=True)
    command: list[str] = [
        sys.executable, "-m", "scripts.evaluate_nvs_campaign", "evaluate",
        "--job", str(job), "--model-path", str(model_dir), "--result", str(result),
    ]  # fmt: skip
    with log.open("a") as stream:
        subprocess.run(command, cwd=cwd, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return read_json(PublishedEvaluation, result)
