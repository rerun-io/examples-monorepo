"""Golden per-image parity with the historical published checkpoint evaluator."""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32
from serde import serde
from serde.json import from_json

from gsplat_rust_renderer.metrics import load_image_rgb, psnr, ssim


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Metrics:
    """Mean image metrics from the Rust evaluator."""
    psnr: float
    """Peak signal-to-noise ratio in dB."""
    ssim: float
    """Structural similarity."""
    lpips: float | None = None
    """Optional perceptual distance."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ViewMetrics:
    """One image's Rust evaluation."""
    name: str
    """Relative PNG path."""
    psnr: float
    """Peak signal-to-noise ratio in dB."""
    ssim: float
    """Structural similarity."""
    lpips: float | None = None
    """Optional perceptual distance."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DependencyVersion:
    """Resolved Cargo dependency."""
    version: str
    """Crate version."""
    source: str
    """Registry or exact git source."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Versions:
    """Build provenance from the Rust evaluator."""
    evaluator: str
    """Evaluator package version."""
    ours: str
    """Build-time git description including dirty state."""
    profile: str
    """Cargo build profile."""
    release_settings: str
    """Workspace release settings."""
    dependencies: dict[str, DependencyVersion]
    """Resolved renderer dependencies."""
    environment: dict[str, str | None]
    """Backend environment overrides at run time."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Evaluation:
    """Rust evaluator report."""
    views: list[ViewMetrics]
    """Per-image measurements."""
    mean: Metrics
    """Arithmetic per-image mean."""
    convention: str
    """Metric convention name."""
    versions: Versions
    """Tool versions."""


@pytest.mark.golden
@pytest.mark.parametrize("scene", ["lego", "hotdog", "chair", "drums", "ficus", "materials", "mic", "ship"])
def test_published_checkpoint_parity(scene: str, tmp_path: Path) -> None:
    """Every downloaded checkpoint image meets the requested Rust/Python tolerance."""
    binary: Path = Path(os.environ.get("GSPLAT_EVAL_BIN", "target/release/gsplat-eval")).resolve()
    root: Path = Path(os.environ.get("GSPLAT_CHECKPOINT_ROOT", "data/nerfbaselines/pretrained"))
    predictions: Path = root / scene / "predictions"
    if not binary.is_file():
        pytest.fail(f"Rust evaluator binary missing: {binary}; run gsplat-bench-build")
    if not (predictions / "color").is_dir() or not (predictions / "gt-color").is_dir():
        pytest.skip(f"Checkpoint prediction/GT asset missing: {predictions}")
    report: Path = tmp_path / f"{scene}.json"
    subprocess.run([
        str(binary), "dirs", "--render", str(predictions / "color"), "--gt", str(predictions / "gt-color"),
        "--convention", "published", "--out", str(report),
    ], check=True)
    result: Evaluation = from_json(Evaluation, report.read_text())
    max_psnr: float = 0.0
    max_ssim: float = 0.0
    for view in result.views:
        rendered: Float32[np.ndarray, "h w 3"] = load_image_rgb(predictions / "color" / view.name)
        gt: Float32[np.ndarray, "h w 3"] = load_image_rgb(predictions / "gt-color" / view.name)
        psnr_diff: float = abs(view.psnr - psnr(rendered, gt))
        ssim_diff: float = abs(view.ssim - ssim(rendered, gt))
        max_psnr = max(max_psnr, psnr_diff)
        max_ssim = max(max_ssim, ssim_diff)
        assert psnr_diff <= 1e-6, (scene, view.name, psnr_diff)
        assert ssim_diff <= 1e-7, (scene, view.name, ssim_diff)
    print(f"{scene}: {len(result.views)} views, maximum PSNR difference {max_psnr:.12g}, SSIM {max_ssim:.12g}")
