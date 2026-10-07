"""Golden per-image checks against the verified published metric references."""
import os
import subprocess
from pathlib import Path

import pytest
from serde.json import from_json

from gsplat_rust_renderer.evaluation import Evaluation, ViewMetrics
from gsplat_rust_renderer.nerfbaselines import BLENDER_SCENES


@pytest.mark.golden
@pytest.mark.parametrize("scene", BLENDER_SCENES)
def test_published_checkpoint_parity(scene: str, tmp_path: Path) -> None:
    """Every downloaded checkpoint image meets the requested Rust/Python tolerance."""
    binary: Path = Path(os.environ.get("GSPLAT_BIN", "target/release/gsplat")).resolve()
    root: Path = Path(os.environ.get("GSPLAT_CHECKPOINT_ROOT", "data/nerfbaselines/pretrained"))
    predictions: Path = root / scene / "predictions"
    if not binary.is_file():
        pytest.fail(f"Rust evaluator binary missing: {binary}; run gsplat-build")
    if not (predictions / "color").is_dir() or not (predictions / "gt-color").is_dir():
        pytest.skip(f"Checkpoint prediction/GT asset missing: {predictions}")
    report: Path = tmp_path / f"{scene}.json"
    subprocess.run([
        str(binary), "eval", "--render", str(predictions / "color"), "--gt", str(predictions / "gt-color"),
        "--convention", "published", "--out", str(report),
    ], check=True)
    result: Evaluation = from_json(Evaluation, report.read_text())
    reference_path: Path = Path(__file__).parent / "fixtures/published" / f"{scene}.json"
    reference: list[ViewMetrics] = from_json(list[ViewMetrics], reference_path.read_text())
    assert len(result.views) == len(reference) == 200
    for view, expected in zip(result.views, reference, strict=True):
        assert view.name == expected.name
        assert abs(view.psnr - expected.psnr) <= 1e-6, (scene, view.name)
        assert abs(view.ssim - expected.ssim) <= 1e-7, (scene, view.name)
