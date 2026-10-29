"""Integration against upstream Hexels and its CUDA kernels (hexels env, a GPU and the published data)."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("hexels.model", reason="upstream Hexels comes from the hexels-src conda outputs (hexels env)")
pytest.importorskip("gsplat.cuda_hexels3d", reason="hexel-kernel CUDA build (hexels env)")

from hexels_bench.apis.render_published import render_published  # noqa: E402
from hexels_bench.checkpoint import find_checkpoint, load_checkpoint, load_run_config  # noqa: E402
from hexels_bench.hexel_ply import HexelVariant  # noqa: E402
from hexels_bench.layout import (  # noqa: E402
    Dataset,
    Roots,
    read_json,  # noqa: E402
)
from hexels_bench.metrics import MetricsFile  # noqa: E402

pytestmark = pytest.mark.integration
SCAN24_V3: tuple[Dataset, str, HexelVariant] = ("dtu", "scan24", "v3")


def needs_data(*markers: str) -> Roots:
    roots = Roots()
    for marker in markers:
        if not (roots.data / "READY" / marker).is_file():
            pytest.skip(f"needs {roots.data}/READY/{marker}")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return roots


def test_export_matches_upstream_pack() -> None:
    """Our activations equal upstream `Hexels3DModel.pack_for_rasterizer` on a published checkpoint."""
    from hexels.render import RenderOptions, _load_render_components

    roots: Roots = needs_data("pretrained-nvs", "dtu")
    model_dir: Path = roots.published_model(*SCAN24_V3)
    checkpoint: Path = find_checkpoint(model_dir)
    options = RenderOptions().parse_args(
        ["--checkpoint", str(checkpoint), "--model_path", str(model_dir), "--source_path", str(roots.scene_dir("dtu", "scan24")), "--skip_path"]
    )
    model = _load_render_components(options)[1]
    pack: dict[str, torch.Tensor] = model.pack_for_rasterizer(child_offset_frac=0.5, child_scale_ratio=0.5)
    cloud = load_checkpoint(checkpoint, load_run_config(model_dir / "config.json")).cloud
    assert cloud.sh_coeffs is not None
    for ours, name in (
        (cloud.centers_xyz, "means3d"),
        (cloud.quats_wxyz, "rotations"),
        (cloud.scales_uv, "scales"),
        (cloud.opacities, "opacities"),
        (cloud.sharpness, "softnesses"),
        (cloud.base_rgb, "colors"),
        (cloud.sh_coeffs, "view_grads"),
    ):
        np.testing.assert_allclose(ours, pack[name].detach().cpu().numpy(), rtol=1e-6, atol=1e-7, err_msg=name)


def test_render_published_scan24_v3_reproduces_published_metrics(tmp_path: Path) -> None:
    """Upstream CUDA renders of the published checkpoint score within 0.05 dB of the HF numbers, and the
    shared 16-bit PNG scorer agrees with upstream's direct-float evaluator to < 0.01 dB."""
    roots = Roots(data=needs_data("pretrained-nvs", "dtu").data, runs=tmp_path)
    shared: MetricsFile = render_published(roots, *SCAN24_V3, upstream_evaluator=True)
    run_dir: Path = roots.run_dir("dtu", "scan24", "hexels-v3-30k-pub-cuda")
    published: MetricsFile = read_json(MetricsFile, run_dir / "published_metrics.json")
    upstream: MetricsFile = read_json(MetricsFile, run_dir / "upstream_float_metrics.json")
    assert abs(upstream.psnr - published.psnr) < 0.05
    assert abs(shared.psnr - upstream.psnr) < 0.01
