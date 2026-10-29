"""The 2D scorer and decode driver against upstream (hexels env, GPU, published HF fits)."""

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scripts.decode_image", reason="needs upstream hexels (hexels env)")

from hexels_bench.apis.decode2d import Config, decode_fit, published_files  # noqa: E402
from hexels_bench.layout import Roots  # noqa: E402
from hexels_bench.results2d import FITS, ImageRun  # noqa: E402
from hexels_bench.score2d import ImageScorer, ImageScores, load_target, time_decoder  # noqa: E402

STEM: str = "anime-1_2k"


@pytest.fixture(autouse=True)
def restore_upstream_import_path() -> Iterator[None]:
    """Upstream's 2D scripts put image-gs first on sys.path; its `utils` package would shadow 2DGS's `utils` for later 3D tests."""
    path: list[str] = list(sys.path)
    yield
    sys.path[:] = path
    for name in [name for name in sys.modules if name == "utils" or name.startswith("utils.")]:
        del sys.modules[name]


def published_or_skip() -> Path:
    folder: Path = Roots().data / "hexels-pretrained/images_2d" / STEM / "hexels_spatial-rgb_pruned-20000-to-10000_seed123_step005100"
    if not folder.is_dir() or not torch.cuda.is_available():
        pytest.skip(f"needs a CUDA GPU and {folder} (READY/pretrained-2d)")
    return folder


@pytest.mark.integration
def test_scorer_equals_upstream_evaluate_image() -> None:
    """Our scorer on upstream's decode of the published .hxm equals upstream's basic_metrics + LPIPS-Alex."""
    from lpips import LPIPS
    from scripts.decode_image import render_representation
    from scripts.image_compression_training import basic_metrics

    folder: Path = published_or_skip()
    device = torch.device("cuda")
    target: torch.Tensor = load_target(Roots().data / "imagegs" / "images" / f"{STEM}.png", device)
    with torch.no_grad():
        prediction: torch.Tensor = render_representation(folder / "hexels_spatial-rgb_mixed14.hxm").clamp(0, 1)
        upstream: dict = basic_metrics(prediction, target)
        lpips_alex: float = float(LPIPS(net="alex", verbose=False).cuda().eval()(prediction[None] * 2 - 1, target[None] * 2 - 1))
    ours: ImageScores = ImageScorer(device).score(prediction, target)
    expected: dict = json.loads((folder / "metadata.json").read_text())["expected_metrics"]
    assert ours.psnr == pytest.approx(upstream["psnr_db"], abs=1e-9)
    assert ours.ssim == pytest.approx(upstream["ssim"], abs=1e-7)
    assert ours.lpips_alex == pytest.approx(lpips_alex, abs=1e-7)
    assert ours.psnr == pytest.approx(expected["psnr_db"], abs=1e-3)
    trials: list[float] = time_decoder(lambda: prediction * 1.0, warmup=1, trials=3, per_trial=2)
    assert len(trials) == 3 and all(ms > 0.0 for ms in trials)


@pytest.mark.golden
def test_wgsl_decode_matches_cuda_and_the_catalog(tmp_path: Path) -> None:
    """specs/twod.md 3 gates on one published fit: CUDA vs catalog < 1e-3 dB, WGSL vs CUDA >= 50 dB, |dPSNR| < 0.01 dB."""
    published_or_skip()
    config = Config(stems=[STEM], roots=Roots(runs=tmp_path))
    if not config.gsplat.is_file():
        pytest.skip(f"needs the release gsplat binary {config.gsplat} (pixi run -e gsplat-rust-renderer gsplat-build)")
    files = published_files(config.roots, STEM, FITS["pub-rich"])
    assert files is not None
    runs: list[ImageRun] = decode_fit(config, ImageScorer(torch.device("cuda")), STEM, files)
    cuda, wgsl = runs
    assert (cuda.decoder, wgsl.decoder) == ("cuda", "wgsl")
    for condition in ("decoded_hxm", "uint16", "raw_fp32"):
        cuda_scores, wgsl_scores = cuda.scores(condition), wgsl.scores(condition)
        if cuda_scores.reference_psnr is not None:
            assert abs(cuda_scores.psnr - cuda_scores.reference_psnr) < 1e-3
        assert wgsl_scores.vs_cuda_psnr is not None and wgsl_scores.vs_cuda_psnr >= 50.0
        assert abs(wgsl_scores.psnr - cuda_scores.psnr) < 0.01
        assert wgsl_scores.fps is not None and len(wgsl_scores.trial_ms) == 5
