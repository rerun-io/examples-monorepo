"""Artifact identity, cache ownership, and eager fallback for the DROID GRU backend."""

import copy
from pathlib import Path

import pytest
import torch

pytest.importorskip("vipe", reason="requires ViPE")
from vipe.slam.networks.droid_net import ConvGRU

from metrictrail.calibration_gru import GruEngineKey, calibration_gru, weights_digest


def test_gru_engine_identity_covers_weights_raster_precision_and_profile() -> None:
    model = ConvGRU(128, 320).eval()
    digest = weights_digest(model)
    assert weights_digest(copy.deepcopy(model)) == digest
    with torch.no_grad():
        model.convz.weight[0, 0, 0, 0] += 0.01
    assert weights_digest(model) != digest
    first = GruEngineKey(digest, 41, 74)
    assert first != GruEngineKey(digest, 47, 64)
    assert first != GruEngineKey(digest, 41, 74, max_edges=64)
    assert "fp16" in first.stem


def test_gru_context_restores_forward_and_cpu_uses_eager(tmp_path: Path) -> None:
    original = ConvGRU.forward
    model = ConvGRU(128, 320).eval()
    hidden = torch.randn(1, 128, 2, 3)
    context = torch.randn(1, 320, 2, 3)
    with torch.no_grad():
        expected = model(hidden, context)
        with calibration_gru("tensorrt", cache_dir=tmp_path) as stats:
            torch.testing.assert_close(model(hidden, context), expected, rtol=0.0, atol=0.0)
            assert stats.eager_calls == 1
        assert ConvGRU.forward is original
    assert not list(tmp_path.iterdir())


@pytest.mark.integration
@pytest.mark.parametrize("raster", [(41, 74), (47, 64)])
def test_gru_dynamic_profile_ownership_and_model_cache(raster: tuple[int, int]) -> None:
    """The production checkpoint, dynamic batches, independent outputs, and two app contexts."""
    import os

    from vipe.slam.networks.droid_net import DroidNet
    from vipe.utils.model_cache import ModelCache

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("tensorrt", reason="TensorRT is required")
    weights = Path(torch.hub.get_dir()) / "droid_slam/droid.pth"
    if not weights.is_file():
        pytest.skip(f"missing DROID checkpoint: {weights}")
    cache_dir = Path(os.environ.get("METRICTRAIL_TRT_CACHE", "~/.cache/metrictrail/droid-gru")).expanduser()
    models = ModelCache()
    model = models.get("test/gru", lambda: DroidNet().cuda().eval().update.gru)
    height, width = raster
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for count in (1, 48, 8, 128):
            hidden = torch.randn(count, 128, height, width, device="cuda", dtype=torch.float16) * 0.2
            context = torch.randn(count, 320, height, width, device="cuda", dtype=torch.float16) * 0.2
            reference = model(hidden, context)
            with calibration_gru("tensorrt", model_cache=models, cache_dir=cache_dir) as first:
                result = model(hidden, context)
                saved = result.clone()
                model(hidden * 0.5, context * 0.5)
                torch.testing.assert_close(result, saved, rtol=0.0, atol=0.0)
            assert (result - reference).abs().max().item() < 0.02
            assert (result - reference).abs().float().mean().item() < 0.001
            with calibration_gru("tensorrt", model_cache=models, cache_dir=cache_dir) as second:
                torch.testing.assert_close(model(hidden, context), result, rtol=0.0, atol=0.0)
            assert second.export_seconds == second.build_seconds == second.load_seconds == 0.0
            assert first.engine_files == second.engine_files
            print(first)
        hidden = torch.randn(129, 128, 2, 3, device="cuda", dtype=torch.float16)
        context = torch.randn(129, 320, 2, 3, device="cuda", dtype=torch.float16)
        reference = model(hidden, context)
        with calibration_gru("tensorrt", cache_dir=cache_dir) as fallback:
            torch.testing.assert_close(model(hidden, context), reference, rtol=0.0, atol=0.0)
        assert fallback.eager_calls == 1 and fallback.engine_calls == 0
