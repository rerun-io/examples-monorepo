"""CUDA-graph replay of DPVO's encoders and recurrent update: one bounded cache, outputs that follow every input."""

import gc
import weakref

import pytest
import torch
from jaxtyping import Float, Float32, Int64
from torch import Tensor, nn

from dpvo.graph_replay import GraphCache, active, graph_replay


def test_one_cache_bounds_every_model_and_admits_a_shape_on_its_nth_call() -> None:
    cache = GraphCache(max_graphs=2)
    first, second = nn.Linear(1, 1), nn.Linear(1, 1)
    assert not cache.admit(first, "update", min_calls=3) and not cache.admit(first, "update", min_calls=3)  # eager twice
    assert cache.admit(first, "update", min_calls=3)
    cache.put(first, "update", "graph a")
    assert cache.get(first, "update") == "graph a" and cache.get(second, "update") is None
    assert cache.admit(second, "features")
    cache.put(second, "features", "graph b")
    assert not cache.admit(second, "other shape")  # the bound covers both models
    assert cache.count() == 2


def test_the_context_is_the_only_switch() -> None:
    assert active() is None
    with graph_replay(max_graphs=4) as cache:
        assert active() is cache and cache.max_graphs == 4
    assert active() is None


pytestmark_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.mark.integration
@pytestmark_cuda
@pytest.mark.parametrize("norm", ["instance", "none"])
def test_encoder_replay_preserves_features_across_precision_changes(norm: str) -> None:
    from dpvo.extractor import BasicEncoder4

    model: BasicEncoder4 = BasicEncoder4(output_dim=128, norm_fn=norm).cuda().eval()
    with torch.inference_mode():
        first: Float32[Tensor, "1 1 3 280 504"] = torch.rand(1, 1, 3, 280, 504, device="cuda")
        second: Float32[Tensor, "1 1 3 280 504"] = first * 0.3
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            expected_first: Float[Tensor, "1 1 128 70 126"] = model(first)
            expected_second: Float[Tensor, "1 1 128 70 126"] = model(second)
        expected_fp32: Float32[Tensor, "1 1 128 70 126"] = model(second)
        with graph_replay() as cache:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                actual_first: Float[Tensor, "1 1 128 70 126"] = model(first)
                actual_second: Float[Tensor, "1 1 128 70 126"] = model(second)
                torch.testing.assert_close(actual_first, expected_first, rtol=0.0, atol=0.0)
                torch.testing.assert_close(actual_second, expected_second, rtol=0.0, atol=0.0)
            torch.testing.assert_close(model(second), expected_fp32, rtol=0.0, atol=0.0)
            assert cache.count() == 2  # one graph per precision
        torch.testing.assert_close(model(second), expected_fp32, rtol=0.0, atol=0.0)


@pytest.mark.integration
@pytestmark_cuda
def test_update_replay_tracks_inputs_topology_and_precision() -> None:
    pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="requires the DPVO Mojo extension")
    from dpvo.net import Update

    torch.manual_seed(18)
    model: Update = Update(p=3).cuda().eval()
    hidden: Float32[Tensor, "1 96 384"] = torch.randn(1, 96, 384, device="cuda")
    context: Float32[Tensor, "1 96 384"] = torch.randn_like(hidden)
    correlations: Float32[Tensor, "1 96 882"] = torch.randn(1, 96, 882, device="cuda")
    patch: Int64[Tensor, "96"] = torch.arange(96, device="cuda") // 4
    source: Int64[Tensor, "96"] = patch // 4
    target: Int64[Tensor, "96"] = torch.arange(96, device="cuda") % 4
    changed_patch: Int64[Tensor, "96"] = patch.roll(7)
    changed_source: Int64[Tensor, "96"] = changed_patch // 4
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            expected = model(hidden, context, correlations, None, source, target, patch)
            changed = model(hidden * 0.3, context, correlations, None, changed_source, target, changed_patch)
        expected_fp32 = model(hidden, context, correlations, None, source, target, patch)
        expected_short = model(hidden[:, :64], context[:, :64], correlations[:, :64], None, source[:64], target[:64], patch[:64])
        with graph_replay() as cache:
            with torch.autocast("cuda", dtype=torch.float16):
                for _ in range(4):
                    actual = model(hidden, context, correlations, None, source, target, patch)
                later = model(hidden * 0.3, context, correlations, None, changed_source, target, changed_patch)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                torch.testing.assert_close(later, changed, rtol=2e-4, atol=2e-4)
            for _ in range(4):
                actual_fp32 = model(hidden, context, correlations, None, source, target, patch)
            torch.testing.assert_close(actual_fp32, expected_fp32, rtol=2e-5, atol=2e-5)
            for _ in range(4):
                actual_short = model(hidden[:, :64], context[:, :64], correlations[:, :64], None, source[:64], target[:64], patch[:64])
            torch.testing.assert_close(actual_short, expected_short, rtol=2e-5, atol=2e-5)
            assert cache.count() == 3  # fp16, fp32 and the shorter edge count
        torch.testing.assert_close(model(hidden, context, correlations, None, source, target, patch), expected_fp32, rtol=2e-5, atol=2e-5)


@pytest.mark.integration
@pytestmark_cuda
def test_replay_releases_a_model_before_recovery_loads_its_replacement() -> None:
    pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="requires the DPVO Mojo extension")
    from dpvo.extractor import BasicEncoder4
    from dpvo.net import Update

    with torch.inference_mode(), graph_replay():
        encoder: BasicEncoder4 = BasicEncoder4(output_dim=128, norm_fn="none").cuda().eval()
        update: Update = Update(p=3).cuda().eval()
        owners = [weakref.ref(encoder), weakref.ref(update)]
        features: Float32[Tensor, "1 1 128 8 12"] = encoder(torch.rand(1, 1, 3, 32, 48, device="cuda"))
        hidden: Float32[Tensor, "1 32 384"] = torch.randn(1, 32, 384, device="cuda")
        correlations: Float32[Tensor, "1 32 882"] = torch.randn(1, 32, 882, device="cuda")
        patch: Int64[Tensor, "32"] = torch.arange(32, device="cuda") // 4
        for _ in range(4):
            result, _ = update(hidden, hidden, correlations, None, patch // 4, torch.arange(32, device="cuda") % 4, patch)
        del encoder, update
        gc.collect()
        assert all(owner() is None for owner in owners), "the graph cache must not retain a released model"
        assert bool(torch.isfinite(features).all()) and bool(torch.isfinite(result).all()), "returned tensors keep their own storage"
