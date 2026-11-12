"""Update graph replay follows changing features, graph topology, and precision."""

import gc
import weakref

import pytest
from jaxtyping import Float32, Int64

torch = pytest.importorskip("torch")
pytest.importorskip("dpvo.mojo_backend._dpvo_mojo", reason="requires the DPVO Mojo extension")

from dpvo.net import Update  # noqa: E402
from torch import Tensor  # noqa: E402

from metrictrail.update_graphs import update_cuda_graphs  # noqa: E402

pytestmark = pytest.mark.integration


def test_update_replay_tracks_inputs_topology_and_precision() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
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
        with update_cuda_graphs():
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
        torch.testing.assert_close(model(hidden, context, correlations, None, source, target, patch), expected_fp32, rtol=2e-5, atol=2e-5)


def test_update_graphs_release_a_model_before_recovery_loads_its_replacement() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    with torch.inference_mode(), update_cuda_graphs():
        model: Update = Update(p=3).cuda().eval()
        owner: weakref.ReferenceType[Update] = weakref.ref(model)
        hidden: Float32[Tensor, "1 32 384"] = torch.randn(1, 32, 384, device="cuda")
        correlations: Float32[Tensor, "1 32 882"] = torch.randn(1, 32, 882, device="cuda")
        patch: Int64[Tensor, "32"] = torch.arange(32, device="cuda") // 4
        source: Int64[Tensor, "32"] = patch // 4
        target: Int64[Tensor, "32"] = torch.arange(32, device="cuda") % 4
        for _ in range(4):
            result, _ = model(hidden, hidden, correlations, None, source, target, patch)
        del model
        gc.collect()
        assert owner() is None, "the graph cache must not retain the released model"
        assert bool(torch.isfinite(result).all()), "returned updates keep their own storage"

