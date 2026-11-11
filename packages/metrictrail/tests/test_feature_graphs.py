"""Feature graphs preserve changing images, retained outputs, rasters, and recovery precision."""

import gc
import weakref
from pathlib import Path

import pytest
from jaxtyping import Float, Float32

torch = pytest.importorskip("torch")
pytest.importorskip("vipe.metrictrail.dino_fusion", reason="requires the ViPE package")

from dpvo.extractor import BasicEncoder4  # noqa: E402
from torch import Tensor  # noqa: E402
from vipe.metrictrail.dino_fusion import FrozenDinoV3  # noqa: E402

from metrictrail.feature_graphs import feature_cuda_graphs  # noqa: E402

pytestmark = pytest.mark.integration


def test_graph_replay_preserves_features_and_precision_changes() -> None:
    weights: Path = Path("/mnt/nas/ssd-8tb/metrictrail/weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    if not weights.is_file():
        pytest.skip(f"missing DINO weights: {weights}")
    model: FrozenDinoV3 = FrozenDinoV3(weights=weights).cuda().eval()
    with torch.inference_mode():
        first: Float32[Tensor, "1 1 3 280 504"] = torch.rand(1, 1, 3, 280, 504, device="cuda")
        second: Float32[Tensor, "1 1 3 280 504"] = first * 0.3
        expected_first: Float32[Tensor, "1 1 384 18 32"] = model(first)
        expected_second: Float32[Tensor, "1 1 384 18 32"] = model(second)
        model.amp_dtype = torch.float32
        expected_fp32: Float32[Tensor, "1 1 384 18 32"] = model(second)
        wider: Float32[Tensor, "1 1 3 388 520"] = torch.rand(1, 1, 3, 388, 520, device="cuda")
        expected_wider: Float32[Tensor, "1 1 384 25 33"] = model(wider)
        model.amp_dtype = torch.bfloat16
        with feature_cuda_graphs():
            actual_first: Float32[Tensor, "1 1 384 18 32"] = model(first)
            actual_second: Float32[Tensor, "1 1 384 18 32"] = model(second)
            torch.testing.assert_close(actual_first, expected_first, rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual_second, expected_second, rtol=0.0, atol=0.0)
            model.amp_dtype = torch.float32
            torch.testing.assert_close(model(second), expected_fp32, rtol=0.0, atol=0.0)
            torch.testing.assert_close(model(wider), expected_wider, rtol=0.0, atol=0.0)
        # The context must leave ordinary inference usable, including another raster.
        torch.testing.assert_close(model(wider), expected_wider, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("norm", ["instance", "none"])
def test_encoder_replay_preserves_features_across_precision_changes(norm: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    model: BasicEncoder4 = BasicEncoder4(output_dim=128, norm_fn=norm).cuda().eval()
    with torch.inference_mode():
        first: Float32[Tensor, "1 1 3 280 504"] = torch.rand(1, 1, 3, 280, 504, device="cuda")
        second: Float32[Tensor, "1 1 3 280 504"] = first * 0.3
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            expected_first: Float[Tensor, "1 1 128 70 126"] = model(first)
            expected_second: Float[Tensor, "1 1 128 70 126"] = model(second)
        expected_fp32: Float32[Tensor, "1 1 128 70 126"] = model(second)
        with feature_cuda_graphs():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                actual_first: Float[Tensor, "1 1 128 70 126"] = model(first)
                actual_second: Float[Tensor, "1 1 128 70 126"] = model(second)
                torch.testing.assert_close(actual_first, expected_first, rtol=0.0, atol=0.0)
                torch.testing.assert_close(actual_second, expected_second, rtol=0.0, atol=0.0)
            torch.testing.assert_close(model(second), expected_fp32, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model(second), expected_fp32, rtol=0.0, atol=0.0)


def test_feature_graphs_release_a_model_before_recovery_loads_its_replacement() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    with torch.inference_mode(), feature_cuda_graphs():
        model: BasicEncoder4 = BasicEncoder4(output_dim=128, norm_fn="none").cuda().eval()
        owner: weakref.ReferenceType[BasicEncoder4] = weakref.ref(model)
        image: Float32[Tensor, "1 1 3 32 48"] = torch.rand(1, 1, 3, 32, 48, device="cuda")
        features: Float32[Tensor, "1 1 128 8 12"] = model(image)
        del model
        gc.collect()
        assert owner() is None, "the graph cache must not retain the released model"
        assert bool(torch.isfinite(features).all()), "returned features keep their own storage"
