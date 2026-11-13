"""DINOv3 graph replay preserves changing images, rasters, and recovery precision (DPVO's own: dpvo/tests)."""

from pathlib import Path

import pytest
from jaxtyping import Float32

torch = pytest.importorskip("torch")
pytest.importorskip("vipe.metrictrail.dino_fusion", reason="requires the ViPE package")

from dpvo.graph_replay import graph_replay  # noqa: E402
from torch import Tensor  # noqa: E402
from vipe.metrictrail.dino_fusion import FrozenDinoV3  # noqa: E402

from metrictrail.feature_graphs import dino_cuda_graphs  # noqa: E402

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
        with graph_replay(), dino_cuda_graphs():
            actual_first: Float32[Tensor, "1 1 384 18 32"] = model(first)
            actual_second: Float32[Tensor, "1 1 384 18 32"] = model(second)
            torch.testing.assert_close(actual_first, expected_first, rtol=0.0, atol=0.0)
            torch.testing.assert_close(actual_second, expected_second, rtol=0.0, atol=0.0)
            model.amp_dtype = torch.float32
            torch.testing.assert_close(model(second), expected_fp32, rtol=0.0, atol=0.0)
            torch.testing.assert_close(model(wider), expected_wider, rtol=0.0, atol=0.0)
        # The context must leave ordinary inference usable, including another raster.
        torch.testing.assert_close(model(wider), expected_wider, rtol=0.0, atol=0.0)
