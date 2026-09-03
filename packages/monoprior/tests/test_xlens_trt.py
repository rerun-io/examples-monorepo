"""Released X-Lens weights through the TensorRT rig predictor on ETH3D ``playground_1l``.

Parity is measured against the eager fp32 frozen-geometry path; the accuracy
band is the same port-regression band as ``test_xlens_eth3d.py``.
"""

from pathlib import Path

import pytest
import torch
from conftest import requires_cuda, slow_cuda
from huggingface_hub import snapshot_download
from jaxtyping import Bool, Float32
from torch import Tensor

from monopriors.apis.rig_depth import ETH3DRigPair, eth3d_rig_metrics, load_eth3d_rig_pair
from monopriors.models.rig_depth import RigDepthPrediction, XLensPredictor, XLensTrtPredictor
from monopriors.models.rig_depth.xlens import download_xlens_checkpoint

pytestmark = [slow_cuda, requires_cuda]

EXPECTED_EPE_PX: float = 4.558876
EXPECTED_BAD1_PERCENT: float = 88.851027
EXPECTED_ABS_REL: float = 0.772609


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    return download_xlens_checkpoint()


@pytest.fixture(scope="module")
def pair() -> ETH3DRigPair:
    root: Path = Path(
        snapshot_download(
            "pablovela5620/monoprior-example",
            repo_type="dataset",
            allow_patterns=["stereo/eth3d/two_view_training*/**"],
        )
    )
    return load_eth3d_rig_pair(root / "stereo/eth3d/two_view_training/playground_1l")


def test_tensorrt_matches_eager_fp32_and_keeps_accuracy(checkpoint: Path, pair: ETH3DRigPair) -> None:
    """fp16 engine depth stays within 2% median abs-rel of eager fp32 and inside the ETH3D regression band."""
    eager = XLensPredictor(device="cuda", checkpoint=checkpoint, amp="fp32")
    reference: RigDepthPrediction = eager(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    del eager
    torch.cuda.empty_cache()

    predictor = XLensTrtPredictor(checkpoint=checkpoint, use_cuda_graph=True)
    prediction: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    assert predictor.engine_path is not None and predictor.engine_path.exists()
    replay: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)

    valid: Bool[Tensor, "s h w"] = (reference.depth_m > 0.0) & torch.isfinite(reference.depth_m)
    abs_rel: Float32[Tensor, "n"] = ((prediction.depth_m - reference.depth_m).abs() / reference.depth_m)[valid]
    median_abs_rel: float = float(abs_rel.median())
    scale_rel_diff: float = abs(prediction.scale / reference.scale - 1.0)
    print(f"X-Lens TensorRT vs eager fp32: median abs-rel {median_abs_rel:.5f}, scale rel diff {scale_rel_diff:.5f}")
    assert median_abs_rel < 0.02
    assert scale_rel_diff < 0.02
    assert torch.allclose(prediction.depth_m, replay.depth_m, rtol=1e-3, atol=1e-3)
    assert prediction.depth_m.shape == reference.depth_m.shape and prediction.mask.shape == reference.mask.shape

    epe_px, bad1_percent, abs_rel_metric = eth3d_rig_metrics(
        prediction.depth_m[0].detach().cpu().numpy(),
        pair.ground_truth_disparity,
        pair.nonoccluded,
        pair.calibration,
    )
    print(f"X-Lens TensorRT playground_1l: EPE {epe_px:.6f} px, bad1 {bad1_percent:.6f}%, abs-rel {abs_rel_metric:.6f}")
    assert epe_px == pytest.approx(EXPECTED_EPE_PX, rel=0.10)
    assert bad1_percent == pytest.approx(EXPECTED_BAD1_PERCENT, rel=0.10)
    assert abs_rel_metric == pytest.approx(EXPECTED_ABS_REL, rel=0.10)
