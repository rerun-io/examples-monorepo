"""Released X-Lens weights through the TensorRT rig predictor on ETH3D ``playground_1l``.

Parity is measured against the eager fp32 frozen-geometry path; the accuracy
band is the same port-regression band as ``test_xlens_eth3d.py``.
"""

from pathlib import Path

import numpy as np
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


@pytest.fixture(scope="module")
def reference(checkpoint: Path, pair: ETH3DRigPair) -> RigDepthPrediction:
    """Eager fp32 frozen-geometry prediction, the parity reference for every engine."""
    eager = XLensPredictor(device="cuda", checkpoint=checkpoint, amp="fp32")
    prediction: RigDepthPrediction = eager(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    del eager
    torch.cuda.empty_cache()
    return prediction


def _assert_parity(prediction: RigDepthPrediction, reference: RigDepthPrediction, label: str) -> None:
    valid: Bool[Tensor, "s h w"] = (reference.depth_m > 0.0) & torch.isfinite(reference.depth_m)
    abs_rel: Float32[Tensor, "n"] = ((prediction.depth_m - reference.depth_m).abs() / reference.depth_m)[valid]
    median_abs_rel: float = float(abs_rel.median())
    scale_rel_diff: float = abs(prediction.scale / reference.scale - 1.0)
    print(f"{label} vs eager fp32: median abs-rel {median_abs_rel:.5f}, scale rel diff {scale_rel_diff:.5f}")
    assert median_abs_rel < 0.02
    assert scale_rel_diff < 0.02
    assert prediction.depth_m.shape == reference.depth_m.shape and prediction.mask.shape == reference.mask.shape


def test_dynamic_profile_engine_serves_batches_and_other_shapes(checkpoint: Path, pair: ETH3DRigPair, reference: RigDepthPrediction) -> None:
    """The default profile, one dynamic engine (views 2-4, up to the ETH3D crop, batch 2), matches eager fp32 at the rig shape and at a smaller off-opt shape."""
    height: int = pair.images.shape[1]
    width: int = pair.images.shape[2]
    predictor = XLensTrtPredictor(
        checkpoint=checkpoint,
        use_cuda_graph=True,
        dynamic_views=(2, 4),
        dynamic_height=(280, height),
        dynamic_width=(336, width),
        max_batch_size=2,
    )
    prediction: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    assert predictor.engine_path is not None and "dyn_v2-4" in predictor.engine_path.name
    _assert_parity(prediction, reference, "X-Lens TensorRT dynamic profile")
    print(predictor.runtime_summary())

    batch: list[RigDepthPrediction] = predictor.predict_batch(np.stack([pair.images, pair.images[:, ::-1].copy()]), pair.rays, pair.cam_types, pair.cam_T_ref)
    assert len(batch) == 2
    assert torch.allclose(batch[0].depth_m, prediction.depth_m, rtol=1e-2, atol=1e-2)
    assert not torch.allclose(batch[1].depth_m, prediction.depth_m, rtol=1e-2, atol=1e-2), "the flipped frameset must give a different answer"

    # Same engine, smaller rig: a centre crop to 392x672 keeps the pinhole geometry consistent.
    crop_h: int = 392
    crop_w: int = 672
    y0: int = (height - crop_h) // 2
    x0: int = (width - crop_w) // 2
    images_small = np.ascontiguousarray(pair.images[:, y0 : y0 + crop_h, x0 : x0 + crop_w])
    rays_small = np.ascontiguousarray(pair.rays[:, y0 : y0 + crop_h, x0 : x0 + crop_w])
    eager = XLensPredictor(device="cuda", checkpoint=checkpoint, amp="fp32")
    reference_small: RigDepthPrediction = eager(images_small, rays_small, pair.cam_types, pair.cam_T_ref)
    del eager
    small: RigDepthPrediction = predictor(images_small, rays_small, pair.cam_types, pair.cam_T_ref)
    assert predictor.engine_path is not None and "dyn_v2-4" in predictor.engine_path.name, "the dynamic engine must be reused, not rebuilt"
    _assert_parity(small, reference_small, "X-Lens TensorRT dynamic profile at 392x672")
    with pytest.raises(ValueError, match="outside the dynamic profile"):
        predictor(pair.images[:, :, :280], pair.rays[:, :, :280], pair.cam_types, pair.cam_T_ref)


def test_tensorrt_matches_eager_fp32_and_keeps_accuracy(checkpoint: Path, pair: ETH3DRigPair, reference: RigDepthPrediction) -> None:
    """The rig-profile engine (dynamic batch up to 4) stays within 2% median abs-rel of eager fp32 and inside the ETH3D regression band."""
    predictor = XLensTrtPredictor(checkpoint=checkpoint, use_cuda_graph=True, profile="rig", max_batch_size=4)
    prediction: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    assert predictor.engine_path is not None and predictor.engine_path.exists() and "_b4_" in predictor.engine_path.name
    replay: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    print(predictor.runtime_summary())
    batch: list[RigDepthPrediction] = predictor.predict_batch(np.stack([pair.images] * 5), pair.rays, pair.cam_types, pair.cam_T_ref)
    assert len(batch) == 5 and all(torch.allclose(item.depth_m, prediction.depth_m, rtol=1e-2, atol=1e-2) for item in batch)

    _assert_parity(prediction, reference, "X-Lens TensorRT rig profile")
    assert torch.allclose(prediction.depth_m, replay.depth_m, rtol=1e-3, atol=1e-3)

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
