"""Released X-Lens weights on ETH3D ``playground_1l``.

The Phase 1 fork measured EPE 4.558876 px, bad1 88.851027%, and metric-depth
abs-rel 0.772609 on the cropped non-occluded sample. This is a port regression
band, not an upstream benchmark claim.
"""

import hashlib
from pathlib import Path

import pytest
from conftest import requires_cuda, slow_cuda
from huggingface_hub import snapshot_download

from monopriors.apis.rig_depth import ETH3DRigPair, eth3d_rig_metrics, load_eth3d_rig_pair
from monopriors.models.rig_depth import RigDepthPrediction, XLensPredictor
from monopriors.models.rig_depth.xlens import download_xlens_checkpoint

pytestmark = [slow_cuda, requires_cuda]

EXPECTED_CHECKPOINT_SHA256: str = "266a0340b53e5cb996cc613a1b0c5966b5bcaeee1ec7c4431e4fc6e7d1e58a0c"
EXPECTED_EPE_PX: float = 4.558876
EXPECTED_BAD1_PERCENT: float = 88.851027
EXPECTED_ABS_REL: float = 0.772609


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    return download_xlens_checkpoint()


def test_released_checkpoint_hash_and_strict_load(checkpoint: Path) -> None:
    """The gated pinned state dict has the recorded SHA-256 and loads strictly."""
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    assert digest.hexdigest() == EXPECTED_CHECKPOINT_SHA256
    XLensPredictor(device="cuda", checkpoint=checkpoint)


def test_eth3d_playground_accuracy(checkpoint: Path) -> None:
    """The port stays within ±10% of the Phase 1 metric-depth baseline."""
    root: Path = Path(
        snapshot_download(
            "pablovela5620/monoprior-example",
            repo_type="dataset",
            allow_patterns=["stereo/eth3d/two_view_training*/**"],
        )
    )
    pair: ETH3DRigPair = load_eth3d_rig_pair(root / "stereo/eth3d/two_view_training/playground_1l")
    predictor = XLensPredictor(device="cuda", checkpoint=checkpoint)
    prediction: RigDepthPrediction = predictor(pair.images, pair.rays, pair.cam_types, pair.cam_T_ref)
    epe_px, bad1_percent, abs_rel = eth3d_rig_metrics(
        prediction.depth_m[0].detach().cpu().numpy(),
        pair.ground_truth_disparity,
        pair.nonoccluded,
        pair.calibration,
    )
    print(f"X-Lens playground_1l: EPE {epe_px:.6f} px, bad1 {bad1_percent:.6f}%, abs-rel {abs_rel:.6f}")
    assert epe_px == pytest.approx(EXPECTED_EPE_PX, rel=0.10)
    assert bad1_percent == pytest.approx(EXPECTED_BAD1_PERCENT, rel=0.10)
    assert abs_rel == pytest.approx(EXPECTED_ABS_REL, rel=0.10)
