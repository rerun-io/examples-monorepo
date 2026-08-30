"""Released Fast-FoundationStereo weights and the ETH3D ``playground_1l`` reference band."""

from pathlib import Path

import cv2
import numpy as np
from conftest import requires_cuda, slow_cuda
from huggingface_hub import snapshot_download
from jaxtyping import Float32, UInt8
from torch import nn

from monopriors.apis.stereo_depth import ETH3D_MAX_DISP, MiddleburyCalibration, read_middlebury_calib, read_pfm, read_rgb, stereo_metrics
from monopriors.models.stereo_depth import FastFoundationStereoPredictor, StereoDepthPrediction
from monopriors.models.stereo_depth.fast_foundationstereo import download_fast_foundationstereo_checkpoint, load_fast_foundationstereo

pytestmark = slow_cuda


def test_released_checkpoint_remaps_serialized_architecture() -> None:
    """The pickled upstream architecture remaps with the released tensor and parameter counts."""
    checkpoint: Path = download_fast_foundationstereo_checkpoint()
    model: nn.Module = load_fast_foundationstereo(checkpoint)
    assert len(model.state_dict()) == 722
    assert sum(parameter.numel() for parameter in model.parameters()) == 17_654_857
    assert model.args.normalize is True
    assert model.args.valid_iters == 8 and model.args.max_disp == 416


@requires_cuda
def test_eth3d_playground_accuracy() -> None:
    """At the shared 192 px cutoff, the release stays near EPE 0.241 px and bad1 0.48%."""
    root: Path = Path(snapshot_download("pablovela5620/monoprior-example", repo_type="dataset", allow_patterns=["stereo/eth3d/two_view_training*/**"]))
    scene: Path = root / "stereo/eth3d/two_view_training/playground_1l"
    gt_dir: Path = root / "stereo/eth3d/two_view_training_gt/playground_1l"
    calibration: MiddleburyCalibration = read_middlebury_calib(scene / "calib.txt")
    predictor: FastFoundationStereoPredictor = FastFoundationStereoPredictor(device="cuda")
    prediction: StereoDepthPrediction = predictor(read_rgb(scene / "im0.png"), read_rgb(scene / "im1.png"), K_33=calibration.K_33, baseline_m=calibration.baseline_m)

    gt_hw: Float32[np.ndarray, "h w"] = read_pfm(gt_dir / "disp0GT.pfm")
    nocc_hw: UInt8[np.ndarray, "h w"] = cv2.imread(str(gt_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    metrics: tuple[float, float] = stereo_metrics(prediction.disparity, gt_hw, nocc_hw, max_disp=ETH3D_MAX_DISP)
    epe_px: float = metrics[0]
    bad1_percent: float = metrics[1]
    print(f"Fast-FoundationStereo playground_1l: EPE {epe_px:.3f} px, bad1 {bad1_percent:.2f}%")
    assert epe_px < 0.30
    assert bad1_percent < 0.75
