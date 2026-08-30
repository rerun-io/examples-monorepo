"""Released LAS2 weights on the ETH3D ``playground_1l`` sample must land near the numbers recorded on the fork
(M: EPE 0.350 / bad1 2.24 %, H: 0.250 / 1.12 %; paper dataset means 2.59 / 1.83). Downloads weights + data."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import requires_cuda, slow_cuda
from huggingface_hub import snapshot_download
from jaxtyping import Float32, UInt8

from monopriors.apis.stereo_depth import ETH3D_MAX_DISP, read_middlebury_calib, read_rgb, stereo_metrics
from monopriors.models.stereo_depth import LiteAnyStereoPredictor

pytestmark = [slow_cuda, requires_cuda]


def _read_pfm(path: Path) -> Float32[np.ndarray, "h w"]:
    with path.open("rb") as f:
        assert f.readline().strip() == b"Pf"
        width, height = (int(v) for v in f.readline().split())
        scale = float(f.readline())
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4", count=width * height)
    return np.flipud(data.reshape(height, width)).astype(np.float32)


@pytest.mark.parametrize(("model_size", "max_bad1_percent"), [("m", 3.5), ("h", 2.0)])
def test_eth3d_playground_bad1(model_size: str, max_bad1_percent: float) -> None:
    root = Path(snapshot_download("pablovela5620/monoprior-example", repo_type="dataset", allow_patterns=["stereo/eth3d/two_view_training*/**"]))
    scene = root / "stereo/eth3d/two_view_training/playground_1l"
    gt_dir = root / "stereo/eth3d/two_view_training_gt/playground_1l"
    calibration = read_middlebury_calib(scene / "calib.txt")
    predictor = LiteAnyStereoPredictor(device="cuda", model_size=model_size)
    pred = predictor(read_rgb(scene / "im0.png"), read_rgb(scene / "im1.png"), K_33=calibration.K_33, baseline_m=calibration.baseline_m)

    gt_hw: Float32[np.ndarray, "h w"] = _read_pfm(gt_dir / "disp0GT.pfm")
    nocc_hw: UInt8[np.ndarray, "h w"] = cv2.imread(str(gt_dir / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    epe_px, bad1_percent = stereo_metrics(pred.disparity, gt_hw, nocc_hw, max_disp=ETH3D_MAX_DISP)
    print(f"LAS2-{model_size.upper()} playground_1l: EPE {epe_px:.3f} px, bad1 {bad1_percent:.2f}%")
    assert bad1_percent < max_bad1_percent, f"LAS2-{model_size.upper()} bad1 {bad1_percent:.2f}%"
    assert np.nanmedian(pred.depth_meters[pred.depth_meters > 0]) < 50.0
