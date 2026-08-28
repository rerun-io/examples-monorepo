"""infer_rerun must produce a recording whose 2D and 3D views actually render (pixel evidence, not logs)."""

import sys
import time
from pathlib import Path

import cv2
import pytest
import rerun as rr
import torch
from rerun.experimental import ViewerClient
from simplecv.rerun_log_utils import RerunTyroConfig

from zipdepth.apis.infer_rerun import InferRerunConfig, infer_rerun

PKG = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and the Hub weights")
def test_saved_recording_renders(tmp_path: Path) -> None:
    rrd = tmp_path / "zipdepth.rrd"
    infer_rerun(InferRerunConfig(rr_config=RerunTyroConfig(headless=True, save=rrd), image=PKG / "assets/examples/im0.jpg"))
    rr.disconnect()
    assert rrd.stat().st_size > 100_000

    rerun_bin = Path(sys.executable).parent / "rerun"  # the env's own viewer, never a stale PATH install
    shot = tmp_path / "shot.png"
    with ViewerClient.spawn(headless=True, port=9899, hide_welcome_screen=True, executable_path=str(rerun_bin)) as viewer:
        rr.init("zipdepth_pixel_check", default_enabled=True, strict=True)
        rr.connect_grpc(url=viewer.url)
        rr.log_file_from_path(rrd)
        recording = rr.get_global_data_recording()
        assert recording is not None
        recording.flush(timeout_sec=30.0)
        time.sleep(3.0)
        viewer.save_screenshot(str(shot))
    img = cv2.imread(str(shot))
    assert img is not None
    h, w, _ = img.shape
    left, right = img[h // 4 : 3 * h // 4, : w // 2], img[: h // 2, 3 * w // 4 :]
    # both the 3D cloud (left) and the RGB view (right) must be non-uniform, i.e. rendered
    assert left.std() > 10 and right.std() > 10
