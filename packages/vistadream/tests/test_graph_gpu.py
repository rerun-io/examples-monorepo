"""The outpaint + coarse graph on the GPU with MoGe and gsplat, FLUX replaced by OpenCV inpainting (integration: GPU and MoGe weights).

FLUX.1-Fill-dev (~34 GB) is not needed: this checks the plumbing of the stages the short verify case does not run. A flat fill
would not do: MoGe marks it invalid, no Gaussian spawns there and every candidate view keeps more than 25 % holes.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from dagkit.runners import run_threaded
from jaxtyping import UInt8
from numpy import ndarray
from vistadream_fakes import Keep

from vistadream.apis.single_image import PORTS
from vistadream.graphs.single_image import single_image_graph
from vistadream.nodes.prepare import DepthNode, ImageSource, OutpaintNode
from vistadream.nodes.splats import ExpandNode, InitNode, RenderNode

pytestmark = pytest.mark.integration
IMAGE: Path = Path(__file__).resolve().parents[1] / "data/office/IMG_4029.jpg"


def opencv_inpaint(rgb: UInt8[ndarray, "h w 3"], mask: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "h w 3"]:
    return cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)


def test_coarse_graph_grows_the_scene_on_the_gpu() -> None:
    pytest.importorskip("gsplat")
    if not torch.cuda.is_available() or not IMAGE.exists():
        pytest.skip("needs a CUDA GPU and data/office/IMG_4029.jpg")
    from monopriors.models.relative_depth.moge_v1 import MoGeV1Predictor

    moge = MoGeV1Predictor(device="cuda")
    keep = Keep(PORTS)
    nodes = (OutpaintNode(opencv_inpaint), DepthNode(moge), InitNode(), ExpandNode(opencv_inpaint, moge, n_frames=4, iters=50), RenderNode())
    report = run_threaded(single_image_graph(ImageSource(IMAGE, 512), *nodes, {"keep": keep}), mode="lossless")
    assert report.status == "complete"
    [views] = [v for port, _, v in keep.items if port == "views"]
    [scene] = [v for port, _, v in keep.items if port == "scene"]
    renders = [v for port, _, v in keep.items if port == "view"]
    assert isinstance(views, dict) and isinstance(scene, dict) and views["input_index"] == 1  # the outpainted canvas comes first
    n_views: int = len(views["views"])
    assert n_views in (3, 4) and views["views"][2]["supervise"].any()  # 1-2 inpainted views (n_frames 4)
    assert views["views"][0]["rgb"].shape[:2] == views["views"][0]["depth"].shape != views["views"][1]["depth"].shape
    assert len(renders) == 25 * n_views and all(np.isfinite(scene[k]).all() for k in ("centers", "scales", "rgb", "opacity"))
