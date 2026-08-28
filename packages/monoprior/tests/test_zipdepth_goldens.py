"""The annotated third_party/zipdepth network must stay numerically identical to upstream.

Goldens were produced on CPU by the upstream ``architecture.py`` (fork 5a80354, byte-identical to
fabiotosi92/ZipDepth) with the released ``zipdepth_base.pth``: an un-fused train-mode forward on a
seeded input, and the fused inference path through ``ZipDepthPredictor`` on ``im0.jpg``.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from monopriors.models.relative_depth.zipdepth import ZipDepthPredictor, download_zipdepth_checkpoint
from monopriors.third_party.zipdepth.architecture import create_model
from monopriors.third_party.zipdepth.model_utils import strip_state_dict_prefixes

REF = Path(__file__).parent / "reference_data" / "zipdepth"


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    try:
        return download_zipdepth_checkpoint()
    except Exception as e:  # noqa: BLE001 — offline hosts skip, they don't fail
        pytest.skip(f"released weights unavailable: {e}")


def test_train_mode_forward_matches_upstream(checkpoint: Path) -> None:
    model = create_model(variant="base")
    model.load_state_dict(strip_state_dict_prefixes(torch.load(checkpoint, map_location="cpu", weights_only=True)), strict=True)
    model.train()
    x = torch.rand(2, 3, 96, 128, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model(x).numpy()
    np.testing.assert_allclose(out, np.load(REF / "train_forward_2x3x96x128.npy"), rtol=0.0, atol=1e-5)


def test_fused_inference_matches_upstream(checkpoint: Path) -> None:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(str(REF / "im0.jpg")), (256, 192)), cv2.COLOR_BGR2RGB)
    disparity = ZipDepthPredictor(device="cpu", checkpoint=checkpoint, input_size=128)(rgb, None).disparity
    np.testing.assert_allclose(disparity, np.load(REF / "im0_192x256_disparity.npy"), rtol=0.0, atol=1e-5)
