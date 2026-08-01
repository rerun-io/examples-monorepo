"""PPISP contract the trainer relies on: identity at init, zero-cost regularizer.

The trainer feeds the raw render through PPISP before the photometric loss and
evaluates holdout frames with ``frame_idx=-1``; both paths must start as an
identity so PPISP-on runs are comparable to PPISP-off runs at step 0.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ppisp is a CUDA extension")


def test_ppisp_identity_at_init() -> None:
    from jaxtyping import Float32
    from ppisp import PPISP, PPISPConfig
    from torch import Tensor

    module: PPISP = PPISP(num_cameras=1, num_frames=4, config=PPISPConfig(use_controller=False))
    optimizers: list[torch.optim.Optimizer] = module.create_optimizers()
    module.create_schedulers(optimizers, 100)
    rgb_hwc: Float32[Tensor, "h w 3"] = torch.rand(17, 23, 3, device="cuda")
    for frame_idx in (0, -1):
        out_hwc: Float32[Tensor, "h w 3"] = module(rgb_hwc.contiguous(), camera_idx=0, frame_idx=frame_idx)
        assert torch.allclose(out_hwc, rgb_hwc, atol=1e-4), f"frame_idx={frame_idx} is not an identity at init"


def test_ppisp_regularization_zero_at_init() -> None:
    from ppisp import PPISP, PPISPConfig

    module: PPISP = PPISP(num_cameras=1, num_frames=4, config=PPISPConfig(use_controller=False))
    assert float(module.get_regularization_loss()) == pytest.approx(0.0, abs=1e-8)
