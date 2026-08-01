"""Golden gate for the trainer: smoke-scale run must hold its measured PSNR.

Skipped without CUDA or the live catalog. First measured value (2026-07-31,
RTX 5090, segment 40753679, 60 views, 300 steps): 22.2 dB. The floor is that
minus 0.5 dB; improvements are welcome and don't fail.
"""

from pathlib import Path

import pytest
import torch
from conftest import PROBE_CATALOG_URL, catalog_reachable

GOLDEN_PSNR_FLOOR_DB: float = 21.7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="trainer requires CUDA")
@pytest.mark.skipif(not catalog_reachable(), reason="ephemeral catalog server on :51299 not running")
def test_smoke_training_psnr_golden(tmp_path: Path) -> None:
    import rerun as rr
    from simplecv.rerun_log_utils import RerunTyroConfig

    from rerun_gsplat.apis.segment_views import SegmentViewsConfig
    from rerun_gsplat.apis.train import Config, main

    rr_config: RerunTyroConfig = RerunTyroConfig(headless=True, save=tmp_path / "golden.rrd")
    config: Config = Config(
        rr_config=rr_config,
        views=SegmentViewsConfig(catalog_url=PROBE_CATALOG_URL, target_view_count=60),
        max_steps=300,
        log_every=100,
        ply_out=tmp_path / "golden.ply",
    )
    final_psnr: float = main(config)
    rr.disconnect()

    assert final_psnr >= GOLDEN_PSNR_FLOOR_DB, f"PSNR regression: {final_psnr:.2f} dB < {GOLDEN_PSNR_FLOOR_DB} dB floor"
    assert config.ply_out.exists()
