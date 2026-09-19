"""One-off full catalog-cache parity gate for the direct trainer."""

import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("rerun.catalog", reason="requires the rerun.catalog dependency in this environment")
pytest.importorskip("rerun.experimental.dataloader", reason="requires the rerun.experimental.dataloader dependency in this environment")
pytest.importorskip("torchcodec", reason="requires the torchcodec dependency in this environment")

from gauss_surf.catalog import SegmentReader
from gauss_surf.train_gsplat.cache import compare_cache_to_bundle, load_gpu_cache


@pytest.mark.skipif(os.environ.get("GAUSS_SURF_RUN_CACHE_PARITY") != "1", reason="requires the live catalog and a large CUDA cache")
@pytest.mark.golden
def test_catalog_gpu_cache_is_pixel_identical_to_part10_bundle() -> None:
    """Compare every RGB, depth, and normal value for both camera streams."""
    root: Path = Path(__file__).parents[1]
    bundle_dir: Path = root / "data/training_bundle_part10/47115416"
    if not (bundle_dir / "transforms.json").is_file():
        pytest.skip(f"local Gauss Surf artifact is unavailable: {bundle_dir / 'transforms.json'}")
    reader: SegmentReader = SegmentReader.open("rerun+http://127.0.0.1:51236", "part8-band", "47115416")

    cache = load_gpu_cache(reader, bundle_dir, torch.device("cuda"))
    parity = compare_cache_to_bundle(cache, bundle_dir)

    assert len(cache.cameras) == 1_302
    assert len(cache.holdout_indices) == 81
    assert len(cache.train_indices) == 1_221
    assert parity.mismatch_count == 0, parity.mismatched_values
    print({"compared_values": parity.compared_values, "mismatched_values": parity.mismatched_values})
