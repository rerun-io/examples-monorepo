"""Integration: the published HF checkpoints and datasets (gsplat-modern-data, READY markers)."""

import json
from pathlib import Path

import numpy as np
import pytest

from hexels_bench.apis.export_published import stage_published
from hexels_bench.cameras import load_cameras_file
from hexels_bench.hexel_ply import HexelPly, HexelVariant, read_hexel_ply
from hexels_bench.layout import Dataset, Roots
from hexels_bench.metrics import MetricsFile, load_metrics_file

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(("dataset", "scene", "variant", "marker"), [("dtu", "scan24", "v3", "dtu"), ("mipnerf360", "garden", "v1", "mipnerf360")])
def test_stage_published_writes_ply_cameras_and_exact_gt(tmp_path: Path, dataset: Dataset, scene: str, variant: HexelVariant, marker: str) -> None:
    data: Path = Roots().data
    for needed in ("pretrained-nvs", marker):
        if not (data / "READY" / needed).is_file():
            pytest.skip(f"needs {data}/READY/{needed}")
    roots = Roots(data=data, runs=tmp_path)
    run_dir: Path = stage_published(roots, dataset, scene, variant)  # raises if the GT digest differs
    metadata: dict = json.loads((roots.published_model(dataset, scene, variant) / "metadata.json").read_text())
    ply: HexelPly = read_hexel_ply(run_dir / "model.ply")
    assert ply.cloud.variant == variant
    assert ply.cloud.count == metadata["primitive_count"]
    assert np.isfinite(ply.cloud.to_rows()).all()
    cameras = load_cameras_file(run_dir / "cameras.json")
    assert len(cameras.cameras) == metadata["expected_metrics"]["view_count"]
    assert sorted(path.stem for path in roots.gt_dir(dataset, scene).glob("*.png")) == sorted(camera.name for camera in cameras.cameras)
    published: MetricsFile = load_metrics_file(run_dir / "published_metrics.json")
    assert published.scorer == "upstream-float"
    assert [view.name for view in published.per_view] == [camera.name for camera in cameras.cameras]
    assert published.psnr == pytest.approx(metadata["expected_metrics"]["mean_psnr_db"], abs=1e-9)
