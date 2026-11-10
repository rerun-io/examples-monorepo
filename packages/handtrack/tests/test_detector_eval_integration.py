"""The detector evaluation scores real held-out frames: the same weights under two names score identically, and the deployed
detector finds most hands on a base dataset it was trained beside."""

import json
from pathlib import Path

import pytest
import torch
from beartype.roar import BeartypeException

pytestmark = pytest.mark.integration

rr = pytest.importorskip("rerun", reason="needs rerun-sdk")
pytest.importorskip("rerun.experimental.dataloader", reason="needs the rerun dataloader extra")

from handtrack.apis.detector_eval import DetectorEvalConfig, main  # noqa: E402
from handtrack.data.catalog import CATALOG_URL  # noqa: E402

DEPLOYED: Path = Path.home() / "handtrack-data/checkpoints/detnet-demo-0929.weights.pt"
HOLDOUT: Path = Path(__file__).parents[1] / "configs/detector_holdout.json"


def test_two_names_for_one_model_score_the_same_frames_identically(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the evaluation decodes on NVDEC")
    if not DEPLOYED.exists():
        pytest.skip(f"deployed DetNet-F weights absent: {DEPLOYED}")
    try:
        rr.catalog.CatalogClient(CATALOG_URL).get_dataset("gsi-six-view")
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {CATALOG_URL} unreachable: {error}")
    holdout = json.loads(HOLDOUT.read_text())
    one_episode = tmp_path / "holdout.json"
    one_episode.write_text(json.dumps({"note": holdout["note"], "segments": {"gsi-six-view": holdout["segments"]["gsi-six-view"][:1]}}))
    report = main(DetectorEvalConfig(names=("a", "b"), weights=(DEPLOYED, DEPLOYED), out_dir=tmp_path, datasets=("gsi-six-view", "dataforge-umetrack"),
                                     holdout=one_episode, base_segments=1, misses_per_dataset=4))
    for result in report.datasets.values():
        assert result.images > 0 and len(result.segments) == 1
        assert result.models["a"] == result.models["b"]
        assert result.models["a"].both.present == result.models["a"].left.present + result.models["a"].right.present
    umetrack = report.datasets["dataforge-umetrack"].models["a"].both
    assert umetrack.recall is not None and umetrack.recall > 0.5
    assert (tmp_path / "detector_eval.json").exists() and (tmp_path / "misses.rrd").stat().st_size > 0
