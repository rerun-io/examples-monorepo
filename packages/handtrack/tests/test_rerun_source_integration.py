"""The Rerun-dataloader producer streams labelled, letterboxed DetNet samples from the real catalog."""

from contextlib import closing
from pathlib import Path

import pytest
import torch
from beartype.roar import BeartypeException

pytestmark = pytest.mark.integration

rr = pytest.importorskip("rerun", reason="needs rerun-sdk")
pytest.importorskip("rerun.experimental.dataloader", reason="needs the rerun dataloader extra")

from handtrack.data.catalog import CATALOG_URL  # noqa: E402
from handtrack.data.rerun_source import SourceSpec, build_manifest, rerun_producer  # noqa: E402

SPECS: dict[str, SourceSpec] = {
    "umetrack": SourceSpec("dataforge-umetrack", 0, tuple(f"/world/rig_00/cam_0{i}" for i in range(4)), "av1", "hand_model", 0),
    "gen-humanego": SourceSpec("dataforge-gen-humanego-subset", 3, tuple(f"/world/rig_00/cam_0{i}" for i in range(6)), "h264", "coco133", 14, 0.8),
}


@pytest.mark.parametrize("key", list(SPECS))
def test_producer_streams_labelled_net_frames(key: str, tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("the producer letterboxes on the training device")
    spec: SourceSpec = SPECS[key]
    try:
        dataset = rr.catalog.CatalogClient(CATALOG_URL).get_dataset(spec.dataset)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {CATALOG_URL} dataset {spec.dataset} unreachable: {error}")
    segments: list[str] = sorted(dataset.segment_ids())[:2]
    manifest: Path = build_manifest(spec, segments, CATALOG_URL, tmp_path / "manifest.parquet", workers=2)
    produce = rerun_producer(spec, segments, catalog_url=CATALOG_URL, manifest_path=manifest, workers=2, device=torch.device("cuda"))
    # 2 x 256 samples: Gen-HumanEgo's 6 H.264 cameras decode on the CPU here. Close the producer: an abandoned one keeps its spawn
    # workers (and their torch_shm_manager) alive, and they hold the resource tracker's pipe, so the interpreter never exits.
    with closing(produce(0)) as stream:
        chunks = [chunk for _, chunk in zip(range(2), stream, strict=False)]
    assert chunks
    present: int = 0
    for chunk in chunks:
        assert chunk.pooled.shape[1:] == (120, 160)
        assert bool((chunk.dataset == spec.dataset_index).all())
        assert bool((chunk.camera >= spec.camera_offset).all()) and bool((chunk.camera < spec.camera_offset + len(spec.cameras)).all())
        assert int(chunk.pooled.max()) > 0
        circles = chunk.circle[chunk.circle_mask]
        assert bool(((circles[:, :2] > -0.5) & (circles[:, :2] < 1.5)).all())
        present += int(chunk.circle_mask.sum())
    assert present > 0
