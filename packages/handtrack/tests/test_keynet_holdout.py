"""The frozen KeyNet holdout is a cut list in code (``data.keynet_holdout_cuts``), cut from the catalog at the start of each run."""
import dataclasses
import hashlib

import pytest
import torch
from beartype.roar import BeartypeException

from handtrack.data.catalog import CATALOG_URL
from handtrack.data.keynet_holdout import HOLDOUT_DATASETS, KeyNetHoldout, load_holdout
from handtrack.data.keynet_holdout_cuts import HOLDOUT_CUTS, HOLDOUT_SEGMENTS
from handtrack.data.rerun_source import DATASET_SPECS
from handtrack.data.stream import KeyNetSamples

SAMPLES_SHA256: dict[str, tuple[int, str]] = {
    "dataforge-umetrack": (4106, "2f43edccff3e2a6ccb66038be49c762732eaaea818fc1f6688eff5014e274db4"),
    "dataforge-show3d": (4101, "e9d6c33197660c6d45961e1e1aeb6887fc5d63968c4204b96ebd3f2946755d29"),
    "dataforge-hot3d-quest3": (4145, "f08d145a4593607d009eb402580ca1d070a8edaff2ca3a7a0b894c8c68d847d8"),
    "gsi-six-view": (4079, "342cde62d0c8d15303fd36744fe1c45dc09bd37e1009ef03fb2288e28ea56dae"),
    "dataforge-gen-humanego-subset": (4109, "b2c221629fdd3a117139277179e88dea6dd5b101eb7508e26155ae35d27914a6"),
}
"""Rows and sha256 of each dataset's samples (every ``KeyNetSamples`` field, then camera and variant) as ``load_holdout`` cut them on
pablo-dl-server (RTX 5090), 2026-11-10; float fields are bit-exact on that host only."""


def samples_sha256(holdout: KeyNetHoldout, index: int) -> tuple[int, str]:
    chosen = holdout.samples.dataset == index
    digest = hashlib.sha256()
    for name in [f.name for f in dataclasses.fields(KeyNetSamples)]:
        digest.update(getattr(holdout.samples, name)[chosen].contiguous().numpy().tobytes())
    digest.update(holdout.camera[chosen].numpy().tobytes() + holdout.variant[chosen].numpy().tobytes())
    return int(chosen.sum()), digest.hexdigest()


def test_every_cut_names_a_held_segment_a_camera_and_frames() -> None:
    assert tuple(HOLDOUT_CUTS) == tuple(HOLDOUT_SEGMENTS) == HOLDOUT_DATASETS
    for name in HOLDOUT_DATASETS:
        cuts = HOLDOUT_CUTS[name]
        assert {cut.segment for cut in cuts} == set(range(len(HOLDOUT_SEGMENTS[name]))), name
        assert all(0 <= cut.camera < len(DATASET_SPECS[name][0]) and len(cut.frames) > 0 for cut in cuts), name


@pytest.mark.golden
def test_the_catalog_cuts_the_recorded_crops() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the holdout decodes on NVDEC")
    import rerun as rr

    try:
        rr.catalog.CatalogClient(CATALOG_URL).get_dataset(HOLDOUT_DATASETS[0])
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {CATALOG_URL} unreachable: {error}")
    holdout = load_holdout(CATALOG_URL, torch.device("cuda"))
    assert {name: samples_sha256(holdout, index) for index, name in enumerate(HOLDOUT_DATASETS)} == SAMPLES_SHA256


def test_the_holdout_refuses_a_cpu_cut() -> None:
    with pytest.raises(ValueError, match="cut on CUDA"):
        load_holdout(CATALOG_URL, torch.device("cpu"))
