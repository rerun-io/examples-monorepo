"""The fixed evaluation frames depend on ground truth only; the catalog labels project back through each native lens."""

import os
from dataclasses import replace

import numpy as np
import pytest
from beartype.roar import BeartypeException
from jaxtyping import Bool, Float64
from numpy import ndarray

from posekit.eval.hand_gt import EVAL_SUBSETS, select_frames, stored_upright_turns


def test_frames_span_visible_frames_without_repeats() -> None:
    valid: Bool[ndarray, "10 2 21"] = np.zeros((10, 2, 21), dtype=bool)
    valid[[1, 3, 5, 7, 9], 0, :] = True
    assert select_frames(valid, max_frames=3).tolist() == [1, 5, 9]


def test_frames_need_a_wrist_and_fifteen_joints() -> None:
    valid: Bool[ndarray, "3 2 21"] = np.ones((3, 2, 21), dtype=bool)
    valid[0, :, 0] = False
    valid[1, :, :8] = False
    assert select_frames(valid, max_frames=100).tolist() == [2]


def test_no_visible_hand_is_explicit() -> None:
    with pytest.raises(ValueError, match="visible"):
        select_frames(np.zeros((2, 2, 21), dtype=bool), max_frames=100)


@pytest.mark.integration
@pytest.mark.parametrize("dataset", ["dataforge-umetrack", "dataforge-hot3d-quest3", "gsi-six-view", "dataforge-gen-humanego-subset"])
def test_catalog_labels_project_back_to_the_source_camera(dataset: str) -> None:
    """SHOW3D is left out: its 2D labels are annotations, not projections of the 3D joints."""
    pytest.importorskip("warp", reason="camera projection needs the posekit-hands Pixi environment")
    pytest.importorskip("egohand_inference", reason="camera projection needs the posekit-hands Pixi environment")
    from rerun.catalog import CatalogClient, DatasetEntry

    from posekit.eval.hand_gt import read_hand_gt
    from posekit.models.egohand_geometry import project

    url: str = os.environ.get("POSEKIT_CATALOG_URL", "rerun+http://127.0.0.1:51235")
    try:
        entry: DatasetEntry = CatalogClient(url).get_dataset(dataset)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {url} dataset {dataset} unreachable: {error}")
    subset = next(subset for subset in EVAL_SUBSETS if subset.dataset == dataset)
    segment = read_hand_gt(entry, replace(subset, segments=subset.segments[:1]), max_frames=10)[0]
    assert len(segment.sample_indices) == 10
    errors: list[float] = []
    for xyz, uv, valid in zip(segment.xyz_camera, segment.uv, segment.valid, strict=True):
        predicted: Float64[ndarray, "2 21 2"] = project(segment.camera, xyz.reshape(-1, 3).astype(np.float64)).reshape(2, 21, 2)
        errors.extend(np.linalg.norm(predicted - uv, axis=-1)[valid].tolist())
    assert np.median(errors) < 1.0


def test_the_stored_upright_turn_comes_from_the_pinhole_and_defaults_to_none() -> None:
    import pyarrow as pa

    camera = "/world/rig_00/cam_01"
    converted = pa.table({f"{camera}/pinhole:upright_quarter_turns_cw": pa.array([[3]], pa.list_(pa.int64()))})
    assert stored_upright_turns(converted, camera) == 3
    assert stored_upright_turns(pa.table({"rerun_segment_id": ["s0"]}), camera) == 0, "converted as recorded"
