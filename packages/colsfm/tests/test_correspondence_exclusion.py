"""The mapper's correspondence graph can trim a pair's inliers without rewriting database rows.

COLMAP rewrites the whole database (`VACUUM`) when it closes a connection that deleted anything, so the graph is rebuilt
from the database's rows with the loop stage's inlier overrides instead; with no override it is the cache's own graph,
pair for pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from mapping_helpers import SyntheticRig, synthetic_matches, write_database

from colsfm.correspondences import Correspondences, load_correspondences
from colsfm.reconstruction import build_reconstruction


@pytest.fixture(scope="module")
def database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path: Path = tmp_path_factory.mktemp("exclusion") / "database.db"
    write_database(path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, synthetic_matches(synthetic_rig))
    return path


def _pairs(graph: pycolmap.CorrespondenceGraph) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for pair_id in graph.image_pairs():
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
        out[(min(image_id1, image_id2), max(image_id1, image_id2))] = graph.num_matches_between_images(image_id1, image_id2)
    return out


def _assert_same_graph(loaded: Correspondences) -> None:
    """The rebuilt graph is the cache's: same pairs, same counts, every correspondence list in the same order."""
    cached: pycolmap.CorrespondenceGraph = loaded.database_cache.correspondence_graph
    assert _pairs(loaded.graph) == _pairs(cached)
    assert loaded.graph.num_matches_between_all_images() == cached.num_matches_between_all_images()
    # The triangulator walks these lists, so their order decides the points it makes.
    for image_id, image in loaded.database_cache.images.items():
        for point2D_idx in range(len(image.points2D)):
            mine = [(c.image_id, c.point2D_idx) for c in loaded.graph.extract_correspondences(image_id, point2D_idx)]
            theirs = [(c.image_id, c.point2D_idx) for c in cached.extract_correspondences(image_id, point2D_idx)]
            assert mine == theirs


def test_with_no_override_the_graph_is_the_caches_own(synthetic_rig: SyntheticRig, database: Path) -> None:
    _assert_same_graph(load_correspondences(build_reconstruction(synthetic_rig.frames_meta).reconstruction, database))


@pytest.mark.parametrize("min_num_matches", [0, 15, 200])
def test_unusual_rows_rebuild_to_the_caches_graph(synthetic_rig: SyntheticRig, tmp_path: Path, min_num_matches: int) -> None:
    """Rows a matcher rarely writes still rebuild to the cache's graph, whatever `min_num_matches` lets through.

    One pair has a many-to-one match listed before the one-to-one match it shares a keypoint with (the second image's
    keypoint then lists two first-image keypoints, in row order), and one row carries an essential matrix and no inlier.
    """
    matches = synthetic_matches(synthetic_rig)
    many_to_one: tuple[int, int] = sorted(matches)[0]
    rows = matches[many_to_one]
    matches[many_to_one] = np.vstack([[[rows[1, 0], rows[0, 1]]], rows]).astype(np.uint32)
    payload_only: tuple[int, int] = sorted(matches)[-1]
    del matches[payload_only]
    path: Path = tmp_path / "unusual.db"
    write_database(path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, matches)
    with pycolmap.Database.open(path) as handle:
        geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
        geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
        geometry.E = np.eye(3)
        handle.write_two_view_geometry(*payload_only, geometry)

    loaded: Correspondences = load_correspondences(
        build_reconstruction(synthetic_rig.frames_meta).reconstruction, path, min_num_matches=min_num_matches
    )
    _assert_same_graph(loaded)
    if min_num_matches > 15:
        assert len(_pairs(loaded.graph)) < len(matches), "the threshold drops pairs, and the rebuild drops the same ones"


def test_an_inlier_override_trims_one_pair_and_stays_out_of_the_database(synthetic_rig: SyntheticRig, database: Path) -> None:
    """The loop stage's pose check keeps a subset of a pair's inliers; the graph carries that subset, the database all of them."""
    full: dict[tuple[int, int], int] = _pairs(load_correspondences(build_reconstruction(synthetic_rig.frames_meta).reconstruction, database).graph)
    trimmed, emptied = sorted(full)[0], sorted(full)[2]
    with pycolmap.Database.open(database) as handle:
        rows: np.ndarray = np.asarray(handle.read_two_view_geometry(*trimmed).inlier_matches)
    kept: np.ndarray = rows[: len(rows) // 2]
    loaded: Correspondences = load_correspondences(
        build_reconstruction(synthetic_rig.frames_meta).reconstruction,
        database,
        min_num_matches=15,
        inlier_overrides={trimmed: kept, emptied: rows[:3]},
    )
    expected: dict[tuple[int, int], int] = {pair: n for pair, n in full.items() if pair != emptied}
    expected[trimmed] = len(kept)
    assert _pairs(loaded.graph) == expected, "a pair the override leaves below min_num_matches leaves the graph"
    with pycolmap.Database.open(database) as handle:
        assert len(handle.read_two_view_geometry(*trimmed).inlier_matches) == len(rows), "nothing is written back"
