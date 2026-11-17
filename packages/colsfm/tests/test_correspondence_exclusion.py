"""The mapper's correspondence graph can leave pairs out without deleting database rows.

`loop_pairs_in_mapper="accepted"` used to delete the loop stage's rejected pairs, and COLMAP rewrites the whole database
(`VACUUM`) when it closes a connection that deleted anything. The graph is now rebuilt without them instead; with nothing
excluded it is the cache's own graph, pair for pair.
"""

from __future__ import annotations

from pathlib import Path

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


def test_with_nothing_excluded_the_graph_is_the_caches_own(synthetic_rig: SyntheticRig, database: Path) -> None:
    loaded: Correspondences = load_correspondences(build_reconstruction(synthetic_rig.frames_meta).reconstruction, database)
    assert _pairs(loaded.graph) == _pairs(loaded.database_cache.correspondence_graph)
    assert loaded.graph.num_matches_between_all_images() == loaded.database_cache.correspondence_graph.num_matches_between_all_images()
    # Same correspondence lists in the same order: the triangulator walks them, so the order decides the points it makes.
    cached: pycolmap.CorrespondenceGraph = loaded.database_cache.correspondence_graph
    for image_id, image in loaded.database_cache.images.items():
        for point2D_idx in range(len(image.points2D)):
            mine = [(c.image_id, c.point2D_idx) for c in loaded.graph.extract_correspondences(image_id, point2D_idx)]
            theirs = [(c.image_id, c.point2D_idx) for c in cached.extract_correspondences(image_id, point2D_idx)]
            assert mine == theirs


def test_excluded_pairs_leave_the_graph_and_stay_in_the_database(synthetic_rig: SyntheticRig, database: Path) -> None:
    full: dict[tuple[int, int], int] = _pairs(load_correspondences(build_reconstruction(synthetic_rig.frames_meta).reconstruction, database).graph)
    excluded: tuple[int, int] = sorted(full)[1]
    loaded: Correspondences = load_correspondences(
        build_reconstruction(synthetic_rig.frames_meta).reconstruction, database, excluded_pairs=[excluded[::-1]]
    )
    assert _pairs(loaded.graph) == {pair: n for pair, n in full.items() if pair != excluded}
    with pycolmap.Database.open(database) as handle:
        assert handle.exists_two_view_geometry(*excluded), "nothing is deleted: the database keeps the pair"
