"""In-memory two-view verification: COLMAP's results, COLMAP's write rules, and a bounded number of chunks in flight."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from mapping_helpers import SyntheticRig, synthetic_matches, write_database

from colsfm import verification
from colsfm.database import ImagePair
from colsfm.matching import MatchingOptions, verification_options, verify_and_report
from colsfm.verification import ChunkedVerifier, PairToVerify, VerifiedPair, VerifyChunking, verify_pair


@pytest.fixture(scope="module")
def matched_database(synthetic_rig: SyntheticRig, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic rig's matches with no geometry yet: what a matcher leaves for verification."""
    path: Path = tmp_path_factory.mktemp("verification") / "database.db"
    write_database(path, synthetic_rig.frames_meta, synthetic_rig.keypoints_px, synthetic_matches(synthetic_rig))
    with pycolmap.Database.open(path) as database:
        database.clear_two_view_geometries()
    return path


def _geometries(path: Path) -> dict[ImagePair, tuple[int, int]]:
    with pycolmap.Database.open(path) as database:
        pair_ids, geometries = database.read_two_view_geometries()
    return {
        pycolmap.pair_id_to_image_pair(int(pair_id)): (int(g.config), len(g.inlier_matches)) for pair_id, g in zip(pair_ids, geometries, strict=True)
    }


def test_chunked_verification_writes_what_colmaps_verify_matches_writes(matched_database: Path, tmp_path: Path) -> None:
    """Same database, verified twice: COLMAP's batch `verify_matches`, and `verify_and_report` in chunks of 3 on 2 threads."""
    colmap_path: Path = tmp_path / "colmap.db"
    ours_path: Path = tmp_path / "ours.db"
    shutil.copy(matched_database, colmap_path)
    shutil.copy(matched_database, ours_path)
    with pycolmap.Database.open(matched_database) as database:
        pair_ids, _ = database.read_all_matches()
    pairs: list[ImagePair] = [pycolmap.pair_id_to_image_pair(int(pair_id)) for pair_id in pair_ids]
    options: MatchingOptions = MatchingOptions(verify=VerifyChunking(chunk_pairs=3, threads=2, chunks_in_flight=1))

    pair_list: Path = tmp_path / "pairs.txt"
    with pycolmap.Database.open(colmap_path) as database:
        names: dict[int, str] = {image.image_id: image.name for image in database.read_all_images()}
    pair_list.write_text("".join(f"{names[a]} {names[b]}\n" for a, b in pairs))
    pycolmap.verify_matches(colmap_path, pair_list, verification_options(options))
    report, _ = verify_and_report(ours_path, pairs, options, device="cpu", started=0.0, cap=False)

    colmap: dict[ImagePair, tuple[int, int]] = _geometries(colmap_path)
    ours: dict[ImagePair, tuple[int, int]] = _geometries(ours_path)
    assert set(ours) == set(colmap) == set(pairs)
    assert ours == colmap, "clean synthetic matches: every pair's configuration and inlier count agree"
    assert {pair: stats.inlier_matches for pair, stats in report.pair_stats.items()} == {pair: n for pair, (_, n) in ours.items()}


def test_too_few_inliers_become_an_empty_degenerate_row(synthetic_rig: SyntheticRig, matched_database: Path) -> None:
    """COLMAP's `WriteVerifiedTwoViewGeometry`: below `min_num_inliers` the pair is stored DEGENERATE, with nothing in it."""
    with pycolmap.Database.open(matched_database) as database:
        pair_ids, _ = database.read_all_matches()
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(int(pair_ids[0]))
        matches = np.asarray(database.read_matches(image_id1, image_id2), dtype=np.int64)
        keypoints1 = np.asarray(database.read_keypoints(image_id1))[:, :2].astype(np.float32)
        keypoints2 = np.asarray(database.read_keypoints(image_id2))[:, :2].astype(np.float32)
        cameras = {camera.camera_id: camera for camera in database.read_all_cameras()}
        camera_of = {image.image_id: cameras[image.camera_id] for image in database.read_all_images()}
    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.min_num_inliers = len(matches) + 1
    verified: VerifiedPair = verify_pair(
        PairToVerify(pair=(image_id1, image_id2), matches=matches, keypoints1=keypoints1, keypoints2=keypoints2),
        camera_of[image_id1],
        camera_of[image_id2],
        options,
        None,
    )
    assert verified.geometry is not None
    assert int(verified.geometry.config) == int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE)
    assert verified.num_inliers == 0


def test_at_most_chunks_in_flight_chunks_are_verifying(monkeypatch: pytest.MonkeyPatch) -> None:
    """With 2 in flight, the third submit waits for the first chunk and hands it back; finish drains the rest, in order."""

    def fake(item: PairToVerify, *_: object) -> VerifiedPair:
        return VerifiedPair(pair=item.pair, matches=item.matches, geometry=None, num_capped=0)

    monkeypatch.setattr(verification, "verify_pair", fake)

    def chunk(k: int) -> list[PairToVerify]:
        no_keypoints = np.zeros((0, 2), np.float32)
        return [PairToVerify(pair=(k, k + 100), matches=np.zeros((0, 2), dtype=np.int64), keypoints1=no_keypoints, keypoints2=no_keypoints)]

    cameras: dict[int, pycolmap.Camera] = {k: pycolmap.Camera() for k in (*range(5), *range(100, 105))}
    with ChunkedVerifier(cameras, pycolmap.TwoViewGeometryOptions(), chunking=VerifyChunking(threads=2, chunks_in_flight=2)) as verifier:
        handed_back: list[list[int]] = [[c[0].pair[0] for c in verifier.submit(chunk(k))] for k in range(5)]
        rest: list[int] = [c[0].pair[0] for c in verifier.finish()]
    assert handed_back == [[], [], [0], [1], [2]]
    assert rest == [3, 4]


@pytest.mark.parametrize("field", ["chunk_pairs", "threads", "chunks_in_flight"])
def test_a_chunk_pool_or_window_of_zero_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        VerifyChunking(**{field: 0})
