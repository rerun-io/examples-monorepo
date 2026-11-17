"""LightGlue matching and COLMAP-side verification, on real Galileo and RoboCap frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest
from jaxtyping import Float32
from numpy import ndarray

from colsfm.database import ImagePair, KeypointsXY, MatchIndices, create_database, read_two_view_geometry
from colsfm.features import FeatureOptions, extract_features
from colsfm.frames_meta import FramesMeta
from colsfm.matching import (
    BLOB_MATCH_TOP_K,
    BLOB_MAX_PIXEL_ERROR,
    BLOB_MIN_NUM_INLIERS,
    BLOB_NUM_POINTS_TOLERANCE_FRACTION,
    BLOB_RANSAC_CONFIDENCE,
    MatchingOptions,
    MatchLimitPolicy,
    MatchReport,
    MatchScores,
    PairMatchStats,
    match_pairs,
    select_by_square_covering,
    subsample_matches_by_coverage,
    verification_options,
)
from colsfm.pairs import select_pairs, stereo_pairs

FAST_IMAGE_SIZE: int = 640
"""Detector input size for the tests; COLMAP rescales keypoints back to the source image."""

FAST_MAX_FEATURES: int = 512
"""Keypoint budget for the tests. LightGlue's cost grows with the keypoint count,
and on the CPU provider a 2048x2048 pair takes ~5 s against ~0.4 s here."""

FAST_FEATURES: FeatureOptions = FeatureOptions(max_image_size=FAST_IMAGE_SIZE, max_num_features=FAST_MAX_FEATURES)
"""Extraction settings shared by every matching test."""


def _prepare(database_path: Path, frames_meta: FramesMeta, image_root: Path, keyframe_ids: list[int]) -> FramesMeta:
    """Create a database over a few keyframes and extract their features.

    Args:
        database_path: Destination database.
        frames_meta: The collection the keyframes come from.
        image_root: Directory the `image_name` values are relative to.
        keyframe_ids: Keyframes to keep.

    Returns:
        The filtered collection that was imported.
    """
    subset: FramesMeta = frames_meta.filtered(keyframe_ids)
    create_database(database_path, subset)
    extract_features(database_path, image_root, [keyframe.image_name for keyframe in subset.keyframes], FAST_FEATURES)
    return subset


@pytest.fixture(scope="module")
def galileo_matched(
    galileo_input: FramesMeta, repo_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, list[ImagePair], MatchReport]:
    """Match the first two Galileo rig frames once and reuse the result."""
    rig_frames = galileo_input.rig_frames()[:2]
    keyframe_ids: list[int] = [keyframe_id for rig_frame in rig_frames for keyframe_id in rig_frame.keyframe_ids][:6]
    database_path: Path = tmp_path_factory.mktemp("matching") / "galileo.db"
    subset: FramesMeta = _prepare(database_path, galileo_input, repo_root / "data" / "r2b_galileo", keyframe_ids)
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    report: MatchReport = match_pairs(database_path, pairs, MatchingOptions())
    return database_path, pairs, report


def test_every_pair_is_matched_and_verified(galileo_matched: tuple[Path, list[ImagePair], MatchReport]) -> None:
    """Each requested pair gets raw matches, a geometry row and an entry in the report."""
    database_path, pairs, report = galileo_matched
    assert pairs
    assert set(report.pair_stats) == set(pairs)
    assert report.elapsed_seconds > 0.0
    assert report.empty_pairs == 0
    assert report.median_inliers > 50.0
    # Verification is a light filter, not a gate: the blob's median retention is 0.985.
    assert report.median_retention > 0.8

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    assert database.num_matched_image_pairs() == len(pairs)
    assert database.num_verified_image_pairs() == len(pairs)
    database.close()


def test_geometries_take_the_calibrated_path(galileo_matched: tuple[Path, list[ImagePair], MatchReport]) -> None:
    """PINHOLE cameras carrying a focal prior verify as CALIBRATED, not UNCALIBRATED."""
    database_path, pairs, _ = galileo_matched
    for pair in pairs:
        geometry: pycolmap.TwoViewGeometry = read_two_view_geometry(database_path, pair[0], pair[1])
        assert geometry.config == int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)


def test_stereo_pairs_inside_one_rig_frame_are_matched(
    galileo_input: FramesMeta, galileo_matched: tuple[Path, list[ImagePair], MatchReport]
) -> None:
    """`skip_image_pairs_in_same_frame` must stay off: cuSFM matches the stereo baseline.

    COLMAP 4.2's matcher is rig aware and can drop pairs whose images share a
    frame; those are exactly the `EXTRINSIC` pairs cuSFM keeps.
    """
    _, pairs, report = galileo_matched
    assert MatchingOptions().skip_image_pairs_in_same_frame is False
    keyframe_by_id = galileo_input.keyframe_by_id()
    same_frame: list[ImagePair] = [
        pair
        for pair in pairs
        if keyframe_by_id[pair[0]].synced_sample_id == keyframe_by_id[pair[1]].synced_sample_id
    ]
    assert same_frame, "expected the rig's stereo pairs in the pair list"
    for pair in same_frame:
        assert report.pair_stats[pair].inlier_matches > 0


def test_min_num_inliers_empties_a_pair_instead_of_trimming_it(
    galileo_input: FramesMeta, repo_root: Path, tmp_path: Path
) -> None:
    """The blob drops the whole pair below 10 inliers; COLMAP's option does the same.

    COLMAP goes one step further than the blob: `min_num_inliers` is applied to
    the *raw* match count as well, so a pair that cannot possibly reach the
    threshold loses its `matches` row too, not just its geometry, and the pair is
    stored as an empty DEGENERATE row. The blob keeps the raw matches and empties
    only the result.
    """
    rig_frame = galileo_input.rig_frames()[0]
    keyframe_by_id = galileo_input.keyframe_by_id()
    stereo_camera_ids: set[int] = {galileo_input.stereo_pairs[0].left_camera_params_id, galileo_input.stereo_pairs[0].right_camera_params_id}
    keyframe_ids: list[int] = [
        keyframe_id
        for keyframe_id in rig_frame.keyframe_ids
        if keyframe_by_id[keyframe_id].camera_params_id in stereo_camera_ids
    ]
    database_path: Path = tmp_path / "thin.db"
    subset: FramesMeta = _prepare(database_path, galileo_input, repo_root / "data" / "r2b_galileo", keyframe_ids)
    pairs: list[ImagePair] = stereo_pairs(subset)
    assert len(pairs) == 1

    permissive: MatchReport = match_pairs(database_path, pairs, MatchingOptions())
    kept: PairMatchStats = permissive.pair_stats[pairs[0]]
    assert kept.raw_matches > 0
    assert kept.inlier_matches > 0

    database: pycolmap.Database = pycolmap.Database.open(database_path)
    database.clear_matches()
    database.clear_two_view_geometries()
    database.close()

    strict: MatchReport = match_pairs(database_path, pairs, MatchingOptions(min_num_inliers=1_000_000))
    dropped: PairMatchStats = strict.pair_stats[pairs[0]]
    assert dropped.inlier_matches == 0
    assert dropped.raw_matches == 0
    assert strict.empty_pairs == 1
    # COLMAP's `WriteVerifiedTwoViewGeometry` at the pin: a rejected pair is stored as an empty DEGENERATE row.
    stored: pycolmap.TwoViewGeometry = read_two_view_geometry(database_path, *pairs[0])
    assert stored.config == int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE)
    assert len(stored.inlier_matches) == 0


def test_verification_options_carry_the_blob_thresholds() -> None:
    """4.0 px, confidence 0.98 and a 10-inlier floor, straight from the worker config.

    COLMAP works in pixels against the real camera model, so the blob's
    conversion of the pixel error onto the normalised plane has no counterpart.
    """
    options: pycolmap.TwoViewGeometryOptions = verification_options(MatchingOptions())
    assert options.min_num_inliers == BLOB_MIN_NUM_INLIERS
    assert options.ransac.max_error == pytest.approx(BLOB_MAX_PIXEL_ERROR)
    assert options.ransac.confidence == pytest.approx(BLOB_RANSAC_CONFIDENCE)


def test_empty_pair_list_does_no_work(galileo_input: FramesMeta, tmp_path: Path) -> None:
    """An empty pair list returns an empty report rather than invoking COLMAP."""
    database_path: Path = tmp_path / "empty.db"
    create_database(database_path, galileo_input.filtered([galileo_input.keyframes[0].keyframe_id]))
    report: MatchReport = match_pairs(database_path, [])
    assert report.pair_stats == {}
    assert report.elapsed_seconds == 0.0
    assert "no pairs" in report.summary()


def test_match_pairs_reports_a_missing_database(tmp_path: Path) -> None:
    """A missing database is an error, not an empty match run."""
    with pytest.raises(FileNotFoundError):
        match_pairs(tmp_path / "absent.db", [(1, 2)])


def test_pair_stats_retention_is_safe_on_an_empty_pair() -> None:
    """A pair with no raw matches reports zero retention rather than dividing by zero."""
    assert PairMatchStats(raw_matches=0, inlier_matches=0).retention == 0.0
    assert PairMatchStats(raw_matches=200, inlier_matches=190).retention == pytest.approx(0.95)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial-coverage subsampling (the score-free stand-in for the blob's SSC NMS)
# ─────────────────────────────────────────────────────────────────────────────


def _dense_grid_keypoints(num_columns: int, num_rows: int, width: int, height: int) -> KeypointsXY:
    """Keypoints on a regular lattice covering the image.

    Args:
        num_columns: Lattice columns.
        num_rows: Lattice rows.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Float32 `[num_columns * num_rows, 2]` xy pixel coordinates, row-major.
    """
    xs: Float32[ndarray, "num_columns"] = np.linspace(0.0, width - 1.0, num_columns, dtype=np.float32)
    ys: Float32[ndarray, "num_rows"] = np.linspace(0.0, height - 1.0, num_rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)


def test_subsample_keeps_everything_below_the_cap() -> None:
    """Fewer matches than the cap is a no-op, so a sparse pair is never thinned."""
    keypoints_xy: KeypointsXY = _dense_grid_keypoints(10, 10, 1920, 1200)
    matches: MatchIndices = np.stack([np.arange(100), np.arange(100)], axis=1).astype(np.int64)
    kept: MatchIndices = subsample_matches_by_coverage(keypoints_xy, matches, 1920, 1200, max_matches=500)
    assert np.array_equal(kept, matches)


def test_subsample_thins_a_dense_pair_towards_the_cap() -> None:
    """A dense pair comes back at most one match per grid cell, near the cap."""
    keypoints_xy: KeypointsXY = _dense_grid_keypoints(80, 50, 1920, 1200)
    matches: MatchIndices = np.stack([np.arange(4000), np.arange(4000)], axis=1).astype(np.int64)
    kept: MatchIndices = subsample_matches_by_coverage(keypoints_xy, matches, 1920, 1200, max_matches=500)
    assert 300 <= len(kept) <= 500
    assert len(kept) < len(matches)


def test_subsample_spreads_the_survivors_over_the_image() -> None:
    """The survivors cover the image: no quadrant is emptied by the thinning.

    The blob's SSC NMS exists to keep the matches spatially uniform; a filter that
    kept the first 500 rows would pass the cap test and fail this one.
    """
    width: int = 1920
    height: int = 1200
    keypoints_xy: KeypointsXY = _dense_grid_keypoints(80, 50, width, height)
    matches: MatchIndices = np.stack([np.arange(4000), np.arange(4000)], axis=1).astype(np.int64)
    kept: MatchIndices = subsample_matches_by_coverage(keypoints_xy, matches, width, height, max_matches=500)
    kept_xy: KeypointsXY = keypoints_xy[kept[:, 0]]
    in_right_half: int = int((kept_xy[:, 0] >= width / 2).sum())
    in_bottom_half: int = int((kept_xy[:, 1] >= height / 2).sum())
    assert in_right_half >= len(kept) // 4
    assert in_bottom_half >= len(kept) // 4


def test_subsample_is_deterministic_and_keeps_the_lowest_index_per_cell() -> None:
    """Two keypoints in one cell collapse to the one with the lower keypoint index.

    A 100x100 image at `max_matches=4` is a 2x2 grid of 50 px cells, so keypoints
    0 and 1 share the top-left cell and the other three sit alone.
    """
    keypoints_xy: KeypointsXY = np.array(
        [[10.0, 10.0], [11.0, 11.0], [60.0, 60.0], [70.0, 10.0], [10.0, 70.0]], dtype=np.float32
    )
    matches: MatchIndices = np.array([[1, 7], [0, 3], [2, 9], [3, 1], [4, 5]], dtype=np.int64)
    kept: MatchIndices = subsample_matches_by_coverage(keypoints_xy, matches, 100, 100, max_matches=4)
    assert np.array_equal(kept, np.array([[0, 3], [2, 9], [3, 1], [4, 5]], dtype=np.int64))
    again: MatchIndices = subsample_matches_by_coverage(keypoints_xy, matches, 100, 100, max_matches=4)
    assert np.array_equal(kept, again)


def test_subsample_survives_an_empty_match_list() -> None:
    """An emptied pair subsamples to an empty array rather than raising."""
    keypoints_xy: KeypointsXY = _dense_grid_keypoints(4, 4, 640, 480)
    matches: MatchIndices = np.zeros((0, 2), dtype=np.int64)
    kept: MatchIndices = subsample_matches_by_coverage(keypoints_xy, matches, 640, 480, max_matches=500)
    assert kept.shape == (0, 2)


def test_matching_caps_the_verified_matches_written_to_the_database(
    galileo_input: FramesMeta, repo_root: Path, tmp_path: Path
) -> None:
    """A fixed `match_limit` reduces what the database stores; `unlimited` leaves it alone."""
    keyframe_ids: list[int] = [keyframe_id for rig_frame in galileo_input.rig_frames()[:2] for keyframe_id in rig_frame.keyframe_ids][:4]
    image_root: Path = repo_root / "data" / "r2b_galileo"

    uncapped_db: Path = tmp_path / "uncapped.db"
    subset: FramesMeta = _prepare(uncapped_db, galileo_input, image_root, keyframe_ids)
    pairs: list[ImagePair] = select_pairs(subset, connected_keyframe_num=1)
    uncapped: MatchReport = match_pairs(uncapped_db, pairs, MatchingOptions(match_limit=MatchLimitPolicy.of("off", None)))

    capped_db: Path = tmp_path / "capped.db"
    _prepare(capped_db, galileo_input, image_root, keyframe_ids)
    cap: int = 40
    capped: MatchReport = match_pairs(capped_db, pairs, MatchingOptions(match_limit=MatchLimitPolicy.of("fixed", cap)))

    for pair in pairs:
        stored: int = len(read_two_view_geometry(capped_db, pair[0], pair[1]).inlier_matches)
        assert stored <= cap
        assert capped.pair_stats[pair].inlier_matches == stored
        assert stored <= uncapped.pair_stats[pair].inlier_matches
    assert sum(stats.inlier_matches for stats in capped.pair_stats.values()) < sum(
        stats.inlier_matches for stats in uncapped.pair_stats.values()
    )


# ── SSC spatial NMS: the blob's real one, reachable only with per-match scores ────────


SSC_IMAGE_WIDTH: int = 1920
"""Galileo's frame width, the size `BLOB_MATCH_TOP_K` was calibrated at."""

SSC_IMAGE_HEIGHT: int = 1200
"""Galileo's frame height."""


def _scattered_points(count: int, seed: int, width: int = SSC_IMAGE_WIDTH, height: int = SSC_IMAGE_HEIGHT) -> tuple[KeypointsXY, MatchScores]:
    """Uniformly scattered points with distinct scores.

    Args:
        count: How many points.
        seed: Generator seed, so every test is reproducible.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Float32 `[count, 2]` xy pixel coordinates and Float32 `[count]` scores.
    """
    generator: np.random.Generator = np.random.default_rng(seed)
    positions_xy: KeypointsXY = (generator.random((count, 2)) * np.array([width, height])).astype(np.float32)
    scores: MatchScores = generator.random(count).astype(np.float32)
    return positions_xy, scores


def test_square_covering_keeps_everything_at_or_below_the_target() -> None:
    """The blob's `matches.size() < match_top_k` early return: a sparse pair is untouched."""
    positions_xy, scores = _scattered_points(count=120, seed=0)
    kept = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, BLOB_MATCH_TOP_K)
    assert np.array_equal(kept, np.arange(120))


def test_square_covering_lands_inside_the_tolerance_band() -> None:
    """SSC targets a band, not a ceiling: `match_top_k` plus or minus 30 %.

    That is the one behavioural difference from `subsample_matches_by_coverage`,
    which is a hard maximum. Checked at the three densities the real data spans:
    Galileo's ~931 raw matches per pair, KITTI's ~1466, and a saturated 4000.
    """
    lowest: int = round(BLOB_MATCH_TOP_K * (1.0 - BLOB_NUM_POINTS_TOLERANCE_FRACTION))
    highest: int = round(BLOB_MATCH_TOP_K * (1.0 + BLOB_NUM_POINTS_TOLERANCE_FRACTION))
    for count in (931, 1466, 4000):
        positions_xy, scores = _scattered_points(count=count, seed=count)
        kept = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, BLOB_MATCH_TOP_K)
        assert lowest <= len(kept) <= highest, f"{count} points thinned to {len(kept)}"
        assert len(kept) < count


def test_square_covering_returns_ascending_unique_indices() -> None:
    """The result indexes the input once each, in order, so a caller can slice with it."""
    positions_xy, scores = _scattered_points(count=2000, seed=7)
    kept = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, BLOB_MATCH_TOP_K)
    assert kept.dtype == np.int64
    assert np.all(np.diff(kept) > 0)
    assert kept.min() >= 0
    assert kept.max() < 2000


def test_square_covering_is_deterministic() -> None:
    """Same points, same scores, same answer: nothing here reads a clock or a seed."""
    positions_xy, scores = _scattered_points(count=3000, seed=11)
    first = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, BLOB_MATCH_TOP_K)
    second = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, BLOB_MATCH_TOP_K)
    assert np.array_equal(first, second)


def test_square_covering_spreads_the_survivors_over_the_image() -> None:
    """Every quadrant keeps roughly its share, which is the whole point of SSC.

    A filter that kept the 500 highest scores would pass the band test and fail
    this one whenever the scores correlate with position — and on real imagery
    they do, because LightGlue is confident where the texture is.
    """
    positions_xy, _ = _scattered_points(count=4000, seed=3)
    # Scores that fall off towards the right edge, so "strongest first" alone would
    # empty the right half.
    scores: MatchScores = (1.0 - positions_xy[:, 0] / SSC_IMAGE_WIDTH).astype(np.float32)
    kept = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, BLOB_MATCH_TOP_K)
    kept_xy: KeypointsXY = positions_xy[kept]
    quadrant_counts: list[int] = [
        int(((kept_xy[:, 0] >= half_width * SSC_IMAGE_WIDTH / 2) & (kept_xy[:, 0] < (half_width + 1) * SSC_IMAGE_WIDTH / 2)
             & (kept_xy[:, 1] >= half_height * SSC_IMAGE_HEIGHT / 2) & (kept_xy[:, 1] < (half_height + 1) * SSC_IMAGE_HEIGHT / 2)).sum())
        for half_width in (0, 1)
        for half_height in (0, 1)
    ]
    assert min(quadrant_counts) >= len(kept) // 8, f"quadrants {quadrant_counts} of {len(kept)}"


def test_square_covering_survives_a_target_of_one() -> None:
    """A one-match target keeps exactly the strongest point, not a division by zero.

    `image_area` mode can resolve to 1 on a tiny frame, and SSC's upper search
    bound divides by `2 * target - 2`.
    """
    positions_xy, scores = _scattered_points(count=50, seed=5)
    kept = select_by_square_covering(positions_xy, scores, SSC_IMAGE_WIDTH, SSC_IMAGE_HEIGHT, target=1)
    assert kept.tolist() == [int(np.argmax(scores))]


def test_square_covering_prefers_the_stronger_of_two_neighbours() -> None:
    """Inside one suppression square the higher score wins, and only it survives.

    Four points, two of them 5 px apart in the top-left corner: at a target of 3
    over a 400x400 image the suppression square is far wider than 5 px, so the
    pair collapses to whichever carries the higher score.
    """
    positions_xy: KeypointsXY = np.array([[10.0, 10.0], [15.0, 12.0], [390.0, 10.0], [10.0, 390.0]], dtype=np.float32)
    weaker_first: MatchScores = np.array([0.1, 0.9, 0.5, 0.5], dtype=np.float32)
    kept = select_by_square_covering(positions_xy, weaker_first, 400, 400, target=3)
    assert 0 not in kept.tolist()
    assert 1 in kept.tolist()
    stronger_first: MatchScores = np.array([0.9, 0.1, 0.5, 0.5], dtype=np.float32)
    kept = select_by_square_covering(positions_xy, stronger_first, 400, 400, target=3)
    assert 0 in kept.tolist()
    assert 1 not in kept.tolist()


# ── The cap modes ────────────────────────────────────────────────────────────────────


def test_the_fixed_cap_mode_ignores_the_image_size() -> None:
    """`fixed` is the historical behaviour: 500 everywhere."""
    policy: MatchLimitPolicy = MatchLimitPolicy.of("fixed", BLOB_MATCH_TOP_K)
    assert policy.limit_for(1920, 1200) == BLOB_MATCH_TOP_K
    assert policy.limit_for(1241, 376) == BLOB_MATCH_TOP_K


def test_the_two_spellings_of_no_cap_resolve_to_one_policy() -> None:
    """`off` and a None budget both mean "keep every verified inlier", and now say so once."""
    off: MatchLimitPolicy = MatchLimitPolicy.of("off", BLOB_MATCH_TOP_K)
    no_budget: MatchLimitPolicy = MatchLimitPolicy.of("fixed", None)
    assert off == no_budget == MatchLimitPolicy(kind="unlimited", matches_per_pair=None)
    assert not off.limits
    assert off.limit_for(1920, 1200) is None
    assert no_budget.limit_for(1920, 1200) is None


def test_the_image_area_cap_mode_scales_with_resolution() -> None:
    """`image_area` keeps the calibration's density: 500 at 1920x1200, 101 on KITTI."""
    policy: MatchLimitPolicy = MatchLimitPolicy.of("image_area", BLOB_MATCH_TOP_K)
    assert policy.limit_for(1920, 1200) == BLOB_MATCH_TOP_K
    assert policy.limit_for(1241, 376) == 101
    assert policy.limit_for(3840, 2400) == 4 * BLOB_MATCH_TOP_K
    # A degenerate size still resolves to a usable positive cap rather than 0.
    assert policy.limit_for(1, 1) == 1
