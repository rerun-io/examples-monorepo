"""In-memory two-view verification: COLMAP's results and write rules, all-or-nothing chunk writes, a real in-flight limit.

The estimator is replaced by a fake with a fixed answer wherever a test is about what happens to a result rather than
how RANSAC reaches it, so these tests are deterministic and need no GPU.
"""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from mapping_helpers import SyntheticRig, synthetic_matches, write_database

from colsfm import verification
from colsfm.database import ImagePair, Keypoints, KeypointsXY, MatchIndices, read_keypoints_batch
from colsfm.matching import MatchingOptions, PairMatchStats, verification_options, verifier_inputs, verify_and_report, write_and_count
from colsfm.verification import CapFunction, ChunkedVerifier, PairToVerify, VerifiedPair, VerifyChunking, verify_pair, write_verified

DEGENERATE: int = int(pycolmap.TwoViewGeometryConfiguration.DEGENERATE)
CALIBRATED: int = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
UNDEFINED: int = int(pycolmap.TwoViewGeometryConfiguration.UNDEFINED)
NO_KEYPOINTS: KeypointsXY = np.zeros((300, 2), np.float32)


def _raw(num_matches: int) -> MatchIndices:
    return np.stack([np.arange(num_matches), np.arange(num_matches)], axis=1).astype(np.int64)


def _item(pair: ImagePair, num_matches: int = 200) -> PairToVerify:
    return PairToVerify(pair=pair, matches=_raw(num_matches), keypoints1=NO_KEYPOINTS, keypoints2=NO_KEYPOINTS)


def _estimate_returning(config: int, num_inliers: int) -> Callable[..., pycolmap.TwoViewGeometry]:
    """A stand-in for `pycolmap.estimate_two_view_geometry` with a fixed answer, models included."""

    def estimate(*_: object) -> pycolmap.TwoViewGeometry:
        geometry: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
        geometry.config = config
        geometry.E = np.eye(3)
        geometry.F = np.eye(3)
        geometry.inlier_matches = _raw(num_inliers).astype(np.uint32)
        return geometry

    return estimate


def _verify(item: PairToVerify, options: pycolmap.TwoViewGeometryOptions | None = None, cap: CapFunction | None = None) -> VerifiedPair:
    return verify_pair(item, pycolmap.Camera(), pycolmap.Camera(), options or pycolmap.TwoViewGeometryOptions(), cap)


def _write(path: Path, chunk: list[VerifiedPair], *, write_matches: bool) -> None:
    with pycolmap.Database.open(path) as database:
        write_verified(database, chunk, write_matches=write_matches)


def _geometry_rows(path: Path) -> dict[ImagePair, tuple[int, int]]:
    """Every stored geometry row as `(config, inliers)`, empty ones included (`read_two_view_geometries` skips those)."""
    rows: dict[ImagePair, tuple[int, int]] = {}
    with pycolmap.Database.open(path) as database:
        for pair in [(a, b) for a in range(1, 8) for b in range(a + 1, 8)]:
            if database.exists_two_view_geometry(*pair):
                stored: pycolmap.TwoViewGeometry = database.read_two_view_geometry(*pair)
                rows[pair] = (int(stored.config), len(stored.inlier_matches))
    return rows


def _match_rows(path: Path) -> dict[ImagePair, int]:
    with pycolmap.Database.open(path) as database:
        return {(a, b): len(database.read_matches(a, b)) for a in range(1, 8) for b in range(a + 1, 8) if database.exists_matches(a, b)}


# --- COLMAP's results -----------------------------------------------------------------------------------------------


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
        names: dict[int, str] = {image.image_id: image.name for image in database.read_all_images()}
    pairs: list[ImagePair] = [pycolmap.pair_id_to_image_pair(int(pair_id)) for pair_id in pair_ids]
    options: MatchingOptions = MatchingOptions(verify=VerifyChunking(verify_chunk_pairs=3, verify_threads=2, verify_chunks_in_flight=1))

    pair_list: Path = tmp_path / "pairs.txt"
    pair_list.write_text("".join(f"{names[a]} {names[b]}\n" for a, b in pairs))
    pycolmap.verify_matches(colmap_path, pair_list, verification_options(options))
    report, _ = verify_and_report(ours_path, pairs, options, device="cpu", started=0.0, cap=False)

    colmap: dict[ImagePair, tuple[int, int]] = _geometries(colmap_path)
    ours: dict[ImagePair, tuple[int, int]] = _geometries(ours_path)
    assert set(ours) == set(colmap) == set(pairs)
    assert ours == colmap, "clean synthetic matches: every pair's configuration and inlier count agree"
    assert {pair: stats.inlier_matches for pair, stats in report.pair_stats.items()} == {pair: n for pair, (_, n) in ours.items()}


def test_a_pair_verifies_the_same_way_whatever_was_verified_before_on_its_thread(matched_database: Path) -> None:
    """The rig's pair (0, 2), on the CALIBRATED / PLANAR_OR_PANORAMIC boundary, verified 30 times in a row on one thread.

    Unseeded, COLMAP's RANSAC continues a thread-local PRNG stream, so this pair came out PLANAR_OR_PANORAMIC on about
    3 calls in 4 and CALIBRATED on the rest, by the work its thread did before (the chunked-vs-COLMAP comparison above
    failed 1 run in 3 that way).
    """
    pair: ImagePair = (0, 2)
    keypoints: dict[int, Keypoints] = read_keypoints_batch(matched_database, list(pair))
    with pycolmap.Database.open(matched_database) as database:
        cameras, _ = verifier_inputs(database)
        matches: MatchIndices = np.asarray(database.read_matches(*pair), dtype=np.int64).reshape(-1, 2)
    item: PairToVerify = PairToVerify(pair=pair, matches=matches, keypoints1=keypoints[pair[0]], keypoints2=keypoints[pair[1]])
    options: pycolmap.TwoViewGeometryOptions = verification_options(MatchingOptions())
    answers: set[tuple[int, int]] = set()
    for _ in range(30):
        verified: VerifiedPair = verify_pair(item, cameras[pair[0]], cameras[pair[1]], options, None)
        assert verified.geometry is not None
        answers.add((int(verified.geometry.config), verified.num_inliers))
    assert len(answers) == 1, answers


# --- COLMAP's write rules (WriteVerifiedTwoViewGeometry) ------------------------------------------------------------


def test_a_degenerate_result_is_stored_as_an_empty_degenerate_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A DEGENERATE estimate that still carries models and inliers is stored with neither."""
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(DEGENERATE, 50))
    _write(tmp_path / "d.db", [_verify(_item((1, 2)))], write_matches=False)
    with pycolmap.Database.open(tmp_path / "d.db") as database:
        assert database.exists_two_view_geometry(1, 2)
        stored: pycolmap.TwoViewGeometry = database.read_two_view_geometry(1, 2)
    assert int(stored.config) == DEGENERATE
    assert len(stored.inlier_matches) == 0
    assert stored.E is None and stored.F is None and stored.H is None and stored.cam2_from_cam1 is None


def test_an_undefined_result_is_not_stored_but_its_raw_matches_are(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(UNDEFINED, 0))
    verified: VerifiedPair = _verify(_item((1, 2), num_matches=40))
    assert verified.geometry is None and verified.num_inliers == 0
    _write(tmp_path / "u.db", [verified], write_matches=True)
    assert _geometry_rows(tmp_path / "u.db") == {}
    assert _match_rows(tmp_path / "u.db") == {(1, 2): 40}


@pytest.mark.parametrize(("num_inliers", "stored"), [(15, (CALIBRATED, 15)), (14, (DEGENERATE, 0))])
def test_min_num_inliers_is_the_exact_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, num_inliers: int, stored: tuple[int, int]) -> None:
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(CALIBRATED, num_inliers))
    options: pycolmap.TwoViewGeometryOptions = pycolmap.TwoViewGeometryOptions()
    options.min_num_inliers = 15
    _write(tmp_path / "b.db", [_verify(_item((1, 2)), options)], write_matches=False)
    assert _geometry_rows(tmp_path / "b.db") == {(1, 2): stored}


def test_the_cap_thins_verified_inliers_and_keeps_the_raw_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The spatial cap runs after verification: it removes inliers, keeps the configuration, and leaves the raw matches."""
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(CALIBRATED, 100))
    seen: list[int] = []

    def keep_ten(pair: ImagePair, keypoints1: KeypointsXY, inliers: MatchIndices) -> MatchIndices:
        seen.append(len(inliers))
        return inliers[:10]

    verified: VerifiedPair = _verify(_item((1, 2), num_matches=200), cap=keep_ten)
    assert seen == [100]
    assert (verified.num_inliers, verified.num_capped, len(verified.matches)) == (10, 90, 200)
    stats: dict[ImagePair, PairMatchStats] = {}
    with pycolmap.Database.open(tmp_path / "c.db") as database:
        assert write_and_count(database, [verified], stats, write_matches=True) == 90
    assert stats == {(1, 2): PairMatchStats(raw_matches=200, inlier_matches=10)}
    assert _geometry_rows(tmp_path / "c.db") == {(1, 2): (CALIBRATED, 10)}
    assert _match_rows(tmp_path / "c.db") == {(1, 2): 200}


def test_the_cap_never_sees_a_rejected_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(DEGENERATE, 100))

    def must_not_run(pair: ImagePair, keypoints1: KeypointsXY, inliers: MatchIndices) -> MatchIndices:
        raise AssertionError("capped a rejected pair")

    verified: VerifiedPair = _verify(_item((1, 2)), cap=must_not_run)
    assert (verified.num_inliers, verified.num_capped) == (0, 0)


# --- all-or-nothing chunk writes ------------------------------------------------------------------------------------


def test_a_write_failing_partway_through_a_chunk_rolls_the_whole_chunk_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chunk 1 commits. Chunk 2 fails on its second pair's geometry (a row is already there), after the whole first pair
    and the second pair's matches were written. Reopened, the database holds chunk 1 and nothing of chunk 2."""
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(CALIBRATED, 30))
    path: Path = tmp_path / "r.db"
    _write(path, [_verify(_item((1, 2)))], write_matches=True)
    with pycolmap.Database.open(path) as database:
        existing: pycolmap.TwoViewGeometry = pycolmap.TwoViewGeometry()
        existing.config = DEGENERATE
        database.write_two_view_geometry(3, 4, existing)
    with pytest.raises(Exception, match="already exists"):
        _write(path, [_verify(_item((2, 3))), _verify(_item((3, 4)))], write_matches=True)
    assert _match_rows(path) == {(1, 2): 200}
    assert _geometry_rows(path) == {(1, 2): (CALIBRATED, 30), (3, 4): (DEGENERATE, 0)}


def test_a_failure_between_a_pairs_two_writes_leaves_neither(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An interrupt after `write_matches` and before `write_two_view_geometry` rolls the matches back too."""
    monkeypatch.setattr(pycolmap, "estimate_two_view_geometry", _estimate_returning(CALIBRATED, 30))
    chunk: list[VerifiedPair] = [_verify(_item((1, 2))), _verify(_item((2, 3)))]

    def interrupted(*_: object) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(pycolmap.Database, "write_two_view_geometry", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _write(tmp_path / "i.db", chunk, write_matches=True)
    assert _match_rows(tmp_path / "i.db") == {}
    assert _geometry_rows(tmp_path / "i.db") == {}


# --- the in-flight limit, order, failures ---------------------------------------------------------------------------


class _Workers:
    """A fake `verify_pair` whose calls block until released, recording which pairs started and finished."""

    def __init__(self, *, fail_on: int | None = None, delay_of: dict[int, float] | None = None) -> None:
        self.release: threading.Event = threading.Event()
        self.started: list[int] = []
        self.finished: list[int] = []
        self._lock: threading.Lock = threading.Lock()
        self._fail_on: int | None = fail_on
        self._delay_of: dict[int, float] = delay_of or {}

    def __call__(self, item: PairToVerify, *_: object) -> VerifiedPair:
        index: int = item.pair[0]
        with self._lock:
            self.started.append(index)
        if index == self._fail_on:
            raise RuntimeError(f"worker failed on pair {index}")
        self.release.wait(timeout=10.0)
        time.sleep(self._delay_of.get(index, 0.0))
        with self._lock:
            self.finished.append(index)
        return VerifiedPair(pair=item.pair, matches=item.matches, geometry=None, num_capped=0)


def _cameras(n: int) -> dict[int, pycolmap.Camera]:
    return {k: pycolmap.Camera() for k in (*range(n), *range(1000, 1000 + n))}


def _chunks(sizes: list[int], pulled: list[int] | None = None) -> Iterator[list[PairToVerify]]:
    """Chunks of pairs `(k, k + 1000)`, numbered in order; `pulled` records the first index of each chunk built."""
    first: int = 0
    for size in sizes:
        if pulled is not None:
            pulled.append(first)
        yield [_item((k, k + 1000), num_matches=1) for k in range(first, first + size)]
        first += size


def _wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline: float = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_no_chunk_is_admitted_while_the_window_is_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """K=2 with idle workers to spare: while both admitted chunks are blocked, a third is built but never started."""
    workers: _Workers = _Workers()
    monkeypatch.setattr(verification, "verify_pair", workers)
    pulled: list[int] = []
    delivered: list[int] = []
    chunking: VerifyChunking = VerifyChunking(verify_threads=4, verify_chunks_in_flight=2)
    with ChunkedVerifier(_cameras(5), pycolmap.TwoViewGeometryOptions(), chunking=chunking) as verifier:

        def consume() -> None:
            delivered.extend(verified.pair[0] for chunk in verifier.verified(_chunks([1] * 5, pulled)) for verified in chunk)

        consumer: threading.Thread = threading.Thread(target=consume)
        consumer.start()
        _wait_for(lambda: len(workers.started) == 2)
        time.sleep(0.3)
        assert workers.started == [0, 1], "a full window admits nothing more"
        assert pulled == [0, 1, 2], "the producer's current chunk is built, and waits"
        assert delivered == []
        workers.release.set()
        consumer.join(timeout=10.0)
    assert delivered == [0, 1, 2, 3, 4]


def test_results_come_once_and_in_order_however_the_workers_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """Later pairs finish first and the last chunk is partial: every pair still comes back once, in submission order."""
    workers: _Workers = _Workers(delay_of={k: 0.02 * (10 - k) for k in range(10)})
    workers.release.set()
    monkeypatch.setattr(verification, "verify_pair", workers)
    chunking: VerifyChunking = VerifyChunking(verify_threads=4, verify_chunks_in_flight=2)
    with ChunkedVerifier(_cameras(10), pycolmap.TwoViewGeometryOptions(), chunking=chunking) as verifier:
        delivered: list[list[int]] = [[verified.pair[0] for verified in chunk] for chunk in verifier.verified(_chunks([3, 3, 3, 1]))]
    assert delivered == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert sorted(workers.finished) == list(range(10))


def test_a_worker_failure_reaches_the_caller_and_cancels_the_queued_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """One worker: pair 0 fails; pairs 1-3 are queued behind it. The error reaches the caller and the queued work is
    cancelled: pair 1 either never starts or (if the worker took it first) runs and is waited for, nothing behind it
    starts, and the chunk that was never admitted is never built into work."""
    workers: _Workers = _Workers(fail_on=0, delay_of={1: 0.2})
    workers.release.set()
    monkeypatch.setattr(verification, "verify_pair", workers)
    chunking: VerifyChunking = VerifyChunking(verify_threads=1, verify_chunks_in_flight=2)
    with pytest.raises(RuntimeError, match="pair 0"), ChunkedVerifier(_cameras(6), pycolmap.TwoViewGeometryOptions(), chunking=chunking) as verifier:
        for _ in verifier.verified(_chunks([2, 2, 2])):
            pass
    assert workers.started in ([0], [0, 1]), "pairs 2 and 3 were cancelled in the queue; 4 and 5 were never admitted"
    assert workers.finished == workers.started[1:], "a pair that started was joined before the verifier closed"


def test_a_producer_failure_reaches_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    workers: _Workers = _Workers()
    workers.release.set()
    monkeypatch.setattr(verification, "verify_pair", workers)

    def failing() -> Iterator[list[PairToVerify]]:
        yield [_item((0, 1000), num_matches=1)]
        raise OSError("frame decode failed")

    with ChunkedVerifier(_cameras(1), pycolmap.TwoViewGeometryOptions()) as verifier, pytest.raises(OSError, match="decode"):
        for _ in verifier.verified(failing()):
            pass


@pytest.mark.parametrize("field", ["verify_chunk_pairs", "verify_threads", "verify_chunks_in_flight"])
def test_a_chunk_pool_or_window_of_zero_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        VerifyChunking(**{field: 0})
