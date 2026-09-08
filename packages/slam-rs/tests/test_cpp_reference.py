"""The basalt C++ reference runs: the numbers this port is measured against.

Two things are pinned here. First, that ``slam_rs.trajectory`` reproduces the
fork's own ATE arithmetic on the fork's own outputs — if it does not, every gate
number in the manifest means something different from what the C++ side measured.
Second, that the feed decodes the same pixels the C++ run consumed, frame for
frame, which is what makes an A/B between the two estimators about the estimator
rather than about the decoder.

The association is driven by the estimate, matching the reference: each basalt
pose takes the nearest ground-truth pose within 5 ms.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from slam_rs import reference_bundle
from slam_rs.catalog_feed import Frameset, LocalSegment, open_segment
from slam_rs.reference import SMOKE_SEGMENTS, CppAte, ReferenceManifest, ReferenceSegment
from slam_rs.trajectory import AteResult, Trajectory, ate, read_trajectory

RMSE_TOLERANCE_CM: float = 0.005
"""Rounding slack when reproducing a published centimetre figure."""
WALL_TOLERANCE_S: float = 0.001
"""Rounding slack on the manifest's copy of the run's wall time, which is recorded to the millisecond."""


def _assert_reproduces(result: AteResult, expected: CppAte, segment_id: str) -> None:
    """The recomputed error must be the published one, to the precision it was published at."""
    assert result.n_associated == expected.associated, segment_id
    assert result.n_estimate == expected.total, segment_id
    assert result.rmse_m * 100 == pytest.approx(expected.rmse_cm, abs=RMSE_TOLERANCE_CM), segment_id
    assert result.max_m * 100 == pytest.approx(expected.max_cm, abs=RMSE_TOLERANCE_CM), segment_id
    assert result.median_m * 100 == pytest.approx(expected.median_cm, abs=RMSE_TOLERANCE_CM), segment_id


def test_every_segment_has_a_reference_run(manifest: ReferenceManifest) -> None:
    for segment in manifest.segments:
        run: Path = manifest.package_root / segment.reference.run_json
        assert run.is_file(), segment.segment_id
        # The manifest's own claims must match the run it points at.
        recorded: dict[str, Any] = json.loads(run.read_text())
        assert recorded["segment"] == segment.segment_id
        assert recorded["decode_path"] == segment.reference.decode_path == segment.decode_path
        assert recorded["fork"]["commit"] == segment.reference.fork_commit
        assert recorded["capture"]["start_time_ns"] == segment.capture.start_time_ns
        assert recorded["capture"]["num_frames"] == segment.capture.num_frames
        assert recorded["run"]["poses"] == segment.reference.expected_cpp_ate.total
        assert recorded["run"]["status"] == "ok"
        # The wall the speed clause is measured against is this run's own.
        assert recorded["run"]["feed_wall_time_s"] == pytest.approx(segment.reference.expected_cpp_wall_s, abs=WALL_TOLERANCE_S)
        assert recorded["outputs"]["trajectory_sha256"] == segment.reference.trajectory_sha256
        # The IMU model the C++ run was tuned with is the one the manifest freezes.
        assert recorded["imu"]["gyro_noise_std"] == segment.imu.gyro_noise_std
        assert recorded["imu"]["accel_noise_std"] == segment.imu.accel_noise_std
        assert recorded["imu"]["cam_time_offset_ns"] == segment.imu.cam_time_offset_ns
        assert recorded["deterministic"]["deterministic"] is segment.reference.deterministic
        assert recorded["deterministic"]["num_threads"] == segment.reference.num_threads
        assert recorded["deterministic"]["use_double"] is segment.reference.use_double


def test_the_vendored_configs_are_the_ones_the_cpp_runs_used(manifest: ReferenceManifest) -> None:
    """Every ``config.*`` key of every C++ run, against the file the port loads.

    The run manifests embed the whole configuration document the C++ binary read,
    so this compares documents rather than the four fields the run summarises and
    the one the binding exposes: a key the Rust struct does not model yet is
    compared here too. What it stops from coming back is C72 — the Python path
    built ``_core.VioConfig()``, basalt's constructor defaults, which differ from
    the shipped MSD files in ``vio_marg_lost_landmarks`` and put the port 1.41 to
    12.05 cm from the C++ instead of 0.31 to 5.19 cm.
    """
    compared: int = 0
    for segment in manifest.segments:
        vendored: dict[str, Any] = json.loads(manifest.vio_config_text(segment.dataset_name))["value0"]
        recorded: dict[str, Any] = json.loads((manifest.package_root / segment.reference.run_json).read_text())["vio_config"]
        # The run names the config by its path inside the fork; the manifest names
        # this repository's copy of that same file.
        assert Path(recorded["path"]) == Path(segment.reference.vio_config), segment.segment_id
        assert Path(recorded["path"]).name == manifest.dataset(segment.dataset_name).vio_config.name, segment.segment_id
        assert set(vendored) == set(recorded["json"]), segment.segment_id
        differing: dict[str, tuple[Any, Any]] = {
            key: (value, recorded["json"][key]) for key, value in vendored.items() if value != recorded["json"][key]
        }
        assert not differing, f"{segment.segment_id}: vendored config disagrees with the C++ run, (vendored, run) = {differing}"
        # Named rather than left implicit: this is the key the constructor's
        # defaults get wrong, and the reason this test exists.
        assert vendored["config.vio_marg_lost_landmarks"] is True, segment.segment_id
        assert vendored["config.optical_flow_image_safe_radius"] == segment.reference.optical_flow_image_safe_radius, segment.segment_id
        compared += len(vendored)
    # Ten runs, 68 keys each: the whole document, every time.
    assert compared == 10 * 68


def test_the_committed_run_manifests_carry_the_bundles_configuration(manifest: ReferenceManifest) -> None:
    """Where the reference bundle is on this host, its run manifests agree with the committed ones.

    The committed ``run.json`` copies are what
    :func:`test_the_vendored_configs_are_the_ones_the_cpp_runs_used` compares
    against, and they are editable text in this repository. The bundle holds the
    originals the C++ runs wrote, so this closes the loop for whichever segments
    the bundle root on this host holds — the two long-tier runs were re-run into a
    second root, so which segments resolve depends on where
    ``SLAM_RS_REFERENCE_DIR`` points. Only the configuration is compared: the
    long-tier copies carry fewer summary keys than the committed ones.
    """
    compared: list[str] = []
    for segment in manifest.segments:
        resolved: reference_bundle.BundleFile = reference_bundle.resolve(segment.segment_id, reference_bundle.RUN_JSON)
        if not resolved.available:
            continue
        original: dict[str, Any] = json.loads(resolved.path.read_text())["vio_config"]
        committed: dict[str, Any] = json.loads((manifest.package_root / segment.reference.run_json).read_text())["vio_config"]
        assert original["json"] == committed["json"], segment.segment_id
        assert original["path"] == committed["path"], segment.segment_id
        compared.append(segment.segment_id)
    if not compared:
        pytest.skip(f"no run manifest resolves through {reference_bundle.bundle_root()}")


def test_only_the_long_tier_is_bundle_only(manifest: ReferenceManifest) -> None:
    bundled: set[str] = {s.segment_id for s in manifest.segments if s.reference.bundle_only}
    assert bundled == {s.segment_id for s in manifest.in_tier("long")}
    for segment in manifest.segments:
        if segment.reference.bundle_only:
            # Committed instead: a README saying where the trajectory lives.
            assert ((manifest.package_root / segment.reference.run_json).parent / "README").is_file()
        else:
            assert (manifest.package_root / segment.reference.trajectory_csv).is_file()


def test_the_gate_policy_matches_what_basalt_can_actually_do(manifest: ReferenceManifest) -> None:
    """A tight gate is only allowed where the C++ reference is itself accurate."""
    policies: dict[str, str] = {s.segment_id: s.reference.gate_policy for s in manifest.segments}
    assert policies["msd-index__MIO_others__MIO10_short_2_panorama"] == "tight"
    assert policies["msd-g2__MGO_others__MGO09_short_1_updown"] == "tight"
    assert policies["msd-index__MIO_others__MIO07_mapping_easy"] == "tight"
    assert policies["msd-g2__MGO_others__MGO07_mapping_easy"] == "tight"
    # basalt is near failure on these two: 43 cm and 78 cm here, 68 cm for the C++
    # binary on the raw files, and 32 cm / 18 cm of spread between two legitimate
    # decode paths of the same estimator. A tolerance would measure noise.
    assert policies["msd-g2__MGO_others__MGO01_low_light"] == "no_divergence"
    assert policies["msd-g2__MGO_others__MGO13_sudden_movements"] == "no_divergence"

    for segment in manifest.segments:
        rmse_cm: float = segment.reference.expected_cpp_ate.rmse_cm
        if segment.reference.gate_policy == "tight":
            assert rmse_cm < 3.0, f"{segment.segment_id} is gated tight but basalt scores {rmse_cm:.2f} cm"
        if segment.reference.gate_policy == "no_divergence":
            assert rmse_cm > 30.0, f"{segment.segment_id} is only gated for divergence but basalt scores {rmse_cm:.2f} cm"


def test_the_reference_runs_tracked_every_frameset(manifest: ReferenceManifest) -> None:
    """One pose per frameset, and near-total ground-truth coverage."""
    for segment in manifest.segments:
        expected: CppAte = segment.reference.expected_cpp_ate
        assert expected.total == segment.capture.num_frames, segment.segment_id
        # MIO14's sidecar has real gaps: 148 of 22,117 poses have no truth within
        # 5 ms. Everything else is within one pose of complete.
        assert expected.associated / expected.total > 0.99, segment.segment_id


@pytest.mark.parametrize("segment_id", SMOKE_SEGMENTS)
def test_the_smoke_tier_numbers_reproduce_offline(manifest: ReferenceManifest, segment_id: str) -> None:
    """The published smoke figures, from committed files only: no NAS, no catalog."""
    segment: ReferenceSegment = manifest.by_id(segment_id)
    assert segment.reference.gt_csv_fixture is not None
    estimate: Trajectory = read_trajectory(manifest.package_root / segment.reference.trajectory_csv)
    truth: Trajectory = read_trajectory(manifest.package_root / segment.reference.gt_csv_fixture)
    assert len(truth) == segment.gt.num_poses
    _assert_reproduces(ate(estimate, truth), segment.reference.expected_cpp_ate, segment_id)


def test_the_committed_gt_fixtures_are_the_sidecars_the_runs_used(manifest: ReferenceManifest) -> None:
    """The committed ``gt.csv`` copies name the same NAS path the C++ runs read."""
    for segment_id in SMOKE_SEGMENTS:
        segment: ReferenceSegment = manifest.by_id(segment_id)
        recorded: dict[str, Any] = json.loads((manifest.package_root / segment.reference.run_json).read_text())
        assert Path(recorded["ate_vs_gt"]["gt_csv"]) == segment.gt_csv


@pytest.mark.parametrize("segment_id", SMOKE_SEGMENTS)
def test_the_committed_frame_digests_cover_every_camera_frame(manifest: ReferenceManifest, segment_id: str) -> None:
    """One digest per (frameset, camera), on the absolute clock, and nothing repeated."""
    segment: ReferenceSegment = manifest.by_id(segment_id)
    assert segment.reference.frames_sha256 is not None
    digests: dict[tuple[int, int], str] = _read_frame_digests(manifest.package_root / segment.reference.frames_sha256)
    assert len(digests) == segment.capture.num_frames * segment.capture.num_cameras
    timestamps: set[int] = {t_ns for t_ns, _ in digests}
    assert len(timestamps) == segment.capture.num_frames
    # Absolute device-clock timestamps, inside the capture window. Video does not
    # necessarily start at video_time zero: on MIO10 it does, on MGO09 the first
    # frame is 17.5 ms in.
    assert min(timestamps) >= segment.capture.start_time_ns
    assert max(timestamps) <= segment.capture.start_time_ns + segment.capture.duration_ns
    assert {camera for _, camera in digests} == set(range(segment.capture.num_cameras))
    assert all(len(digest) == 64 for digest in digests.values())
    recorded: dict[str, Any] = json.loads((manifest.package_root / segment.reference.run_json).read_text())
    assert recorded["outputs"]["frame_hashes"] == len(digests)


def _read_frame_digests(path: Path) -> dict[tuple[int, int], str]:
    """``t_ns,cam_index,sha256`` lines, keyed by (absolute timestamp, camera)."""
    digests: dict[tuple[int, int], str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        t_ns, camera, digest = line.split(",")
        digests[(int(t_ns), int(camera))] = digest
    return digests


def test_the_long_tier_trajectories_resolve_through_the_bundle(manifest: ReferenceManifest) -> None:
    """Present or not, the resolution has to name a path and a reason."""
    for segment in manifest.in_tier("long"):
        resolved = reference_bundle.resolve(segment.segment_id, reference_bundle.TRAJECTORY_CSV)
        assert resolved.path.name == "basalt_traj.csv"
        assert resolved.path.parent.name == segment.segment_id
        if not resolved.available:
            assert reference_bundle.BUNDLE_DIR_VARIABLE in (resolved.reason or "")


@pytest.mark.slow
@pytest.mark.parametrize("segment_id", SMOKE_SEGMENTS)
def test_a_redecode_reproduces_the_cpp_pixel_digests(manifest: ReferenceManifest, segment_id: str) -> None:
    """The feed hands the core byte-identical pixels to what the C++ reference consumed.

    This is the load-bearing claim behind every A/B between the two estimators: if
    the digests match, a trajectory difference is the estimator, not the decoder.
    """
    segment: ReferenceSegment = manifest.by_id(segment_id)
    if not segment.base_path.is_file():
        pytest.skip(f"{segment.base_path} is not mounted on this host")
    assert segment.reference.frames_sha256 is not None
    expected: dict[tuple[int, int], str] = _read_frame_digests(manifest.package_root / segment.reference.frames_sha256)

    compared: int = 0
    frameset: Frameset
    with open_segment(LocalSegment(base_rrd=segment.base_path), segment.imu) as feed:
        offset: int = feed.capture_start_time_ns
        for frameset in feed.framesets():
            for camera, digest in zip(feed.cameras, frameset.image_digests(), strict=True):
                key: tuple[int, int] = (frameset.t_ns + offset, camera.index)
                assert key in expected, f"{segment_id}: the reference has no frame at {key}"
                assert digest == expected[key], f"{segment_id}: pixels differ at {key}"
                compared += 1
    assert compared == len(expected)


@pytest.mark.slow
def test_every_committed_trajectory_reproduces_its_published_ate(manifest: ReferenceManifest) -> None:
    """All eight committed runs against the NAS sidecars, plus the long pair when bundled."""
    checked: list[str] = []
    for segment in manifest.segments:
        if segment.reference.bundle_only:
            resolved = reference_bundle.resolve(segment.segment_id, reference_bundle.TRAJECTORY_CSV)
            if not resolved.available:
                continue
            trajectory_path: Path = resolved.path
        else:
            trajectory_path = manifest.package_root / segment.reference.trajectory_csv
        if not segment.gt_csv.is_file():
            continue
        estimate: Trajectory = read_trajectory(trajectory_path)
        truth: Trajectory = read_trajectory(segment.gt_csv)
        _assert_reproduces(ate(estimate, truth), segment.reference.expected_cpp_ate, segment.segment_id)
        checked.append(segment.segment_id)
    if not checked:
        pytest.skip("no gt.csv sidecar is mounted on this host")
    # The eight committed ones at least; the long pair joins when the bundle is present.
    assert len(checked) >= 8


@pytest.mark.slow
def test_the_committed_trajectories_match_their_recorded_digest(manifest: ReferenceManifest) -> None:
    """A fixture that was re-saved or line-ending-mangled would silently change the gate."""
    for segment in manifest.segments:
        if segment.reference.bundle_only:
            continue
        path: Path = manifest.package_root / segment.reference.trajectory_csv
        assert hashlib.sha256(path.read_bytes()).hexdigest() == segment.reference.trajectory_sha256, segment.segment_id


@pytest.mark.slow
def test_the_committed_gt_fixture_matches_the_nas_sidecar(manifest: ReferenceManifest) -> None:
    """The offline smoke gate must be reading the same bytes the NAS holds."""
    for segment_id in SMOKE_SEGMENTS:
        segment: ReferenceSegment = manifest.by_id(segment_id)
        assert segment.reference.gt_csv_fixture is not None
        if not segment.gt_csv.is_file():
            pytest.skip(f"{segment.gt_csv} is not mounted on this host")
        committed: Trajectory = read_trajectory(manifest.package_root / segment.reference.gt_csv_fixture)
        on_nas: Trajectory = read_trajectory(segment.gt_csv)
        np.testing.assert_array_equal(committed.t_ns, on_nas.t_ns)
        np.testing.assert_array_equal(committed.position_m, on_nas.position_m)
