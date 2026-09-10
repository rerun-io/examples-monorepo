"""What a fleet row says, and when it says the machine failed (D60).

The tool that produces these rows runs on machines with no NAS, no repository
and — on the pack target — no pixi, so what is under test here is the part that
needs none of that: the verdict a clip's numbers earn, and how the row reads.
"""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from fixture_types import never
from jaxtyping import Float64, Int64
from numpy import ndarray

from slam_rs import _core
from slam_rs.apis import fleet_check
from slam_rs.apis.fleet_check import CLIP_JSON_KEYS, ClipResult, Config, clip_json, main, measure, this_lane
from slam_rs.machine import Machine
from slam_rs.reference import (
    MANIFEST_PATH,
    MIN_TRACKED_POSES,
    PATH_BOUND_MAX_CLIP_S,
    SMOKE_SEGMENTS,
    ReferenceManifest,
    ReferenceSegment,
    pose_floor_text,
)
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import ASSOCIATION_TOLERANCE_NS, Trajectory, empty_trajectory, shift_clock, write_trajectory

CLOCK_GAP_NS: int = 10_433_867_587_166
"""What the Index smoke segment's two clocks are apart: ``video_time`` zero against the device clock every basalt CSV uses.

The gap a machine lands on when it exports the wrong one of the two
(:func:`slam_rs.trajectory.shift_clock`), and the one this suite uses to make an
estimate that associates with nothing.
"""
MACHINE: Machine = Machine(hostname="pablo-rpi", arch="aarch64", libc="2.36", cores=4)
DIGEST: str = hashlib.sha256(b"the resolved config text").hexdigest()
"""What a run reports as the digest of the config text its estimator was built from."""
"""A four-core Pi, which is the smallest machine that runs a full install."""
PASSING: ClipResult = ClipResult(
    segment_id=SMOKE_SEGMENTS[1],
    framesets=412,
    tracked=412,
    lost=0,
    cpp_rmse_cm=0.31,
    gt_rmse_cm=1.50,
    cpp_gt_band_cm=(1.427751, 1.427823),
    wall_s=30.0,
    cpp_wall_s=7.674,
    peak_rss_mb=512.0,
    gate_policy="tight",
    replayed_s=7.6,
    cpp_associated=412,
    extent_m=3.4,
    truth_extent_m=3.4,
    poses_finite=True,
    unscored=None,
    config_sha256=DIGEST,
)
"""The smoke clip as this host measures it, on a machine four times slower."""


def test_a_clip_inside_the_bands_passes_wherever_it_ran() -> None:
    """The three D60 clauses, and the speed ratio that is not one of them.

    The Pi is four times slower than the x86-64 host the C++ wall was measured
    on. That is a fact about the machine, not a failure of the port, so the ratio
    is reported and the verdict does not read it — the gate's speed clause is
    against a wall measured on one host and means nothing on another.
    """
    assert PASSING.failures == ()
    assert PASSING.verdict == "pass"
    assert PASSING.cpp_wall_ratio == 30.0 / 7.674


def test_every_clause_a_machine_can_miss_is_named_in_its_own_row() -> None:
    """A lost frameset, a path error and a ground-truth error outside the band."""
    lost: ClipResult = replace(PASSING, tracked=410, lost=2)
    assert lost.verdict.startswith("fail")
    assert "2 of 412" in lost.failures[0]

    adrift: ClipResult = replace(PASSING, cpp_rmse_cm=2.5, gt_rmse_cm=9.0)
    assert len(adrift.failures) == 2
    assert "2.50 cm from the C++ trajectory" in adrift.failures[0]
    assert "9.00 cm from ground truth" in adrift.failures[1]
    # 1.2 x the worse of the C++'s own two precisions on this clip.
    assert adrift.gt_allowed_cm == 1.2 * 1.427823


def test_the_row_carries_the_machine_beside_the_numbers() -> None:
    """One markdown row: the machine, the clip, both errors, the verdict, the cost."""
    cells: list[str] = [cell.strip() for cell in PASSING.row(MACHINE).strip().strip("|").split("|")]
    assert cells[:4] == ["pablo-rpi", "aarch64", "2.36", "4"]
    assert cells[4].endswith("MIO10_short_2_panorama")
    assert cells[5] == "412/412/0"
    assert cells[6] == "0.31"
    assert cells[7] == "1.50 (allowed 1.71)"
    assert cells[8] == "pass"
    assert cells[9] == "30.00"
    assert cells[10] == "3.91x"
    assert cells[11] == "512"


def test_a_clip_longer_than_the_path_bound_allows_is_not_held_to_two_centimetres() -> None:
    """D60 clause 4: past about a hundred seconds the C++ misses 2 cm against its own other precision.

    ``MIO14_moving_props`` is 410 s and the C++'s two precisions are 4.24 cm
    apart on it, so a 5.5 cm path error there is the clip's length, not the
    port. The gate has always guarded the bound this way
    (:data:`~slam_rs.reference.PATH_BOUND_MAX_CLIP_S`); a fleet row measured by
    ``--segments`` had not.
    """
    long_clip: ClipResult = replace(PASSING, replayed_s=410.5, cpp_rmse_cm=5.51, gt_rmse_cm=8.73, cpp_gt_band_cm=(8.86, 6.56))
    assert long_clip.replayed_s > PATH_BOUND_MAX_CLIP_S
    assert long_clip.verdict == "pass"
    short_clip: ClipResult = replace(long_clip, replayed_s=99.0)
    assert "5.51 cm from the C++ trajectory" in short_clip.failures[0]


def test_a_no_divergence_clip_gates_a_bounded_path_and_nothing_else() -> None:
    """D60 clause 5: on ``MGO01_low_light`` the C++ binary itself gets 43 cm, so a tolerance measures noise.

    The clip has to keep tracking and stay bounded. Reporting it 42 cm from
    ground truth as a failure calls a working port broken, which is what a fleet
    row run with ``--segments`` used to do.
    """
    low_light: ClipResult = replace(PASSING, gate_policy="no_divergence", cpp_rmse_cm=68.0, gt_rmse_cm=42.75, extent_m=3.4, truth_extent_m=3.4)
    assert low_light.verdict == "pass"
    adrift: ClipResult = replace(low_light, extent_m=340.0)
    assert "spans 340.0 m against the truth's 3.4 m" in adrift.failures[0]
    infinite: ClipResult = replace(low_light, poses_finite=False)
    assert infinite.failures == ("a pose is not finite",)


def test_a_handful_of_associated_poses_is_not_a_comparison() -> None:
    """``ate`` returns a number for two poses on purpose; the floor is what makes it a verdict."""
    thin: ClipResult = replace(PASSING, cpp_associated=3)
    assert "only 3 poses associated with the C++ run" in thin.failures[0]


def test_a_run_below_the_pose_floor_is_not_a_trajectory_at_all() -> None:
    """D60's first clause, which the fleet row did not have: three poses is not a trajectory.

    The gate has always refused a run this short in these words, and a fleet row
    used to reach the association floor instead and call it "only 3 poses
    associated with the C++ run" — a verdict about the comparison, where the
    finding is about the run. Below the floor nothing was scored, so the two
    errors are not numbers.
    """
    dead: ClipResult = replace(PASSING, tracked=3, cpp_associated=3, cpp_rmse_cm=math.nan, gt_rmse_cm=math.nan)
    assert dead.failures == (pose_floor_text(tracked=3, framesets=412),)
    assert dead.verdict == "fail: 3 poses over 412 framesets is not a trajectory"


def test_a_clip_the_estimator_never_tracked_is_a_row_and_not_a_traceback(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A machine where the port produces nothing has to report that, which is the whole purpose of the tool.

    ``ate`` refuses a run with no pose — there is nothing to align — so a row
    built by calling it first was a traceback on exactly the machine the fleet
    lane exists to find. The verdict is D60's pose floor, the two errors read as
    NaN, the JSON keeps its keys, and the run exits non-zero.
    """

    monkeypatch.setattr(fleet_check, "ate", never("`ate` was called on a run below D60's pose floor"))
    monkeypatch.setattr(
        fleet_check,
        "run_segment",
        lambda *_args, **_kwargs: SegmentRun(estimate=empty_trajectory(), framesets=412, lost=412, wall_s=1.0, config_sha256=DIGEST),
    )
    # Neither reference is opened for its numbers here, and this keeps the case
    # runnable on a machine with no corpus at all.
    monkeypatch.setattr(fleet_check, "read_trajectory", lambda _path: empty_trajectory())
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])

    dead: ClipResult = measure(manifest, segment)
    assert dead.tracked == 0
    assert dead.failures == (pose_floor_text(tracked=0, framesets=412),)
    assert math.isnan(dead.cpp_rmse_cm)
    assert math.isnan(dead.gt_rmse_cm)
    assert dead.cpp_associated == 0

    output: Path = tmp_path / "fleet_check.json"
    with pytest.raises(SystemExit, match="is not a trajectory"):
        main(Config(segments=(SMOKE_SEGMENTS[1],), output_json=output))
    written: dict = json.loads(output.read_text())
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)
    assert written["clips"][0]["verdict"].startswith("fail:")
    # NaN is what the chart's own `f"{value:.2f}"` reads; None is what it cannot.
    assert math.isnan(cast("float", clip_json(dead)["cpp_rmse_cm"]))


def test_an_estimate_on_another_clock_is_a_row_and_not_a_traceback(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ten poses clears D60's floor and can still be scored against nothing.

    The floor was the only unscored case a row knew, and it is not the only one
    there is: :func:`~slam_rs.trajectory.ate` needs an association, not a pose
    count, so an estimate on the wrong clock — the Index segment's two are
    :data:`CLOCK_GAP_NS` apart — tracked well enough to be scored and reached a
    traceback instead of a row. The clause names what ``ate`` refused and the
    tolerance it refused it at, both errors read as NaN exactly as they do below
    the floor, the JSON keeps its keys, and the run exits non-zero.
    """
    poses: int = MIN_TRACKED_POSES
    # Both references are on the device clock, so one shifted estimate misses
    # both of them; positions are never reached, because nothing associates.
    reference: Trajectory = Trajectory(
        t_ns=np.arange(poses, dtype=np.int64) * 20_000_000, position_m=np.zeros((poses, 3)), quaternion_wxyz=np.zeros((poses, 4))
    )
    monkeypatch.setattr(fleet_check, "read_trajectory", lambda _path: reference)
    monkeypatch.setattr(
        fleet_check,
        "run_segment",
        lambda *_args, **_kwargs: SegmentRun(
            estimate=shift_clock(reference, CLOCK_GAP_NS), framesets=poses, lost=0, wall_s=1.0, config_sha256=DIGEST
        ),
    )
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])

    adrift: ClipResult = measure(manifest, segment)
    assert adrift.tracked == poses
    assert len(adrift.failures) == 1
    assert f"no pose associated within {ASSOCIATION_TOLERANCE_NS} ns" in adrift.failures[0]
    assert math.isnan(adrift.cpp_rmse_cm)
    assert math.isnan(adrift.gt_rmse_cm)
    assert adrift.cpp_associated == 0

    output: Path = tmp_path / "fleet_check.json"
    with pytest.raises(SystemExit, match="no pose associated within"):
        main(Config(segments=(SMOKE_SEGMENTS[1],), output_json=output))
    written: dict = json.loads(output.read_text())
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)
    assert "no pose associated within" in written["clips"][0]["verdict"]
    assert math.isnan(written["clips"][0]["gt_rmse_cm"])


def test_a_non_finite_estimate_is_a_row_and_not_an_alignment_traceback(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diverged estimator is what a fleet lane exists to find, and it used to be a traceback.

    An estimate whose positions carry a NaN clears D60's pose floor and
    associates on the references' clock, so a row was built by aligning it — and
    :func:`~slam_rs.trajectory.rigid_alignment` hands the covariance to
    ``np.linalg.svd``, which raises ``LinAlgError: SVD did not converge``. That
    is not the :class:`ValueError` the association case is caught as, so the clip
    left no row and the JSON the chart reads was never written: the machine that
    diverged is the one machine this tool reported nothing about. Finiteness is
    tested before the alignment, the clause names it, both errors read as NaN,
    the JSON keeps its keys, and the run exits non-zero.
    """
    poses: int = MIN_TRACKED_POSES
    t_ns: Int64[ndarray, " n"] = np.arange(poses, dtype=np.int64) * 20_000_000
    reference: Trajectory = Trajectory(
        t_ns=t_ns, position_m=np.arange(3 * poses, dtype=np.float64).reshape(poses, 3), quaternion_wxyz=np.zeros((poses, 4))
    )
    positions: Float64[ndarray, "n 3"] = np.arange(3 * poses, dtype=np.float64).reshape(poses, 3).copy()
    positions[4, 1] = np.nan
    monkeypatch.setattr(fleet_check, "read_trajectory", lambda _path: reference)
    monkeypatch.setattr(
        fleet_check,
        "run_segment",
        lambda *_args, **_kwargs: SegmentRun(
            estimate=Trajectory(t_ns=t_ns, position_m=positions, quaternion_wxyz=np.zeros((poses, 4))),
            framesets=poses,
            lost=0,
            wall_s=1.0,
            config_sha256=DIGEST,
        ),
    )
    monkeypatch.setattr(fleet_check, "ate", never("an estimate with a non-finite position was handed to the alignment"))
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])

    diverged: ClipResult = measure(manifest, segment)
    assert diverged.tracked == poses
    assert diverged.poses_finite is False
    assert diverged.unscored is not None
    assert "1 of 10 estimated positions is not finite" in diverged.unscored
    assert diverged.failures == (diverged.unscored,)
    assert diverged.verdict.startswith("fail: ")
    assert math.isnan(diverged.cpp_rmse_cm)
    assert math.isnan(diverged.gt_rmse_cm)
    assert diverged.cpp_associated == 0

    output: Path = tmp_path / "fleet_check.json"
    with pytest.raises(SystemExit, match="is not finite"):
        main(Config(segments=(SMOKE_SEGMENTS[1],), output_json=output))
    written: dict = json.loads(output.read_text())
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)
    assert "is not finite" in written["clips"][0]["verdict"]
    assert math.isnan(written["clips"][0]["gt_rmse_cm"])


def test_a_reference_trajectory_that_is_not_here_is_refused_before_the_replay(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manifest says why it cannot be read; paying a 410 s replay to reach a bare ``FileNotFoundError`` does not.

    The two long-tier segments keep their trajectory in the machine-local
    reference bundle, and the pack carries two of ten, so a fleet machine meets
    this whenever it names a clip whose reference did not ship.
    """
    monkeypatch.setenv("SLAM_RS_REFERENCE_DIR", str(tmp_path))

    monkeypatch.setattr(fleet_check, "run_segment", never("the replay was paid for before the reference was checked"))
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])
    absent: ReferenceSegment = replace(segment, reference=replace(segment.reference, bundle_only=True))
    with pytest.raises(FileNotFoundError, match="is not in SLAM_RS_REFERENCE_DIR"):
        measure(manifest, absent)


def test_an_unknown_segment_id_is_refused_before_the_first_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--segments <410 s clip> typo`` used to pay the clip and then traceback on the typo.

    Every id the run was given is resolved before the loop opens anything, and
    the manifest's own selector error names the id and the ten it has.
    """

    monkeypatch.setattr(fleet_check, "measure", never("a clip was measured before every id was resolved"))
    with pytest.raises(ValueError, match="MIO10_typo.*MIO10_short_2_panorama"):
        main(Config(segments=(SMOKE_SEGMENTS[1], "MIO10_typo"), output_json=tmp_path / "fleet_check.json"))


def test_a_partial_corpus_is_refused_before_the_first_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The second clip's missing sidecar must not cost the first clip's replay.

    A relocated corpus can be partial — the pack carries two of the ten — and the
    ``gt.csv`` used to be opened only after ``run_segment`` had returned, so a
    pack that unpacked half way spent a whole replay to find out. Every clip's
    scoring inputs are opened before the loop starts; the first clip here has its
    sidecar and the second does not, and nothing is replayed.
    """

    monkeypatch.setattr(fleet_check, "run_segment", never("a replay was paid for before every scoring input was opened"))
    write_trajectory(tmp_path / SMOKE_SEGMENTS[0] / "gt.csv", empty_trajectory())
    with pytest.raises(FileNotFoundError, match=f"{SMOKE_SEGMENTS[1]}.*is not a file on this machine"):
        main(Config(artifact_root=tmp_path, segments=SMOKE_SEGMENTS, output_json=tmp_path / "fleet_check.json"))


def test_the_first_clips_evidence_survives_a_directory_that_is_not_there_yet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The JSON is the whole point of the run, and it is written after every clip.

    On a 2 GB device the second clip is what the kernel may refuse, so the first
    clip's row has to be on disk already — and it is not, if the last line of the
    run is what discovers that ``out/`` does not exist.
    """
    monkeypatch.setattr(fleet_check, "measure", lambda _manifest, _segment, _gpu, *, profile: PASSING)
    output: Path = tmp_path / "out" / "fleet_check.json"
    main(Config(segments=(SMOKE_SEGMENTS[1],), output_json=output))
    written: dict = json.loads(output.read_text())
    assert list(written) == ["machine", "lane", "profile", "config_sha256", "clips"]
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)


def test_the_lane_is_on_the_json_and_the_clip_columns_are_not(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A GPU row and a CPU row differ in the run, not in the clip's columns.

    The chart reads :data:`~slam_rs.apis.fleet_check.CLIP_JSON_KEYS` as a
    contract, so the lane cannot be a thirteenth column of it; it is one key
    beside ``machine``.
    """
    monkeypatch.setattr(_core, "gpu_backend", "wgpu")
    monkeypatch.setattr(fleet_check, "measure", lambda _manifest, _segment, _gpu, *, profile: PASSING)
    output: Path = tmp_path / "fleet_check.json"
    main(Config(segments=(SMOKE_SEGMENTS[1],), output_json=output, gpu=True))
    written: dict = json.loads(output.read_text())
    assert written["lane"] == "wgpu"
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)


def test_the_lane_names_the_gpu_runtime_this_core_was_built_with() -> None:
    """Fleet rows name the runtime reported by the extension."""
    assert _core.gpu_backend in (None, "wgpu")
    assert this_lane(gpu=False) == "cpu"


def test_wgpu_build_reports_its_backend_as_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lane name comes from the build, not the flag."""
    assert this_lane(gpu=False) == "cpu"
    monkeypatch.setattr(_core, "gpu_backend", "wgpu")
    assert this_lane(gpu=True) == "wgpu"


def test_a_gpu_run_on_a_core_without_a_gpu_feature_is_refused_before_any_file_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A CPU-only core has no GPU lane to name, and the refusal is worth nothing after a replay.

    :class:`slam_rs._core.Vio` refuses ``gpu=True`` on such a core anyway; asking
    the extension which runtime it carries is what lets the row say so before the
    manifest is even read.
    """

    monkeypatch.setattr(_core, "gpu_backend", None)
    monkeypatch.setattr(fleet_check, "measure", never("a clip was measured on a core that has no GPU lane"))
    output: Path = tmp_path / "fleet_check.json"
    with pytest.raises(ValueError, match="built with a GPU cargo feature"):
        main(Config(segments=(SMOKE_SEGMENTS[1],), output_json=output, gpu=True))
    assert not output.exists()


def test_an_empty_segment_selection_is_refused_rather_than_read_as_a_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--segments`` with nothing in it validated nothing, wrote no JSON and exited 0.

    Which is the worst shape a fleet gate can take: the exit status a script reads
    said the machine had passed, and the evidence file the chart reads was not
    even created (S24 review).
    """

    monkeypatch.setattr(fleet_check, "load_manifest", never("a run with no clip selected reached the manifest"))
    output: Path = tmp_path / "fleet_check.json"
    with pytest.raises(ValueError, match="--segments named no clip"):
        main(Config(segments=(), output_json=output))
    assert not output.exists()


def test_the_gpu_flag_reaches_the_estimator_and_nothing_else_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--gpu`` is only worth a row if it arrives at the run that produced it.

    The flag crosses two hops — the config to :func:`~slam_rs.apis.fleet_check.measure`,
    then ``measure`` to :func:`~slam_rs.tracking.run_segment` — and a lost hop
    would label a CPU row ``gpu`` with nothing to notice, which is the one
    failure this whole tool exists to rule out.
    """
    seen: list[bool] = []

    def record(_manifest: ReferenceManifest, _segment: ReferenceSegment, *, gpu: bool, profile: str) -> SegmentRun:
        assert profile == "reference"
        seen.append(gpu)
        return SegmentRun(estimate=empty_trajectory(), framesets=412, lost=412, wall_s=1.0, config_sha256=DIGEST)

    monkeypatch.setattr(fleet_check, "run_segment", record)
    manifest: ReferenceManifest = fleet_check.load_manifest(MANIFEST_PATH)
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])
    measure(manifest, segment, True)
    measure(manifest, segment)
    assert seen == [True, False]


@pytest.mark.parametrize("profile", ["reference", "fast"])
def test_run_provenance_is_the_runs_own_digest_outside_clip_columns(
    profile: Literal["reference", "fast"],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exported digest is the one each run's estimator was built from, once per dataset, beside the profile.

    ``run_segment`` is the only place that knows what text the estimator parsed,
    so the digest has to come out of it; a tool that resolved the file again
    could name a config the estimator never read.
    """
    requested: str = profile
    digests: dict[str, str] = {}

    def record(_manifest: ReferenceManifest, segment: ReferenceSegment, *, gpu: bool, profile: str) -> SegmentRun:
        assert profile == requested
        digest: str = hashlib.sha256(f"{segment.dataset_name}:{profile}".encode()).hexdigest()
        digests[segment.dataset_name] = digest
        return SegmentRun(estimate=empty_trajectory(), framesets=412, lost=412, wall_s=1.0, config_sha256=digest)

    monkeypatch.setattr(fleet_check, "run_segment", record)
    output: Path = tmp_path / "fleet.json"
    with pytest.raises(SystemExit):
        main(Config(segments=SMOKE_SEGMENTS, profile=profile, output_json=output))
    written: dict = json.loads(output.read_text())
    assert written["profile"] == profile
    assert written["config_sha256"] == digests
    assert all(list(clip) == list(CLIP_JSON_KEYS) for clip in written["clips"])
    printed: str = capsys.readouterr().out
    assert f"profile={profile}" in printed
    for digest in digests.values():
        assert printed.count(digest) == 1


def test_a_config_that_changes_during_a_run_stops_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two clips of one dataset built from different texts cannot share one digest in the JSON."""
    seen: list[str] = []

    def record(_manifest: ReferenceManifest, _segment: ReferenceSegment, *, gpu: bool, profile: str) -> SegmentRun:
        seen.append(profile)
        return SegmentRun(
            estimate=empty_trajectory(), framesets=412, lost=412, wall_s=1.0, config_sha256=hashlib.sha256(str(len(seen)).encode()).hexdigest()
        )

    monkeypatch.setattr(fleet_check, "run_segment", record)
    with pytest.raises(RuntimeError, match="the config changed during the run"):
        main(Config(segments=(SMOKE_SEGMENTS[1], SMOKE_SEGMENTS[1]), output_json=tmp_path / "fleet.json"))
    # The first clip's JSON stands; the conflicting second clip was never written under the first digest.
    assert len(json.loads((tmp_path / "fleet.json").read_text())["clips"]) == 1
