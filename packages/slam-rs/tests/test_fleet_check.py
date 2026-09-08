"""What a fleet row says, and when it says the machine failed (D60).

The tool that produces these rows runs on machines with no NAS, no repository
and — on the pack target — no pixi, so what is under test here is the part that
needs none of that: the verdict a clip's numbers earn, and how the row reads.
"""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from slam_rs.apis import fleet_check
from slam_rs.apis.fleet_check import ClipResult, Config, clip_json, main, measure
from slam_rs.machine import Machine
from slam_rs.reference import MANIFEST_PATH, PATH_BOUND_MAX_CLIP_S, SMOKE_SEGMENTS, ReferenceManifest, ReferenceSegment, pose_floor_text
from slam_rs.tracking import SegmentRun
from slam_rs.trajectory import empty_trajectory

CLIP_JSON_KEYS: tuple[str, ...] = (
    "segment_id",
    "framesets",
    "tracked",
    "lost",
    "cpp_rmse_cm",
    "gt_rmse_cm",
    "cpp_gt_band_cm",
    "wall_s",
    "cpp_wall_s",
    "peak_rss_mb",
    "gt_allowed_cm",
    "cpp_wall_ratio",
    "verdict",
)
"""The clip keys the fleet chart reads, in the order it reads them: a consumer contract, not a dump of the row."""
MACHINE: Machine = Machine(hostname="pablo-rpi", arch="aarch64", libc="2.36", cores=4)
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

    def never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("`ate` was called on a run below D60's pose floor")

    monkeypatch.setattr(fleet_check, "ate", never)
    monkeypatch.setattr(
        fleet_check, "run_segment", lambda *_args, **_kwargs: SegmentRun(estimate=empty_trajectory(), framesets=412, lost=412, wall_s=1.0)
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
        main(Config(manifest=MANIFEST_PATH, segments=(SMOKE_SEGMENTS[1],), output_json=output))
    written: dict = json.loads(output.read_text())
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)
    assert written["clips"][0]["verdict"].startswith("fail:")
    # NaN is what the chart's own `f"{value:.2f}"` reads; None is what it cannot.
    assert math.isnan(asdict(clip_json(dead))["cpp_rmse_cm"])


def test_a_reference_trajectory_that_is_not_here_is_refused_before_the_replay(
    manifest: ReferenceManifest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The manifest says why it cannot be read; paying a 410 s replay to reach a bare ``FileNotFoundError`` does not.

    The two long-tier segments keep their trajectory in the machine-local
    reference bundle, and the pack carries two of ten, so a fleet machine meets
    this whenever it names a clip whose reference did not ship.
    """
    monkeypatch.setenv("SLAM_RS_REFERENCE_DIR", str(tmp_path))

    def never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the replay was paid for before the reference was checked")

    monkeypatch.setattr("slam_rs.apis.fleet_check.run_segment", never)
    segment: ReferenceSegment = manifest.by_id(SMOKE_SEGMENTS[1])
    absent: ReferenceSegment = replace(segment, reference=replace(segment.reference, bundle_only=True))
    with pytest.raises(FileNotFoundError, match="is not in SLAM_RS_REFERENCE_DIR"):
        measure(manifest, absent)


def test_an_unknown_segment_id_is_refused_before_the_first_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--segments <410 s clip> typo`` used to pay the clip and then traceback on the typo.

    Every id the run was given is resolved before the loop opens anything, and
    the manifest's own selector error names the id and the ten it has.
    """

    def never(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a clip was measured before every id was resolved")

    monkeypatch.setattr(fleet_check, "measure", never)
    with pytest.raises(ValueError, match="MIO10_typo.*MIO10_short_2_panorama"):
        main(Config(manifest=MANIFEST_PATH, segments=(SMOKE_SEGMENTS[1], "MIO10_typo"), output_json=tmp_path / "fleet_check.json"))


def test_the_first_clips_evidence_survives_a_directory_that_is_not_there_yet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The JSON is the whole point of the run, and it is written after every clip.

    On a 2 GB device the second clip is what the kernel may refuse, so the first
    clip's row has to be on disk already — and it is not, if the last line of the
    run is what discovers that ``out/`` does not exist.
    """
    monkeypatch.setattr(fleet_check, "measure", lambda _manifest, _segment: PASSING)
    output: Path = tmp_path / "out" / "fleet_check.json"
    main(Config(manifest=MANIFEST_PATH, segments=(SMOKE_SEGMENTS[1],), output_json=output))
    written: dict = json.loads(output.read_text())
    assert list(written) == ["machine", "clips"]
    assert list(written["clips"][0]) == list(CLIP_JSON_KEYS)
