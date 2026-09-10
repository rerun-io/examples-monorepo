"""Replay one RoboCap session on whatever machine this is, and report what it cost there.

:mod:`slam_rs.apis.fleet_check` answers "does the port run here" with two short
MSD clips. This answers the harder half of the same question with the recording
the target device was built for: 52.9 s of a hand-carried four-camera fisheye
rig at 640x360, the whole clip, scored against the basalt C++ trajectory stored
beside it on the NAS. RoboCap has no ground truth, so agreement with the C++ is
the only accuracy number there is, and it is **reported, not gated** (S17): the
port's kornia-rs frontend picks different corners from basalt's, which on this
rig costs a few centimetres.

Nothing is logged and no viewer is contacted, which is what lets the same run
happen on the cap — a Buildroot appliance with no repository, no compiler and no
Rerun viewer, reached through a pack. The per-frameset cost this prints is
therefore the estimator plus the decode and nothing else.

That is **not** the shape of the C++ wall it is read against: the manifest's
88.91 s for session 15 was measured on cap A with basalt's own Rerun logging on,
at 30 fps over four cameras. So the ratio a row prints compares a wall with
nothing logged against a wall with logging, which is the conservative direction
for the port — the C++ was carrying work this run does not.

The two budgets a row is read against are the cap's, because the cap is where
this has to run live one day: **66.7 ms** per frameset at 15 fps and **33.3 ms**
at 30 fps, over four cameras.
"""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from slam_rs.machine import Machine, this_machine, this_peak_rss_mb, this_temperature_c
from slam_rs.reference import ReferenceManifest, RobocapSession, config_text_sha256, load_manifest, profiled_config_text
from slam_rs.tracking import SegmentRun, robocap_cpp_trajectory, run_robocap
from slam_rs.trajectory import AteResult, Trajectory, ate, nonfinite_position_text, read_trajectory, write_trajectory

BUDGET_15FPS_MS: float = 1e3 / 15.0
"""What one four-camera frameset may cost for the cap to keep up at 15 fps."""
BUDGET_30FPS_MS: float = 1e3 / 30.0
"""The same at 30 fps, which is the rate session 15 was recorded at."""


@dataclass(slots=True, frozen=True)
class RobocapRow:
    """One RoboCap session on one machine: what it agreed with, what it cost, how hot it got.

    :attr:`ms_per_frameset` and the two realtime factors are stored rather than
    derived because ``asdict`` is what the fleet chart reads and a property
    would drop the columns; whoever builds a row owes it the invariant that all
    three follow from :attr:`wall_s` and :attr:`framesets`.
    """

    machine: Machine
    """The host this ran on, as a fleet row names it."""
    segment_id: str
    """Which session ran, in the fleet's short form, e.g. ``robocap-s15``."""
    framesets: int
    """Framesets fed to the estimator."""
    tracked: int
    """Poses it reported."""
    lost: int
    """Framesets still held when the clip ended, so never covered by inertial samples (D17)."""
    cpp_rmse_cm: float
    """ATE against the basalt C++ trajectory beside the session; reported, not gated."""
    cpp_max_cm: float
    """Largest single-pose residual against the C++."""
    cpp_median_cm: float
    """Median residual against the C++."""
    wall_s: float
    """Wall time of the feed loop: decode plus ``track``, nothing logged."""
    ms_per_frameset: float
    """That wall divided by the framesets fed — what one four-camera frameset costs here."""
    cpp_wall_s: float | None
    """What the C++ took over the same footage, where a wall has been measured for this session; None where none has.

    The manifest names the machine it was measured on — cap A for session 15 —
    so on any other machine this is the reference wall and not this host's, the
    same reading the MSD rows' own C++ wall gets.
    """
    realtime_factor_15fps: float
    """The 15 fps budget divided by :attr:`ms_per_frameset`: at or above 1.0 the machine keeps up."""
    realtime_factor_30fps: float
    """The same against the 30 fps budget, which is the rate the session was recorded at."""
    peak_rss_mb: float
    """Peak resident set this process reached, which is what a constrained device is judged on."""
    temp_c_before: float | None
    """Warmest thermal zone before the loop; None on a machine that publishes none."""
    temp_c_after: float | None
    """The same after it."""
    cross_platform_ate_cm: float | None
    """ATE against another machine's trajectory for the same session; None when none was given.

    The MSD smoke clips came out identical to the last printed digit on every
    machine, but MSD is 3 s and 7 s of two- and four-camera video. This is 52.9 s
    on a rig whose frontend has more corners to choose between, so the number is
    measured rather than assumed.
    """
    unscored: str | None
    """Why no agreement could be measured, or None where it was; the sentence :func:`~slam_rs.trajectory.ate` refused the pair with.

    Accuracy is reported and not gated here, and a refusal leaves the three
    ``cpp_`` fields NaN and :attr:`cross_platform_ate_cm` unmeasured while the
    cost beside them is still measured; see ``ate`` for why it is a row.
    """

    def row(self) -> str:
        """This session as one row of the fleet's runtime-budget table."""
        cpp: str = "—" if self.cpp_wall_s is None else f"{self.cpp_wall_s:.1f}"
        temps: str = "—" if self.temp_c_before is None or self.temp_c_after is None else f"{self.temp_c_before:.1f} → {self.temp_c_after:.1f}"
        across: str = "—" if self.cross_platform_ate_cm is None else f"{self.cross_platform_ate_cm:.3f}"
        return (
            f"| {self.machine.hostname} | {self.machine.arch} | {self.machine.cores} | {self.segment_id} "
            f"| {self.framesets}/{self.tracked}/{self.lost} | {self.cpp_rmse_cm:.2f} | {across} "
            f"| {self.wall_s:.1f} | {self.ms_per_frameset:.1f} | {cpp} "
            f"| {self.realtime_factor_15fps:.2f}x | {self.realtime_factor_30fps:.2f}x "
            f"| {self.peak_rss_mb:.0f} | {temps} |"
        )


@dataclass(slots=True)
class Config:
    """Replay one RoboCap session on this machine, with nothing logged."""

    profile: Literal["reference", "fast"] = "reference"
    """Config overlay applied before tracking."""

    artifact_root: Path | None = None
    """Read every recording and sidecar from one directory per segment; see :func:`slam_rs.reference.relocate`."""
    session: str = "s00000015"
    """RoboCap session id from the manifest. Session 15 is the one with a C++ wall on the cap."""
    seconds: float = 0.0
    """Replay this much video time from the first frameset; 0 replays the whole session, which is what a fleet row is."""
    output_json: Path = Path("robocap_fleet.json")
    """Where the machine's facts and the session's numbers are written."""
    output_csv: Path | None = None
    """Where the estimated trajectory goes; defaults to ``<output_json stem>.csv`` beside it."""
    reference_csv: Path | None = None
    """Another machine's trajectory for the same session, for the cross-platform figure."""
    window_s: float = 30.0
    """Longest time window of encoded samples fetched in one round trip."""


def measure(manifest: ReferenceManifest, session: RobocapSession, config: Config, machine: Machine) -> tuple[RobocapRow, Trajectory]:
    """Replay one session and score it against the C++ beside it, and optionally another machine.

    Args:
        manifest: The reference set, which carries the RoboCap lane's configuration.
        session: The session to replay.
        config: Parsed CLI options; the replay reads the span, the window and the
            other machine's trajectory off it.
        machine: The host this is running on, read once by the caller.

    Returns:
        The row, and the estimated trajectory on the device clock so the caller
        can export it. An estimate that associates with nothing, or that carries
        a position which is not finite, has NaN where the agreement would be and
        the reason on :attr:`RobocapRow.unscored`.

    Raises:
        FileNotFoundError: If the session's ``slam`` layer or the other machine's
            trajectory is not a readable file here, before anything is replayed.
    """
    # Every scoring input before the replay: 52.9 s of video must not be spent to
    # reach a `slam` layer that did not ship or a mistyped `--reference-csv`
    # (S22 review). Existence and readability first, and for both together,
    # because reading the layer spins a catalog server and the typed path is the
    # cheap mistake.
    scoring: tuple[Path, ...] = (session.slam_path,) if config.reference_csv is None else (session.slam_path, config.reference_csv)
    for path in scoring:
        if not path.is_file():
            raise FileNotFoundError(f"{session.fleet_id}: {path} is not a file on this machine")
        with path.open("rb") as handle:
            handle.read(1)
    cpp: Trajectory = robocap_cpp_trajectory(manifest, session)
    across_reference: Trajectory | None = None if config.reference_csv is None else read_trajectory(config.reference_csv)
    before: float | None = this_temperature_c()
    run: SegmentRun = run_robocap(manifest, session, seconds=config.seconds, window_s=config.window_s, profile=config.profile)
    after: float | None = this_temperature_c()
    against_cpp: AteResult | None = None
    across: float | None = None
    # Finiteness before the alignment, not after it: a NaN position reaches
    # `np.linalg.svd` inside `rigid_alignment` as `LinAlgError`, which the
    # `ValueError` below does not catch — so a diverged estimator lost the whole
    # 52.9 s replay to a traceback, with no row, no trajectory and no JSON, on
    # exactly the machine whose cost this lane exists to measure (S25 review).
    unscored: str | None = nonfinite_position_text(run.estimate)
    if unscored is None:
        try:
            against_cpp = ate(run.estimate, cpp)
            across = None if across_reference is None else 100.0 * ate(run.estimate, across_reference).rmse_m
        except ValueError as association_failed:
            # 52.9 s of video has already been paid for by here, and the wall,
            # the budget and the temperatures it bought are the row's reason to
            # exist — so an estimate that associates with nothing loses its
            # agreement and not the run. The clause is the sentence `ate`
            # refused with, tolerance and all; both numbers go, not the half
            # that may have associated already. `ValueError` and not
            # `Exception`, so a beartype violation still raises. `main` prints
            # the row and then exits non-zero.
            against_cpp = None
            across = None
            unscored = str(association_failed)
    ms_per_frameset: float = 1e3 * run.wall_s / max(run.framesets, 1)
    return (
        RobocapRow(
            machine=machine,
            segment_id=session.fleet_id,
            framesets=run.framesets,
            tracked=len(run.estimate),
            lost=run.lost,
            cpp_rmse_cm=100.0 * against_cpp.rmse_m if against_cpp is not None else math.nan,
            cpp_max_cm=100.0 * against_cpp.max_m if against_cpp is not None else math.nan,
            cpp_median_cm=100.0 * against_cpp.median_m if against_cpp is not None else math.nan,
            wall_s=run.wall_s,
            ms_per_frameset=ms_per_frameset,
            cpp_wall_s=session.expected_cpp_wall_s,
            realtime_factor_15fps=BUDGET_15FPS_MS / ms_per_frameset,
            realtime_factor_30fps=BUDGET_30FPS_MS / ms_per_frameset,
            peak_rss_mb=this_peak_rss_mb(),
            temp_c_before=before,
            temp_c_after=after,
            cross_platform_ate_cm=across,
            unscored=unscored,
        ),
        run.estimate,
    )


def main(config: Config) -> None:
    """Replay the session, print its row, and write the trajectory and the JSON.

    The result carries the profile and the resolved configuration SHA-256 once.

    Args:
        config: Parsed CLI options.

    Raises:
        SystemExit: If the estimate associated with nothing, which is not an
            accuracy number to report but a run that went wrong. Both outputs
            are written first: the cost they carry was measured.
    """
    manifest: ReferenceManifest = load_manifest(artifact_root=config.artifact_root)
    session: RobocapSession = manifest.robocap.session(config.session)
    machine: Machine = this_machine()
    print(f"{machine.hostname}: {machine.arch}, libc {machine.libc}, {machine.cores} cores")
    print(f"{session.segment_id}: {session.base_path.name} against {session.slam_path.name} ({session.basalt_num_poses} C++ poses)")
    config_digest: str = config_text_sha256(
        profiled_config_text(manifest.package_root / manifest.robocap.vio_config, config.profile, manifest.package_root / "configs/profiles")
    )
    print(f"profile={config.profile} config_sha256={config_digest}")
    started: float = time.monotonic()
    row: RobocapRow
    estimate: Trajectory
    row, estimate = measure(manifest, session, config, machine)
    output_csv: Path = config.output_csv if config.output_csv is not None else config.output_json.with_suffix(".csv")
    # Above both writes: a run that spent 52.9 s of video must not lose it to a
    # directory that is not there. `write_trajectory` makes its own parents,
    # which is why the CSV used to work by accident when the two shared one.
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_trajectory(output_csv, estimate)
    config.output_json.write_text(json.dumps({"profile": config.profile, "config_sha256": config_digest, **asdict(row)}, indent=2) + "\n")
    print(row.row())
    print(
        f"{row.tracked} tracked poses -> {output_csv}; {row.cpp_rmse_cm:.2f} cm rmse / {row.cpp_max_cm:.2f} max / "
        f"{row.cpp_median_cm:.2f} median from the C++, {row.ms_per_frameset:.1f} ms per frameset "
        f"({row.realtime_factor_15fps:.2f}x the 15 fps budget, {row.realtime_factor_30fps:.2f}x the 30 fps one)"
    )
    print(f"{config.output_json} written; whole job {time.monotonic() - started:.1f} s")
    if row.unscored is not None:
        raise SystemExit(f"{row.segment_id}: {row.unscored}")
