"""Measure catalog RoboCap replay; regression agreement is reported, not gated."""

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from slam_rs.machine import Machine, this_machine, this_peak_rss_mb, this_temperature_c
from slam_rs.reference import ReferenceManifest, RobocapSession, load_manifest
from slam_rs.tracking import SegmentRun, run_robocap
from slam_rs.trajectory import AteResult, Trajectory, ate, empty_trajectory, nonfinite_position_text, read_trajectory, write_trajectory

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
    reference_rmse_cm: float
    """ATE against the regression reference trajectory beside the session; reported, not gated."""
    reference_max_cm: float
    """Largest single-pose residual against the reference."""
    reference_median_cm: float
    """Median residual against the reference."""
    wall_s: float
    """Wall time of the feed loop: decode plus ``track``, nothing logged."""
    ms_per_frameset: float
    """That wall divided by the framesets fed — what one four-camera frameset costs here."""
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
    config_sha256: str
    """SHA-256 of the exact config text the run's estimator was built from."""

    def row(self) -> str:
        """This session as one row of the fleet's runtime-budget table."""
        temps: str = "—" if self.temp_c_before is None or self.temp_c_after is None else f"{self.temp_c_before:.1f} → {self.temp_c_after:.1f}"
        across: str = "—" if self.cross_platform_ate_cm is None else f"{self.cross_platform_ate_cm:.3f}"
        return (
            f"| {self.machine.hostname} | {self.machine.arch} | {self.machine.cores} | {self.segment_id} "
            f"| {self.framesets}/{self.tracked}/{self.lost} | {self.reference_rmse_cm:.2f} | {across} "
            f"| {self.wall_s:.1f} | {self.ms_per_frameset:.1f} "
            f"| {self.realtime_factor_15fps:.2f}x | {self.realtime_factor_30fps:.2f}x "
            f"| {self.peak_rss_mb:.0f} | {temps} |"
        )


@dataclass(slots=True)
class Config:
    """Replay one RoboCap session on this machine, with nothing logged."""

    profile: Literal["reference", "fast"] = "fast"
    """Config overlay applied before tracking."""

    catalog: str | None = None
    """Catalog URL; defaults to the manifest."""
    gpu: bool = False
    """Use the GPU frontend."""
    session: str = "s00000015"
    """RoboCap session id from the manifest. Session 15 is the one with a reference wall on the cap."""
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
    """Replay the session and report regression agreement without gating accuracy."""
    # Every scoring input before the replay: 52.9 s of video must not be spent to
    # reach a `slam` layer that did not ship or a mistyped `--reference-csv`
    # (S22 review). Existence and readability first, and for both together,
    # because reading the layer spins a catalog server and the typed path is the
    # cheap mistake.
    reference_path: Path | None = config.reference_csv or (manifest.package_root / session.reference_csv if session.reference_csv else None)
    reference: Trajectory = read_trajectory(reference_path) if reference_path else empty_trajectory()
    across_reference: Trajectory | None = None
    before: float | None = this_temperature_c()
    run: SegmentRun = run_robocap(
        manifest, session, seconds=config.seconds, window_s=config.window_s, profile=config.profile, catalog=config.catalog, gpu=config.gpu
    )
    after: float | None = this_temperature_c()
    against_reference: AteResult | None = None
    across: float | None = None
    # Finiteness before the alignment, not after it: a NaN position reaches
    # `np.linalg.svd` inside `rigid_alignment` as `LinAlgError`, which the
    # `ValueError` below does not catch — so a diverged estimator lost the whole
    # 52.9 s replay to a traceback, with no row, no trajectory and no JSON, on
    # exactly the machine whose cost this lane exists to measure (S25 review).
    unscored: str | None = nonfinite_position_text(run.estimate)
    if unscored is None:
        try:
            against_reference = ate(run.estimate, reference) if len(reference) else None
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
            against_reference = None
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
            reference_rmse_cm=100.0 * against_reference.rmse_m if against_reference is not None else math.nan,
            reference_max_cm=100.0 * against_reference.max_m if against_reference is not None else math.nan,
            reference_median_cm=100.0 * against_reference.median_m if against_reference is not None else math.nan,
            wall_s=run.wall_s,
            ms_per_frameset=ms_per_frameset,
            realtime_factor_15fps=BUDGET_15FPS_MS / ms_per_frameset,
            realtime_factor_30fps=BUDGET_30FPS_MS / ms_per_frameset,
            peak_rss_mb=this_peak_rss_mb(),
            temp_c_before=before,
            temp_c_after=after,
            cross_platform_ate_cm=across,
            unscored=unscored,
            config_sha256=run.config_sha256,
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
    manifest: ReferenceManifest = load_manifest()
    session: RobocapSession = manifest.robocap.session(config.session)
    machine: Machine = this_machine()
    print(f"{machine.hostname}: {machine.arch}, libc {machine.libc}, {machine.cores} cores")
    print(f"{session.segment_id}: ground truth absent, not scored; regression agreement is not gated")
    started: float = time.monotonic()
    row: RobocapRow
    estimate: Trajectory
    row, estimate = measure(manifest, session, config, machine)
    print(f"profile={config.profile} config_sha256={row.config_sha256}")
    output_csv: Path = config.output_csv if config.output_csv is not None else config.output_json.with_suffix(".csv")
    # Above both writes: a run that spent 52.9 s of video must not lose it to a
    # directory that is not there. `write_trajectory` makes its own parents,
    # which is why the CSV used to work by accident when the two shared one.
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_trajectory(output_csv, estimate)
    config.output_json.write_text(json.dumps({"profile": config.profile, **asdict(row)}, indent=2) + "\n")
    print(row.row())
    print(
        f"{row.tracked} tracked poses -> {output_csv}; {row.reference_rmse_cm:.2f} cm rmse / {row.reference_max_cm:.2f} max / "
        f"{row.reference_median_cm:.2f} median from the regression reference (not gated), {row.ms_per_frameset:.1f} ms per frameset "
        f"({row.realtime_factor_15fps:.2f}x the 15 fps budget, {row.realtime_factor_30fps:.2f}x the 30 fps one)"
    )
    print(f"{config.output_json} written; whole job {time.monotonic() - started:.1f} s")
    if row.unscored is not None:
        raise SystemExit(f"{row.segment_id}: {row.unscored}")
