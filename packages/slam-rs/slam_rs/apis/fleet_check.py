"""Run the reference smoke clips on whatever machine this is, and report the D60 verdict beside it.

The V2 gate answers "is the port accurate enough" on one host. This answers
"does it run here" on any of them: the same two smoke clips through the same
:func:`slam_rs.tracking.run_segment` the gate drives, scored against the same two
references, with the machine's own facts on the row so a number is never read
apart from the CPU that produced it. Nothing is logged and no viewer is
contacted, which is what lets it run on the constrained target — 2 GB of RAM, no
repository, the sources and the compiled core delivered in a pack.

Speed is **reported, not gated**: ``expected_cpp_wall_s`` was measured on one
x86-64 host, so the ratio to it says how fast this machine is, not whether the
port is a port.
"""

import json
import os
import platform
import resource
from dataclasses import asdict, dataclass
from pathlib import Path

from slam_rs.reference import ATE_VS_CPP_CM, GT_BAND_RATIO, MANIFEST_PATH, CppAte, ReferenceManifest, ReferenceSegment, load_manifest
from slam_rs.tracking import SegmentRun, run_segment
from slam_rs.trajectory import AteResult, Trajectory, ate, read_trajectory

SMOKE_SEGMENTS: tuple[str, ...] = ("msd-g2__MGO_others__MGO09_short_1_updown", "msd-index__MIO_others__MIO10_short_2_panorama")
"""Both smoke clips, the four-camera 3 s one first: a broken machine says so sooner."""


@dataclass(slots=True, frozen=True)
class Machine:
    """The host a row was measured on."""

    hostname: str
    """What the machine calls itself."""
    arch: str
    """``platform.machine()``: the port is built for ``x86_64`` and ``aarch64``."""
    libc: str
    """The C library the compiled core is linked against, which is what limits where a pack can go."""
    cores: int
    """Cores visible to the process; the estimator runs single-threaded, the decode does not."""


def this_libc() -> str:
    """The C library this machine offers, in the terms that decide where a compiled core can travel.

    ``CS_GNU_LIBC_VERSION`` is a glibc name and macOS raises ``ValueError`` on
    it rather than returning ``None``, so it has to be looked up before it is
    asked for. Where there is no glibc the answer is the macOS release, because
    that is what the extension binds ``libSystem`` from.
    """
    if "CS_GNU_LIBC_VERSION" in os.confstr_names:
        version: str | None = os.confstr("CS_GNU_LIBC_VERSION")
        if version is not None:
            return version.removeprefix("glibc ")
    macos: str = platform.mac_ver()[0]
    return f"libSystem, macOS {macos}" if macos else "unknown"


def this_peak_rss_mb() -> float:
    """The largest resident set this process has held, in megabytes.

    ``ru_maxrss`` counts kilobytes on Linux and bytes on macOS. The unit cannot
    be guessed, because on the constrained target this number is the whole
    question.
    """
    peak: int = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024.0 * 1024.0) if platform.system() == "Darwin" else peak / 1024.0


def this_machine() -> Machine:
    """What this host is, as a row names it."""
    return Machine(hostname=platform.node(), arch=platform.machine(), libc=this_libc(), cores=os.cpu_count() or 0)


@dataclass(slots=True, frozen=True)
class ClipResult:
    """One clip's numbers on one machine, and the D60 clauses they meet."""

    segment_id: str
    """Manifest id of the clip that ran."""
    framesets: int
    """Framesets fed to the estimator."""
    tracked: int
    """Poses it reported."""
    lost: int
    """Framesets still held when the clip ended, so never covered by inertial samples (D17)."""
    cpp_rmse_cm: float
    """ATE against the basalt C++ trajectory on the same footage."""
    gt_rmse_cm: float
    """ATE against the ``gt.csv`` sidecar."""
    cpp_gt_band_cm: tuple[float, float]
    """The C++'s own ground-truth error on this clip, in its ``f32`` and ``f64`` precisions (D60)."""
    wall_s: float
    """Wall time of the feed loop: decode plus ``track``, nothing logged."""
    cpp_wall_s: float
    """What the C++ took over the same footage, measured on one x86-64 host."""
    peak_rss_mb: float
    """Peak resident set this process reached, which is what a 2 GB device is judged on."""

    @property
    def gt_allowed_cm(self) -> float:
        """Ground-truth error D60 allows: the ratio times the worse of the C++'s own two precisions."""
        return GT_BAND_RATIO * max(self.cpp_gt_band_cm)

    @property
    def cpp_wall_ratio(self) -> float:
        """Times the C++'s wall this run took — a fact about the machine, not a clause."""
        return self.wall_s / self.cpp_wall_s

    @property
    def failures(self) -> tuple[str, ...]:
        """Every D60 clause these numbers miss, in the order D60 states them; both default clips are far under :data:`~slam_rs.reference.PATH_BOUND_MAX_CLIP_S` seconds, which is what makes the 2 cm path bound apply."""
        failures: list[str] = []
        if self.lost:
            failures.append(f"{self.lost} of {self.framesets} framesets never got the inertial samples that cover them")
        if self.cpp_rmse_cm > ATE_VS_CPP_CM:
            failures.append(f"{self.cpp_rmse_cm:.2f} cm from the C++ trajectory, gate is {ATE_VS_CPP_CM:.0f} cm")
        if self.gt_rmse_cm > self.gt_allowed_cm:
            failures.append(
                f"{self.gt_rmse_cm:.2f} cm from ground truth, gate is {GT_BAND_RATIO}x the worse of the C++'s own "
                f"[f32 {self.cpp_gt_band_cm[0]:.2f}, f64 {self.cpp_gt_band_cm[1]:.2f}] cm = {self.gt_allowed_cm:.2f} cm"
            )
        return tuple(failures)

    @property
    def verdict(self) -> str:
        """``pass``, or every clause this clip missed."""
        return "pass" if not self.failures else "fail: " + "; ".join(self.failures)

    def row(self, machine: Machine) -> str:
        """This clip as one row of the fleet table, the machine it ran on first."""
        return (
            f"| {machine.hostname} | {machine.arch} | {machine.libc} | {machine.cores} | {self.segment_id} "
            f"| {self.framesets}/{self.tracked}/{self.lost} | {self.cpp_rmse_cm:.2f} "
            f"| {self.gt_rmse_cm:.2f} (allowed {self.gt_allowed_cm:.2f}) | {self.verdict} "
            f"| {self.wall_s:.2f} | {self.cpp_wall_ratio:.2f}x | {self.peak_rss_mb:.0f} |"
        )


def measure(manifest: ReferenceManifest, segment: ReferenceSegment) -> ClipResult:
    """Run one clip through the estimator and score it against both references.

    Args:
        manifest: The reference set, which resolves the dataset's config and the C++ trajectory.
        segment: The clip to run; its three artifacts must be on this machine.

    Returns:
        The clip's numbers, with the peak resident set the process has reached.
    """
    run: SegmentRun = run_segment(manifest, segment)
    against_cpp: AteResult = ate(run.estimate, read_trajectory(manifest.cpp_trajectory(segment).path))
    truth: Trajectory = read_trajectory(segment.gt_csv)
    against_gt: AteResult = ate(run.estimate, truth)
    expected: CppAte = segment.reference.expected_cpp_ate
    return ClipResult(
        segment_id=segment.segment_id,
        framesets=run.framesets,
        tracked=len(run.estimate),
        lost=run.lost,
        cpp_rmse_cm=100.0 * against_cpp.rmse_m,
        gt_rmse_cm=100.0 * against_gt.rmse_m,
        cpp_gt_band_cm=(expected.rmse_cm, expected.rmse_cm_f64),
        wall_s=run.wall_s,
        cpp_wall_s=segment.reference.expected_cpp_wall_s,
        peak_rss_mb=this_peak_rss_mb(),
    )


@dataclass(slots=True)
class Config:
    """Run the reference smoke clips on this machine and report the D60 verdict."""

    manifest: Path = MANIFEST_PATH
    """Reference manifest; a machine without the NAS runs a copy with the artifact prefix rewritten."""
    segments: tuple[str, ...] = SMOKE_SEGMENTS
    """Clips to run, in order."""
    output_json: Path = Path("fleet_check.json")
    """Where the machine's facts and every clip's numbers are written."""


def main(config: Config) -> None:
    """Run each clip, print its row as it is measured, and write the JSON.

    The JSON is rewritten after every clip rather than at the end: on a 2 GB
    device the second clip is what the kernel may refuse, and the first clip's
    evidence has to survive it.

    Args:
        config: Parsed CLI options.

    Raises:
        SystemExit: If any clip missed a D60 clause.
    """
    manifest: ReferenceManifest = load_manifest(config.manifest)
    machine: Machine = this_machine()
    print(f"{machine.hostname}: {machine.arch}, libc {machine.libc}, {machine.cores} cores")
    results: list[ClipResult] = []
    for segment_id in config.segments:
        results.append(measure(manifest, manifest.by_id(segment_id)))
        print(results[-1].row(machine))
        payload: dict[str, object] = {
            "machine": asdict(machine),
            "clips": [asdict(clip) | {"gt_allowed_cm": clip.gt_allowed_cm, "cpp_wall_ratio": clip.cpp_wall_ratio, "verdict": clip.verdict} for clip in results],
        }
        config.output_json.write_text(json.dumps(payload, indent=2))
    missed: list[ClipResult] = [clip for clip in results if clip.failures]
    if missed:
        raise SystemExit("\n".join(f"{clip.segment_id}: {clip.verdict}" for clip in missed))
