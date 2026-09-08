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
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from slam_rs import _core
from slam_rs.machine import Machine, this_machine, this_peak_rss_mb
from slam_rs.reference import (
    GT_BAND_RATIO,
    MIN_TRACKED_POSES,
    SMOKE_SEGMENTS,
    CppAte,
    GatePolicy,
    ReferenceManifest,
    ReferenceSegment,
    d60_failures,
    load_manifest,
)
from slam_rs.reference_bundle import BundleFile
from slam_rs.tracking import SegmentRun, run_segment
from slam_rs.trajectory import AteResult, Trajectory, ate, extent_m, nonfinite_position_text, read_trajectory


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
    """ATE against the basalt C++ trajectory on the same footage; NaN where the run was below D60's pose floor and never scored."""
    gt_rmse_cm: float
    """ATE against the ``gt.csv`` sidecar; NaN on the same runs :attr:`cpp_rmse_cm` is."""
    cpp_gt_band_cm: tuple[float, float]
    """The C++'s own ground-truth error on this clip, in its ``f32`` and ``f64`` precisions (D60)."""
    wall_s: float
    """Wall time of the feed loop: decode plus ``track``, nothing logged."""
    cpp_wall_s: float
    """What the C++ took over the same footage, measured on one x86-64 host."""
    peak_rss_mb: float
    """Peak resident set this process reached, which is what a 2 GB device is judged on."""
    # What D60's clauses are decided from, beyond the numbers a row prints. The
    # JSON the chart reads is :data:`CLIP_JSON_KEYS`, not this value, so a clause
    # input is not a column until it is named there.
    gate_policy: GatePolicy
    """How hard D60 lets this clip be gated; a ``no_divergence`` clip gates loss and extent alone."""
    replayed_s: float
    """Sensor seconds the estimate spans, which is what decides whether the 2 cm path bound applies."""
    cpp_associated: int
    """Estimate poses that found a C++ pose inside the association tolerance."""
    extent_m: float
    """Diagonal of the estimate's bounding box, metres."""
    truth_extent_m: float
    """The same for the ground truth, which is what a ``no_divergence`` run is bounded against."""
    poses_finite: bool
    """Whether every estimated position is finite."""
    unscored: str | None
    """Why nothing could be scored, or None where it was; the sentence :func:`~slam_rs.trajectory.ate` refused the pair with.

    D60's pose floor is not the only way a clip goes unscored; see ``ate`` for
    why a refusal is a row here.
    """

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
        """Every D60 clause these numbers miss, through the gate's own clause builder.

        The rule is :func:`slam_rs.reference.d60_failures` and not a second copy
        of it: a row is read on a machine that may have been given any clip by
        ``--segments``, so the conditions on the two error bounds — the clip's
        length and its gate policy — decide the verdict as much as the numbers do.
        """
        if self.unscored is not None:
            # Alone, like D60's own pose floor and for the same reason: with no
            # error to read, the rest of D60 has nothing to say, and what the row
            # owes its reader is the sentence that stopped the scoring.
            return (self.unscored,)
        return tuple(
            d60_failures(
                gate_policy=self.gate_policy,
                framesets=self.framesets,
                tracked=self.tracked,
                lost=self.lost,
                associated=self.cpp_associated,
                replayed_s=self.replayed_s,
                cpp_rmse_cm=self.cpp_rmse_cm,
                gt_rmse_cm=self.gt_rmse_cm,
                band=self.cpp_gt_band_cm,
                extent_m=self.extent_m,
                truth_extent_m=self.truth_extent_m,
                poses_finite=self.poses_finite,
            )
        )

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


def check_scoring_inputs(manifest: ReferenceManifest, segment: ReferenceSegment) -> tuple[Path, Path]:
    """Where one clip's two scoring inputs are, refusing the clip when either cannot be read here.

    Both of them, before anything is replayed: a 410 s clip is a long way to
    travel to reach a bare ``FileNotFoundError`` from the CSV reader, and the
    corpus a fleet machine reads is often partial — the pack carries two of the
    ten segments, and ``--artifact-root`` points at whatever unpacked.
    :func:`main` asks this of every clip it was given before the first replay,
    and :func:`measure` asks it again for the clip it is about to run.

    Args:
        manifest: The reference set, which resolves the C++ trajectory.
        segment: The clip about to be scored.

    Returns:
        The basalt C++ trajectory and the ``gt.csv`` sidecar, in that order.

    Raises:
        FileNotFoundError: If either is not a readable file on this machine — the
            C++ trajectory with the manifest's own sentence about why, since the
            two long-tier segments keep theirs in a bundle.
    """
    reference: BundleFile = manifest.cpp_trajectory(segment)
    if not reference.available:
        raise FileNotFoundError(reference.reason)
    for path in (reference.path, segment.gt_csv):
        if not path.is_file():
            raise FileNotFoundError(f"{segment.segment_id}: {path} is not a file on this machine")
        # Opened, not only counted: a pack unpacked without the permission to
        # read it costs the same replay as a pack that is missing the file.
        with path.open("rb") as handle:
            handle.read(1)
    return reference.path, segment.gt_csv


def measure(manifest: ReferenceManifest, segment: ReferenceSegment, gpu: bool = False) -> ClipResult:
    """Run one clip through the estimator and score it against both references.

    Args:
        manifest: The reference set, which resolves the dataset's config and the C++ trajectory.
        segment: The clip to run; its three artifacts must be on this machine.
        gpu: Put the frontend on this machine's GPU through CubeCL instead of the
            CPU port. A core built without a GPU cargo feature refuses it rather
            than quietly running on the CPU, which is what makes a GPU row a GPU
            row.

    Returns:
        The clip's numbers, with the peak resident set the process has reached.
        A run that could not be scored carries NaN for both errors and the reason
        on :attr:`ClipResult.unscored`, and the verdict says so: below D60's pose
        floor, with a position that is not finite, or with poses enough to score
        and none of them on the references' clock.

    Raises:
        FileNotFoundError: If either scoring input cannot be read here
            (:func:`check_scoring_inputs`).
    """
    cpp_csv: Path
    gt_csv: Path
    cpp_csv, gt_csv = check_scoring_inputs(manifest, segment)
    # Read before the replay too, so a CSV that is there but is not a trajectory
    # costs nothing either.
    cpp: Trajectory = read_trajectory(cpp_csv)
    truth: Trajectory = read_trajectory(gt_csv)
    run: SegmentRun = run_segment(manifest, segment, gpu=gpu)
    tracked: int = len(run.estimate)
    # A run below the floor is not scored at all: `ate` has no pose to align and
    # raises, and a machine that tracked nothing is precisely the machine this
    # tool exists to report on. The floor is the verdict (D60, `d60_failures`).
    scored: bool = tracked >= MIN_TRACKED_POSES
    against_cpp: AteResult | None = None
    against_gt: AteResult | None = None
    # Finiteness before the alignment, not after it: a NaN position reaches
    # `np.linalg.svd` inside `rigid_alignment` as `LinAlgError`, which the
    # `ValueError` below does not catch, so the clip left no row and this
    # tool's whole output — the JSON `main` writes after every clip — was
    # never written for the machine that diverged (S25 review). Below the
    # floor the clause is the floor's, which stands alone.
    nonfinite: str | None = nonfinite_position_text(run.estimate)
    unscored: str | None = nonfinite if scored else None
    if scored and nonfinite is None:
        try:
            against_cpp = ate(run.estimate, cpp)
            against_gt = ate(run.estimate, truth)
        except ValueError as association_failed:
            # The other unscored case, and the one only a whole machine reaches:
            # poses enough to score, none of them on the references' clock. The
            # clause is the sentence `ate` refused with, tolerance and all, read
            # off the failure rather than by counting the associations again.
            # Both errors go, not the half that may have associated already: the
            # two references share the device clock, so whichever call refused,
            # the estimate is what moved. `ValueError` and not `Exception`, so a
            # beartype violation still raises.
            against_cpp = None
            against_gt = None
            unscored = str(association_failed)
    expected: CppAte = segment.reference.expected_cpp_ate
    return ClipResult(
        segment_id=segment.segment_id,
        framesets=run.framesets,
        tracked=tracked,
        lost=run.lost,
        cpp_rmse_cm=100.0 * against_cpp.rmse_m if against_cpp is not None else math.nan,
        gt_rmse_cm=100.0 * against_gt.rmse_m if against_gt is not None else math.nan,
        cpp_gt_band_cm=(expected.rmse_cm, expected.rmse_cm_f64),
        wall_s=run.wall_s,
        cpp_wall_s=segment.reference.expected_cpp_wall_s,
        peak_rss_mb=this_peak_rss_mb(),
        gate_policy=segment.reference.gate_policy,
        replayed_s=float(run.estimate.t_ns[-1] - run.estimate.t_ns[0]) * 1e-9 if tracked else 0.0,
        cpp_associated=against_cpp.n_associated if against_cpp is not None else 0,
        extent_m=extent_m(run.estimate),
        truth_extent_m=extent_m(truth),
        poses_finite=nonfinite is None,
        unscored=unscored,
    )


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
"""The clip keys the fleet chart reads, in the order it reads them.

A selection of :class:`ClipResult` and not a dump of it, on purpose: the row
carries what a verdict is decided from, which grows as D60 is read more
carefully, and the JSON is a consumer contract — a new clause input must not
become a new column by accident. Each name is a :class:`ClipResult` field or
one of its properties, and its docstring there is the column's meaning.
"""


def clip_json(clip: ClipResult) -> dict[str, object]:
    """One measured clip in the shape the chart reads."""
    return {key: getattr(clip, key) for key in CLIP_JSON_KEYS}


Lane: TypeAlias = Literal["cpu", "cuda", "wgpu"]
"""Which frontend measured a row: the CPU port, or the GPU runtime the core was built with."""


def this_lane(gpu: bool) -> Lane:
    """The lane this core runs a clip on, named after the runtime rather than the flag.

    ``--gpu`` does not say which GPU: the NVIDIA ``gpu`` feature and the portable
    ``gpu-wgpu`` one are two builds of one source behind the same flag, and they
    do not agree on every clip, so a row labelled ``gpu`` has lost the first
    thing its reader asks. The extension reports the feature it was compiled with
    (:data:`slam_rs._core.gpu_backend`) and the lane is that name.

    Args:
        gpu: Whether the run was asked for the GPU frontend.

    Returns:
        ``cpu`` for the CPU port, or the compiled-in runtime's own name.

    Raises:
        ValueError: If the GPU frontend was asked of a core built without a GPU
            cargo feature. :class:`slam_rs._core.Vio` refuses such a run too;
            asking here is what keeps the refusal ahead of the manifest and the
            first replay.
    """
    if not gpu:
        return "cpu"
    backend: Lane | None = _core.gpu_backend
    if backend is None:
        raise ValueError("--gpu needs a core built with a GPU cargo feature; this one has none (slam-rs-gpu-build for CUDA, slam-rs-wgpu-build for wgpu)")
    return backend


@dataclass(slots=True)
class Config:
    """Run the reference smoke clips on this machine and report the D60 verdict."""

    artifact_root: Path | None = None
    """Read every recording and sidecar from one directory per segment; see :func:`slam_rs.reference.relocate`."""
    segments: tuple[str, ...] = SMOKE_SEGMENTS
    """Clips to run, in order; naming none of them is refused rather than run as a pass."""
    output_json: Path = Path("fleet_check.json")
    """Where the machine's facts and every clip's numbers are written."""
    gpu: bool = False
    """Run the frontend on this machine's GPU through CubeCL instead of the CPU port.

    Which lane produced a row is a fact about the run and not about the machine
    or the clip, so it is written as the JSON's own ``lane`` key beside
    ``machine`` and ``clips`` — :data:`CLIP_JSON_KEYS` is a consumer contract and
    gains nothing. The key's value is the runtime, not the flag:
    :func:`this_lane`.
    """


def main(config: Config) -> None:
    """Run each clip, print its row as it is measured, and write the JSON.

    The JSON is rewritten after every clip rather than at the end: on a 2 GB
    device the second clip is what the kernel may refuse, and the first clip's
    evidence has to survive it.

    Args:
        config: Parsed CLI options.

    Raises:
        ValueError: If ``--segments`` names no clip at all, if it names an id the
            manifest does not have, or if ``--gpu`` was asked of a core built
            without a GPU cargo feature (:func:`this_lane`).
        FileNotFoundError: If any named clip's scoring inputs cannot be read
            here. All of them are decided before the first replay.
        SystemExit: If any clip missed a D60 clause.
    """
    # An empty selection used to validate nothing, write no JSON and return zero,
    # which a script reads as this machine having passed (S24 review).
    if not config.segments:
        raise ValueError("--segments named no clip; a run that measures nothing is not a pass")
    # Before any file is opened: a `--gpu` run has no lane to report on a core
    # built without a GPU feature, and that costs nothing to say here.
    lane: Lane = this_lane(config.gpu)
    manifest: ReferenceManifest = load_manifest(artifact_root=config.artifact_root)
    # Every id resolved before the first replay, not one at a time inside the
    # loop: `--segments <410 s clip> typo` used to pay that clip and then reach
    # the typo (S22 review).
    segments: tuple[ReferenceSegment, ...] = tuple(manifest.by_id(segment_id) for segment_id in config.segments)
    # The same for every clip's scoring inputs: the second clip's missing
    # sidecar must not cost the first clip's replay (S22 review).
    for segment in segments:
        check_scoring_inputs(manifest, segment)
    machine: Machine = this_machine()
    print(f"{machine.hostname}: {machine.arch}, libc {machine.libc}, {machine.cores} cores, {lane} lane")
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    results: list[ClipResult] = []
    for segment in segments:
        results.append(measure(manifest, segment, config.gpu))
        print(results[-1].row(machine))
        payload: dict[str, object] = {
            "machine": asdict(machine),
            "lane": lane,
            "clips": [clip_json(clip) for clip in results],
        }
        config.output_json.write_text(json.dumps(payload, indent=2))
    missed: list[ClipResult] = [clip for clip in results if clip.failures]
    if missed:
        raise SystemExit("\n".join(f"{clip.segment_id}: {clip.verdict}" for clip in missed))
