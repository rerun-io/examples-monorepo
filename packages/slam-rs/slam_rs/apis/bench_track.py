"""Measure ``Vio.track`` on a dumped segment, one lane against another, interleaved.

This is the harness every accepted speed row in the slam-rs reports was decided
by. It lived in ``/tmp`` through three stages and was forked twice; it is here so
the evidence behind a row can be re-run rather than re-derived.

**The protocol**, which is the part that matters more than the code:

* **No decoder and no Rerun.** The framesets are replayed out of an ``.npz``
  dumped once by ``tests/tools/dump_clip.py --npz``, so the number is the
  estimator's and not AV1's.
* **One pinned core** (``--pin-core``). The lane runs single-threaded by
  contract (D31), and on this host an unpinned run drifts by more than the
  differences being measured. The GPU lane's CubeCL worker thread shares that
  core, deliberately: the pinned figure is what a cap-sized machine sees.
* **Interleaved lanes, best-of-three medians.** Every round runs every lane, in
  order, so a host that warms or drifts moves all of them together; the reported
  figure per lane is the **lowest** of its per-round medians. That is what
  cancels drift, and it is why a row is only believed when it is lower in
  *every* pair.
* **Isolation A/B.** To attribute a change, run one lane before and after it and
  compare pair by pair — not two lanes against each other, and not two builds
  whose codegen also differs. A change that gains less than this host's
  build-to-build noise (about 0.4 ms) is not a change.
* **Never set ``CUBECL_DEBUG_LOG``** while timing: it turns on Vulkan validation
  layers and costs 30x per launch.

The dump is what ``tests/tools/dump_clip.py --npz`` writes:
``images``/``t_ns``/``imu_counts``/``imu_t``/``imu_g``/``imu_a``/``safe_radius``
in the ``.npz``, plus a sibling ``<dump>.calib.pkl`` holding the feed's own
``cameras`` and ``imu`` dataclasses:

.. code-block:: text

    images       uint8   (framesets, cameras, height, width)
    t_ns         int64   (framesets,)          frameset timestamps
    imu_counts   int64   (framesets,)          IMU samples belonging to each
    imu_t        int64   (samples,)            concatenated over framesets
    imu_g        float64 (samples, 3)          rad/s
    imu_a        float64 (samples, 3)          m/s^2
    safe_radius  int64   ()                    asserted against the config
"""

import os
import pickle
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, get_args

import numpy as np
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CameraCalib, Frameset, ImuCalib, ImuStream
from slam_rs.reference import config_text_sha256, profiled_config_text
from slam_rs.tracking import Lockstep

Lane: TypeAlias = Literal["cpu", "gpu"]
"""Which frontend backend a round runs: the ported CPU one, or the CubeCL one."""

CLOCK_TICKS: float = float(os.sysconf("SC_CLK_TCK"))
"""Kernel clock ticks per second, for reading process CPU time out of /proc."""
SAMPLE_PERIOD_S: float = 0.1
"""How often the CPU sampler wakes: 10 Hz, as every recorded row was sampled."""


def process_cpu_s() -> float:
    """User plus system CPU time of this process, in seconds.

    Returns:
        The sum of ``utime`` and ``stime`` from ``/proc/self/stat``. Read after
        the ``comm`` field, which may itself contain spaces and brackets.
    """
    fields: list[str] = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    # `state` is fields[0] once `comm` is behind us, so `utime` is 11 and `stime` 12.
    return (int(fields[11]) + int(fields[12])) / CLOCK_TICKS


class CpuSampler(threading.Thread):
    """Process CPU utilisation sampled at 10 Hz on a side thread until stopped.

    The GPU lane's whole argument is that the host is the critical path, so the
    host figure has to be measured beside the timings rather than inferred.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop: threading.Event = threading.Event()
        self.samples: list[float] = []

    def run(self) -> None:
        """Append one utilisation percentage per period until ``stop`` is set."""
        last_cpu: float = process_cpu_s()
        last_wall: float = time.monotonic()
        while not self.stop.wait(SAMPLE_PERIOD_S):
            now_cpu: float = process_cpu_s()
            now_wall: float = time.monotonic()
            span: float = now_wall - last_wall
            if span > 0.0:
                self.samples.append(100.0 * (now_cpu - last_cpu) / span)
            last_cpu, last_wall = now_cpu, now_wall


@dataclass(slots=True, frozen=True)
class Framesets:
    """One segment's decoded framesets and IMU, as the dump holds them."""

    images: UInt8[ndarray, "n cameras height width"]
    """Grayscale frames, frameset-major then camera."""
    t_ns: Int64[ndarray, " n"]
    """Frameset timestamps."""
    imu_offsets: Int64[ndarray, " n_plus_one"]
    """Frameset ``i``'s IMU samples are ``imu_offsets[i]..imu_offsets[i + 1]``."""
    imu_t: Int64[ndarray, " samples"]
    """IMU timestamps, concatenated over framesets."""
    imu_gyro: Float64[ndarray, "samples 3"]
    """Angular rate in rad/s."""
    imu_accel: Float64[ndarray, "samples 3"]
    """Specific force in m/s^2."""
    safe_radius: float
    """``optical_flow_image_safe_radius`` the dump was decoded under."""
    cameras: tuple[CameraCalib, ...]
    """The feed's per-camera calibration, from the sibling pickle."""
    imu: ImuCalib
    """The feed's IMU noise model, from the same pickle."""


@dataclass(slots=True, frozen=True)
class LaneRound:
    """What one lane produced in one round."""

    lane: Lane
    """Backend this round ran."""
    round_index: int
    """Round it belongs to, counted from one."""
    track_ms: Float64[ndarray, " calls"]
    """Wall time of every ``Vio.track`` call, in milliseconds."""
    wall_s: float
    """Wall time of the whole replay."""
    cpu_pct: Float64[ndarray, " samples"]
    """Process CPU utilisation, sampled at 10 Hz."""

    @property
    def median_ms(self) -> float:
        """Median ``track`` time of this round."""
        return float(np.median(self.track_ms))


@dataclass(slots=True)
class Config:
    """Options for the slam-rs track benchmark."""

    dump: Path
    """The ``.npz`` of decoded framesets to replay; a sibling ``.calib.pkl`` carries the calibration."""
    config: Path
    """The basalt VIO config JSON the reference run used."""
    profile: Literal["reference", "fast"] = "fast"
    """Config overlay applied before tracking."""
    lanes: tuple[Lane, ...] = ("cpu", "gpu")
    """Backends to interleave, in the order each round runs them."""
    rounds: int = 3
    """Rounds per lane; the reported figure is the lowest of a lane's per-round medians."""
    limit: int | None = None
    """Replay only the first this many framesets."""
    pin_core: int | None = None
    """Pin the process to this core before measuring, as every recorded row was."""
    label: str = "run"
    """Tag printed on every ``RESULT`` line, so a shell loop can grep its own rows."""


def interleave(lanes: tuple[Lane, ...], rounds: int) -> list[tuple[int, Lane]]:
    """The measurement schedule: every lane once per round, rounds in order.

    Round-major rather than lane-major on purpose. A host that warms up, or
    drifts over the minutes a sweep takes, moves the lanes it visits *within* a
    round together, so the comparison survives it; running all of lane A and
    then all of lane B measures the drift as if it were the change.

    Args:
        lanes: Backends to compare, in the order each round runs them.
        rounds: Rounds to run.

    Returns:
        ``(round_index, lane)`` pairs in execution order, rounds counted from one.
    """
    return [(index + 1, lane) for index in range(rounds) for lane in lanes]


def best_median_ms(rounds: list[LaneRound]) -> float:
    """The lowest per-round median of one lane's rounds, in milliseconds.

    The best of the medians, not the median of everything and not the minimum
    call: a per-round median is robust to the few framesets where the estimator
    takes a keyframe, and taking the best of them reports the host at its least
    disturbed, which is the figure that reproduces.

    Args:
        rounds: One lane's rounds; must not be empty.

    Returns:
        The lowest median.

    Raises:
        ValueError: If ``rounds`` is empty.
    """
    if not rounds:
        raise ValueError("a lane with no rounds has no best median")
    return min(entry.median_ms for entry in rounds)


def load_framesets(dump: Path, limit: int | None) -> Framesets:
    """Read a dumped segment and its calibration.

    Args:
        dump: The ``.npz`` written for a reference segment.
        limit: Replay only the first this many framesets, or all of them.

    Returns:
        The framesets, with the IMU counts turned into offsets.
    """
    data = np.load(dump)
    # Our own dump, written beside the npz by the same tool that decoded it.
    with dump.with_suffix(dump.suffix + ".calib.pkl").open("rb") as handle:
        calibration = pickle.load(handle)
    # Annotated so the two shapes the pickle must have are written down once;
    # this package runs beartype with PEP 526 checks off (a per-frame loop
    # cannot afford them), so the guard against a dump from another version is
    # the safe-radius check in `main` rather than these lines.
    cameras: tuple[CameraCalib, ...] = tuple(calibration["cameras"])
    imu: ImuCalib = calibration["imu"]
    t_ns: Int64[ndarray, " n"] = data["t_ns"]
    if limit is not None:
        t_ns = t_ns[:limit]
    counts: Int64[ndarray, " n"] = data["imu_counts"]
    offsets: Int64[ndarray, " n_plus_one"] = np.concatenate([[0], np.cumsum(counts)])
    return Framesets(
        images=data["images"],
        t_ns=t_ns,
        imu_offsets=offsets,
        imu_t=data["imu_t"],
        imu_gyro=data["imu_g"],
        imu_accel=data["imu_a"],
        safe_radius=float(data["safe_radius"]),
        cameras=cameras,
        imu=imu,
    )


def run_lane(lane: Lane, round_index: int, framesets: Framesets, config: _core.VioConfig) -> LaneRound:
    """Replay every frameset through a fresh ``Vio`` on ``lane`` and time each call.

    The drive is :class:`slam_rs.tracking.Lockstep`, the one a real replay and
    the V2 gate use, so the D17 hold a refused frameset costs is the hold they
    pay and not a second account of it — including its bound on how deep the
    hold may go. The timings are its own ``elapsed_ms``: the ``track`` call
    alone, nothing around it.

    Args:
        lane: Which frontend backend to build.
        round_index: Round this belongs to, counted from one.
        framesets: The dump to replay.
        config: The parsed VIO config.

    Returns:
        The round's per-call timings, wall time and CPU samples.
    """
    calibration: _core.Calibration = _core.Calibration.from_catalog(framesets.cameras, framesets.imu)
    lockstep: Lockstep = Lockstep(vio=_core.Vio(calibration, config, gpu=lane == "gpu"))

    sampler: CpuSampler = CpuSampler()
    sampler.start()
    started: float = time.monotonic()
    for index in range(len(framesets.t_ns)):
        low, high = int(framesets.imu_offsets[index]), int(framesets.imu_offsets[index + 1])
        frameset: Frameset = Frameset(
            t_ns=int(framesets.t_ns[index]),
            images=[np.ascontiguousarray(framesets.images[index, camera]) for camera in range(framesets.images.shape[1])],
            imu=ImuStream(
                t_ns=framesets.imu_t[low:high],
                gyro_rad_s=np.ascontiguousarray(framesets.imu_gyro[low:high]),
                accel_m_s2=np.ascontiguousarray(framesets.imu_accel[low:high]),
            ),
            ground_truth=None,
        )
        for _held, _result in lockstep.push(frameset):
            pass
    wall_s: float = time.monotonic() - started
    sampler.stop.set()
    sampler.join(timeout=1.0)
    return LaneRound(
        lane=lane,
        round_index=round_index,
        track_ms=np.asarray(lockstep.elapsed_ms, dtype=np.float64),
        wall_s=wall_s,
        cpu_pct=np.asarray(sampler.samples or [0.0], dtype=np.float64),
    )


def main(config: Config) -> None:
    """Run the interleaved schedule and print one row per round plus a summary.

    RESULT and BEST carry the profile; CONFIG records the resolved text SHA-256
    once for all lanes and rounds.

    Args:
        config: Parsed CLI options.

    Raises:
        ValueError: If ``--lanes`` names no backend to measure, or if the dump
            was decoded at a different optical-flow safe radius than the config
            sets, which would compare two frontends rather than two lanes.
    """
    # Before the affinity call and before the dump: an empty selection measured
    # nothing, printed a zero-lane header and then reached `config.lanes[0]` —
    # the lane every ratio is read against — as an `IndexError`, which the shim
    # does not turn into a sentence (S25 review).
    if not config.lanes:
        raise ValueError(f"--lanes named no backend to measure; the lanes are {', '.join(get_args(Lane))}")
    if config.pin_core is not None:
        os.sched_setaffinity(0, {config.pin_core})
    framesets: Framesets = load_framesets(config.dump, config.limit)
    resolved_config: str = profiled_config_text(config.config, config.profile)
    vio_config: _core.VioConfig = _core.VioConfig.from_json(resolved_config)
    if vio_config.optical_flow_image_safe_radius != framesets.safe_radius:
        raise ValueError(
            f"the dump was decoded at safe radius {framesets.safe_radius} and {config.config} sets {vio_config.optical_flow_image_safe_radius}"
        )
    print(
        f"# {len(framesets.t_ns)} framesets x {framesets.images.shape[1]} cameras, "
        f"lanes {'/'.join(config.lanes)}, {config.rounds} rounds, "
        f"core {os.sched_getaffinity(0) if config.pin_core is None else config.pin_core}"
    )

    print(f"CONFIG profile={config.profile} config_sha256={config_text_sha256(resolved_config)}")
    measured: dict[Lane, list[LaneRound]] = {lane: [] for lane in config.lanes}
    for round_index, lane in interleave(config.lanes, config.rounds):
        entry: LaneRound = run_lane(lane, round_index, framesets, vio_config)
        measured[lane].append(entry)
        print(
            f"RESULT label={config.label} lane={lane} profile={config.profile} round={round_index} "
            f"n={len(entry.track_ms)} wall={entry.wall_s:.1f} "
            f"median={entry.median_ms:.3f} mean={entry.track_ms.mean():.3f} "
            f"p95={np.percentile(entry.track_ms, 95):.3f} max={entry.track_ms.max():.3f} "
            f"cpu_pct_median={statistics.median(entry.cpu_pct):.1f}",
            flush=True,
        )

    best: dict[Lane, float] = {lane: best_median_ms(rounds) for lane, rounds in measured.items()}
    reference: float = best[config.lanes[0]]
    for lane in config.lanes:
        medians: str = " / ".join(f"{entry.median_ms:.3f}" for entry in measured[lane])
        print(
            f"BEST lane={lane} profile={config.profile} best_median={best[lane]:.3f} rounds={medians} vs_{config.lanes[0]}={reference / best[lane]:.3f}x"
        )
