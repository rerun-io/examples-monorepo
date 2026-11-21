"""Types for the compiled ``slam_rs._core`` extension (built by the ``slam-rs-build`` task).

The signatures here are the only static check on the FFI boundary, so they stay
exact: no ``Any``, and every array carries its dtype.
"""

from collections.abc import Sequence
from typing import ClassVar, Literal

from jaxtyping import Bool, Float32, Float64, Int32, Int64, UInt8, UInt32
from numpy import ndarray
from simplecv.imu_calibration import ImuCalibration

from slam_rs import BackendRequest, GbaRequest, Lane
from slam_rs.catalog_calibration import CameraStatics
from slam_rs.catalog_feed import RigProfile
from slam_rs.catalog_timing import ImuStream
from slam_rs.rig import CameraCalib, ImuCalib

__version__: str

gpu_backend: Literal["wgpu"] | None
"""Which GPU runtime this build's frontend carries, or None for a CPU-only build.

Every pixi build task builds the core with the ``gpu-wgpu`` cargo feature, which
carries ``"wgpu"``. ``None`` is a build without it (the cargo default of the core
crate, kept for wasm, iOS and embedded consumers): there ``backend="auto"`` runs
the CPU and ``backend="gpu"`` is refused.
"""

def device_probes_run() -> int:
    """How many device probes this process has run: capability validations plus GBA round trips.

    Each runs once per process, so this is 0 before the first ``auto`` or
    ``gpu`` request and constant after it. The unit test tier asserts it stays
    at 0.
    """

def reanchor(
    frame_t_ns: Int64[ndarray, " n"],
    frame_poses: Float64[ndarray, "n 7"],
    keyframe_t_ns: Int64[ndarray, " k"],
    keyframe_poses: Float64[ndarray, "k 7"],
) -> Float64[ndarray, "n 7"]:
    """The final trajectory: each raw frame pose re-anchored on the optimized keyframes on both sides.

    A frame between keyframes ``i`` and ``i+1`` blends ``T_i_opt T_i_raw^-1 T_f_raw`` and the same from ``i+1``
    (translation lerp, rotation slerp, by time); a keyframe takes its optimized pose; frames outside the keyframes
    take the nearest keyframe's correction. ``T_i_raw`` is the keyframe's own row in the frames. Poses are
    ``[tx, ty, tz, qx, qy, qz, qw]``; timestamps increase strictly.

    Raises:
        ValueError: Timestamps that do not increase, a keyframe that is not a frame, or mismatched shapes.
    """

def select_backend(request: BackendRequest) -> Lane:
    """Resolve a backend request to the lane :class:`Vio` would run (``slam_rs::select_backend``).

    ``auto`` returns ``"gpu"`` when this build carries the GPU frontend and the
    host has an adapter that passes the storage and subgroup probes; otherwise
    it returns ``"cpu"`` and writes one warning, naming the reason, to stderr
    (once per process). ``"gpu"`` raises ``ValueError`` with that reason instead.
    The probe runs once per process.
    """

class VioStatus:
    """How far the estimator has got.

    ``NeedMoreImu`` refuses a frame for retry. ``Tracking`` carries an estimate.
    ``Buffered`` accepts the first frame in lag mode; submit the next frame or
    call ``flush()``, without retrying it.

    A PyO3 enum, not a ``enum.Enum``: it carries no ``name`` or ``value``, it is
    unhashable, and ``VioStatus(1)`` raises ``TypeError``. It does convert to
    ``int`` and compares equal both to its own variants and to their ordinals.
    """

    NeedMoreImu: ClassVar[VioStatus]
    Tracking: ClassVar[VioStatus]
    Buffered: ClassVar[VioStatus]
    NoVisualFeatures: ClassVar[VioStatus]
    __hash__: ClassVar[None]

    def __int__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class BootstrapAttempt:
    """One solve attempt of the inertial bootstrap (``config.vio_bootstrap``)."""

    @property
    def first_t_ns(self) -> int:
        """First frameset of the window, ns."""

    @property
    def last_t_ns(self) -> int:
        """Last frameset of the window, ns: when the attempt ran."""

    @property
    def frames(self) -> int:
        """Framesets in the window."""

    @property
    def tracks(self) -> int:
        """Tracks handed to the solver."""

    @property
    def tracks_used(self) -> int:
        """Inlier tracks of the solve."""

    @property
    def verdict(self) -> str:
        """The verdict's name: ``Accepted`` or the rejection reason, e.g. ``LowExcitation``."""

    @property
    def accepted(self) -> bool:
        """Whether the solver accepted the window."""

    @property
    def excitation_mps2(self) -> float:
        """Excitation, m/s^2."""

    @property
    def residual_rms_rad(self) -> float | None:
        """RMS bearing error of the solve, rad; None when the attempt reports none."""

    @property
    def condition(self) -> float | None:
        """Condition number of the reduced linear system; None when the attempt reports none."""

    @property
    def gravity_norm_unconstrained(self) -> float | None:
        """Unconstrained ``|g|``, m/s^2; None when the attempt reports none."""

    @property
    def scale_sigma(self) -> float | None:
        """Relative 1-sigma of the speed; None when the attempt reports none."""

    def __repr__(self) -> str: ...

class BootstrapReport:
    """Every attempt of the inertial bootstrap and where it started the estimator."""

    @property
    def attempts(self) -> list[BootstrapAttempt]:
        """The attempts in order."""

    @property
    def start_t_ns(self) -> int | None:
        """The accepted window's first frameset, ns (the estimator's first state); None before a window is accepted."""

    @property
    def accepted_at_t_ns(self) -> int | None:
        """The frameset that completed the accepted window, ns (the first pose published); None before a window is accepted."""

    @property
    def replayed(self) -> int:
        """Framesets replayed into the fresh estimator."""

    def __repr__(self) -> str: ...

class VioResult:
    """What one :meth:`Vio.track` call produced."""

    @property
    def status(self) -> VioStatus: ...
    @property
    def t_ns(self) -> int: ...
    @property
    def world_from_rig(self) -> Float64[ndarray, " 7"] | None:
        """``[tx, ty, tz, qx, qy, qz, qw]``, metres and a unit quaternion (xyzw).

        Only ``Tracking`` supplies a pose; every other status returns ``None``. This is the raw, causal VIO; the
        back end never moves it, and the pose-graph gauge (window fixed) makes it this frame's corrected pose too.
        The corrected history of earlier frames is ``slam_rs.tracking.reanchored(poses, vio.map_snapshot())``.
        """

    @property
    def velocity(self) -> Float64[ndarray, " 3"] | None:
        """Rig velocity in the world frame, m/s."""

    @property
    def gyro_bias(self) -> Float64[ndarray, " 3"] | None:
        """Gyroscope bias estimate, rad/s."""

    @property
    def accel_bias(self) -> Float64[ndarray, " 3"] | None:
        """Accelerometer bias estimate, m/s^2."""

    def __repr__(self) -> str: ...

class LoopCounts:
    """What loop closing has done so far (all zero when it is off)."""

    @property
    def indexed(self) -> int:
        """Keyframes described and added to the database."""

    @property
    def without_images(self) -> int:
        """Queries that found no images for their keyframe."""

    @property
    def images_stripped(self) -> int:
        """Keyframe stamps whose images were dropped before the back end took them (Realtime, behind)."""

    @property
    def with_candidates(self) -> int:
        """Queries that retrieved at least one candidate."""

    @property
    def candidates(self) -> int:
        """Candidates validated."""

    @property
    def pnp_errors(self) -> int:
        """GP3P RANSAC calls that refused their input (a bug upstream of it), each logged."""

    @property
    def verified(self) -> int:
        """Loops that passed verification."""

    @property
    def fused(self) -> int:
        """Loops fused at once."""

    @property
    def needs_pgo(self) -> int:
        """Loops at or above the drift gate: the pose-graph optimization's triggers."""

    @property
    def optimized(self) -> int:
        """Of those, closed by an applied pose-graph optimization."""

class MapResult:
    """The map's keyframes at the end of a sequence (:meth:`Vio.finish_map`).

    Poses are ``T_w_i``, ``[tx, ty, tz, qx, qy, qz, qw]``, on the clock :meth:`Vio.track` was given.
    """

    @property
    def keyframe_t_ns(self) -> Int64[ndarray, " n_keyframes"]:
        """Map keyframe timestamps, oldest first."""

    @property
    def keyframe_poses_before_gba(self) -> Float64[ndarray, "n_keyframes 7"]:
        """``T_w_i`` in the map after loops and PGO, before the global BA (the +LC keyframes).

        The window's own poses are :attr:`MapSnapshot.keyframe_raw_poses`.
        """

    @property
    def keyframe_poses(self) -> Float64[ndarray, "n_keyframes 7"]:
        """``T_w_i`` after the global BA (equal to :attr:`keyframe_poses_before_gba` without it)."""

    @property
    def gba(self) -> GbaRun | None:
        """The global BA, or None when :meth:`Vio.finish_map` ran without it or it failed."""

    @property
    def gba_failure(self) -> str | None:
        """Why the global BA, asked for, failed (``"global BA: <reason>"``), or None; the keyframes are then the map's after loops and PGO."""

    def __repr__(self) -> str: ...

class GbaRun:
    """One global BA over the map: what the problem held, each LM attempt, and the wall of each stage."""

    @property
    def lane(self) -> Lane:
        """Where the PCG ran (always ``"cpu"`` with ``backend.gba_use_double``: the device lane is f32 only)."""

    @property
    def keyframes(self) -> int:
        """Keyframe variables."""

    @property
    def keyframes_without_reprojection(self) -> int:
        """Keyframes under ``backend.gba_min_obs_per_kf``: pose-graph rows only (B22)."""

    @property
    def landmarks(self) -> int:
        """Landmarks in the problem."""

    @property
    def observations(self) -> int:
        """Reprojection observations in the problem."""

    @property
    def pose_graph_rows(self) -> int:
        """Pose-graph rows."""

    @property
    def nonfinite_landmarks(self) -> int:
        """Landmarks left out for want of a finite position."""

    @property
    def initial_cost(self) -> float:
        """``ERROR_VALID`` before the first step."""

    @property
    def final_cost(self) -> float:
        """``ERROR_VALID`` of the returned state."""

    @property
    def termination(self) -> Literal["converged", "max_iterations", "lambda_overflow"]:
        """Why the LM loop stopped."""

    @property
    def iteration_lambda(self) -> Float64[ndarray, " n_attempts"]:
        """LM damping of each attempt."""

    @property
    def iteration_cost(self) -> Float64[ndarray, " n_attempts"]:
        """Cost after each attempt's step; NaN where the step was not finite."""

    @property
    def iteration_accepted(self) -> Bool[ndarray, " n_attempts"]:
        """Whether each attempt's step was kept."""

    @property
    def iteration_cg(self) -> Int64[ndarray, " n_attempts"]:
        """PCG iterations of each attempt."""

    @property
    def iteration_stage_s(self) -> Float64[ndarray, "n_attempts 5"]:
        """Wall seconds per attempt: stage 1 (linearize, QR), stage 2 (damping, preconditioner), PCG, back-substitution, evaluation."""

    @property
    def build_s(self) -> float:
        """Wall seconds turning the map into the problem."""

    @property
    def solve_s(self) -> float:
        """Wall seconds of the solve (centring, LM, uncentring, gauge)."""

    @property
    def apply_s(self) -> float:
        """Wall seconds writing the result into the map."""

    @property
    def gauge_tilt_deg(self) -> float:
        """How far the free gauge had tilted the solution's world z, degrees, before the driver brought it back onto the input poses."""

    @property
    def threads(self) -> int:
        """Worker threads (``backend.gba_threads``)."""

    @property
    def use_double(self) -> bool:
        """Whether it solved in f64 (``backend.gba_use_double``)."""

    def __repr__(self) -> str: ...

class MapKeyframes:
    """The map's keyframe half, which stays small: every keyframe's pose and window state, and the loops. A
    :class:`MapDelta` carries it whole; :attr:`MapSnapshot.map_keyframes` gives the same view of a snapshot."""

    @property
    def t_ns(self) -> Int64[ndarray, " n_keyframes"]:
        """Map keyframe timestamps, oldest first."""

    @property
    def poses(self) -> Float64[ndarray, "n_keyframes 7"]:
        """Keyframe poses ``T_w_i`` as the map holds them (PGO-corrected), ``[tx, ty, tz, qx, qy, qz, qw]``."""

    @property
    def raw_poses(self) -> Float64[ndarray, "n_keyframes 7"]:
        """The window's keyframe poses, before any correction."""

    @property
    def active(self) -> Bool[ndarray, " n_keyframes"]:
        """Whether each keyframe is still in the window."""

    @property
    def loop_edges(self) -> Int64[ndarray, "n_loops 2"]:
        """Verified loops whose keyframes are still in the map, oldest first, as ``[current, candidate]`` timestamps."""

    @property
    def loop_fused(self) -> Bool[ndarray, " n_loops"]:
        """Whether each loop's landmarks were fused."""

class MapDelta:
    """What changed in the map since a revision (:meth:`Vio.map_delta`): the keyframe half, and the points of only the
    keyframes a write moved."""

    @property
    def revision(self) -> int:
        """The map's revision when the delta was taken: the next read's ``since``."""

    @property
    def map_keyframes(self) -> MapKeyframes:
        """Every keyframe's pose and window state, and the loops."""

    @property
    def changed_hosts(self) -> Int64[ndarray, " n_changed"]:
        """Keyframes in the map whose hosted points moved since the read's revision, in id order. A keyframe the reader
        holds that is missing from ``map_keyframes.t_ns`` was removed."""

    @property
    def point_counts(self) -> Int64[ndarray, " n_changed"]:
        """How many of the points below each changed host has, aligned with :attr:`changed_hosts`."""

    @property
    def point_ids(self) -> Int64[ndarray, " n_points"]:
        """The changed hosts' point ids, host by host in :attr:`changed_hosts` order, then by camera and id."""

    @property
    def points_world(self) -> Float32[ndarray, "n_points 3"]:
        """The changed hosts' points in the world frame, metres."""

class MapSnapshot:
    """The persistent keyframe map and the back end's counters, copied out (PR a2).

    Arrays are copies in id order. The keyframe poses are the window's solved (or, once evicted, final) poses as the
    pose-graph optimization (PGO) corrected them; the raw poses are the window's.
    """

    @property
    def mode(self) -> Literal["offline", "realtime"]:
        """Where the back end runs."""

    @property
    def keyframe_t_ns(self) -> Int64[ndarray, " n_keyframes"]:
        """Map keyframe timestamps, oldest first."""

    @property
    def keyframe_poses(self) -> Float64[ndarray, "n_keyframes 7"]:
        """Keyframe poses ``T_w_i`` as the map holds them (PGO-corrected), ``[tx, ty, tz, qx, qy, qz, qw]``."""

    @property
    def keyframe_raw_poses(self) -> Float64[ndarray, "n_keyframes 7"]:
        """The window's keyframe poses, before any correction."""

    @property
    def pgo_epoch(self) -> int:
        """Applied pose-graph optimizations so far (the correction epoch)."""

    @property
    def pgo_runs(self) -> Int64[ndarray, "n_pgo 5"]:
        """Each applied PGO, oldest first: ``[current, candidate, moved, dropped_edges, iterations]``."""

    @property
    def pgo_us(self) -> tuple[int, int]:
        """Wall time of the loops that ran a PGO (graph, solve, map write, fusion): total and slowest, microseconds."""

    @property
    def keyframe_active(self) -> Bool[ndarray, " n_keyframes"]:
        """Whether each keyframe is still in the window."""

    @property
    def point_ids(self) -> Int64[ndarray, " n_points"]:
        """Map point (landmark) ids."""

    @property
    def point_hosts(self) -> Int64[ndarray, " n_points"]:
        """Timestamp of the keyframe hosting each point."""

    @property
    def points_world(self) -> Float32[ndarray, "n_points 3"]:
        """Map points in the world frame, metres."""

    @property
    def covisibility_edges(self) -> Int64[ndarray, "n_edges 2"]:
        """Covisibility edges as keyframe timestamp pairs, smaller first."""

    @property
    def covisibility_weights(self) -> UInt32[ndarray, " n_edges"]:
        """Shared landmarks per covisibility edge."""

    @property
    def pose_graph_edges(self) -> Int64[ndarray, "n_pose_graph_edges 2"]:
        """The edges a pose-graph optimization would take now, ``[from, to]`` keyframe timestamps."""

    @property
    def pose_graph_edge_kinds(self) -> UInt8[ndarray, " n_pose_graph_edges"]:
        """Kind of each pose-graph edge: 0 spanning tree, 1 loop closure, 2 high covisibility."""

    @property
    def map_keyframes(self) -> MapKeyframes:
        """The keyframe half, as a :class:`MapDelta` carries it (a copy of the fields above)."""

    @property
    def loop_edges(self) -> Int64[ndarray, "n_loops 2"]:
        """Verified loops whose keyframes are still in the map, oldest first, as ``[current, candidate]`` timestamps."""

    @property
    def loop_fused(self) -> Bool[ndarray, " n_loops"]:
        """Whether each loop's landmarks were fused, at once or after a PGO; otherwise it was left unapplied."""

    @property
    def loop_drift(self) -> Float64[ndarray, "n_loops 2"]:
        """Each loop's drift when it was verified, ``[metres, degrees]`` of ``T_map^-1 T_pnp``, in :attr:`loop_edges` order."""

    @property
    def loop_counts(self) -> LoopCounts:
        """What loop closing has done so far."""

    @property
    def keyframes(self) -> int:
        """Map keyframes (RootSLAM's ``num_kfs``)."""

    @property
    def landmarks(self) -> int:
        """Map landmarks (``num_lms``), placed in :attr:`points_world` or not."""

    @property
    def observations(self) -> int:
        """(landmark, image) observations (``num_obs``)."""

    @property
    def culled_redundant(self) -> int:
        """Keyframes culled as redundant when they left the window."""

    @property
    def culled_over_budget(self) -> int:
        """Keyframes culled while the map was over its keyframe budget."""

    @property
    def high_covisibility_weight(self) -> int:
        """Shared landmarks at which a covisibility edge is "high"."""

    @property
    def median_keyframe_landmarks(self) -> int:
        """Median number of landmarks a map keyframe sees."""

    @property
    def retained_bytes(self) -> int:
        """The map's payload in bytes (a lower bound on its heap)."""

    @property
    def stamps(self) -> int:
        """Stamps the back end has taken."""

    @property
    def queries_run(self) -> int:
        """Loop queries run."""

    @property
    def queries_dropped(self) -> int:
        """Loop queries dropped because the back end was behind (Realtime only)."""

    @property
    def max_pending_stamps(self) -> int:
        """Most stamps ever waiting for the back-end thread (0 Offline)."""

    @property
    def step_count(self) -> int:
        """Steps timed."""

    @property
    def step_total_us(self) -> int:
        """Wall time of all steps, microseconds."""

    @property
    def step_max_us(self) -> int:
        """Wall time of the slowest step, microseconds."""

    @property
    def query_total_us(self) -> int:
        """Wall time of all loop queries, microseconds."""

    @property
    def query_max_us(self) -> int:
        """Wall time of the slowest loop query, microseconds."""

    @property
    def publish_count(self) -> int:
        """Views the Realtime back-end thread posted for readers (0 in Offline)."""

    @property
    def publish_max_us(self) -> int:
        """Wall time of the slowest posted view, microseconds."""

    @property
    def publish_total_us(self) -> int:
        """Wall time of all posted views, microseconds."""

    def __repr__(self) -> str: ...

class VioSnapshot:
    """The estimator's window, its landmarks and the last measured frame's statistics.

    The window is the 15-dof states followed by the pose-only blocks, each oldest
    first; :attr:`window_keyframe` and :attr:`window_long_term` say what each
    frame is, :attr:`kf_ids` is the keyframes as an id list, and
    :attr:`marginalized` is what the last marginalization removed. Everything is
    a copy, so a snapshot stays valid across the next :meth:`Vio.track`.
    """

    @property
    def t_ns(self) -> int:
        """Frameset timestamp of the newest state in the window."""

    @property
    def window_t_ns(self) -> Int64[ndarray, " n_frames"]: ...
    @property
    def window_poses(self) -> Float64[ndarray, "n_frames 7"]:
        """``[tx, ty, tz, qx, qy, qz, qw]`` per window frame, metres and a unit quaternion (xyzw)."""

    @property
    def window_keyframe(self) -> Bool[ndarray, " n_frames"]:
        """Whether each window frame is a keyframe, as the estimator itself answers it."""

    @property
    def window_long_term(self) -> Bool[ndarray, " n_frames"]:
        """Whether each window frame is a long-term keyframe."""

    @property
    def kf_ids(self) -> Int64[ndarray, " n_keyframes"]:
        """The keyframes' timestamps, oldest first."""

    @property
    def marginalized(self) -> Int64[ndarray, " n_marginalized"]:
        """Frames the last marginalization removed from the window."""

    @property
    def landmark_ids(self) -> Int64[ndarray, " n_landmarks"]:
        """Landmark ids, which are the ids of the keypoints that spawned them."""

    @property
    def landmark_hosts(self) -> Int64[ndarray, " n_landmarks"]:
        """Timestamp of the keyframe hosting each landmark."""

    @property
    def landmark_positions(self) -> Float64[ndarray, "n_landmarks 3"]:
        """Landmark positions in the world frame, metres."""

    @property
    def lm_iterations(self) -> int:
        """Levenberg-Marquardt steps the last frame took; the rejected ones are the rest."""

    @property
    def lm_lambda(self) -> float:
        """Damping the last step solved with; ``0.0`` when no step ran."""

    @property
    def lm_error_before(self) -> float:
        """Total cost before the first step; ``0.0`` when no step ran."""

    @property
    def lm_error_after(self) -> float:
        """Total cost after the last step; ``0.0`` when no step ran."""

    @property
    def num_observations(self) -> int:
        """Landmark observations the window holds."""

    @property
    def frame_update(self) -> str:
        """What the non-keyframe frame update did: ``not_attempted``, ``taken``, or ``declined_<precondition>``."""

    @property
    def gpu_one_wait(self) -> bool:
        """The GPU lane collected this frameset's temporal and stereo results in one device wait; always False on the CPU lane."""

    @property
    def gpu_lookahead(self) -> bool:
        """The GPU lane reused this frameset's lookahead image upload; always False on the CPU lane."""

    @property
    def timings_ms(self) -> dict[str, float]:
        """Wall time each stage took on the last frame, in milliseconds.

        Estimator keys: ``predict``, ``keyframe``, ``optimize``, ``linearize``,
        ``solver``, ``back_substitution``, ``error``, ``marginalize``, ``measure``.
        Frontend keys: ``frontend_pyramid``, ``frontend_detect``,
        ``frontend_track`` (temporal only; formerly included stereo),
        ``frontend_stereo`` (cross-camera matching plus epipolar filter),
        and ``frontend_imu`` (the frontend's motion preintegration).

        ``keyframe`` includes decision and landmark initialization on keyframes,
        and is zero otherwise. ``optimize`` contains ``linearize``, ``solver``,
        ``back_substitution`` and ``error``. ``measure`` contains ``keyframe``,
        ``optimize``, ``marginalize``, state prediction and untimed glue.
        ``predict`` includes that state prediction plus IMU integration before
        ``measure``; its integration portion is outside ``measure``. Timers
        overlap and must not be summed as independent frame costs.
        With frontend lag, frontend timers describe the newer input images;
        estimator timers describe the returned pose.
        """

    def __repr__(self) -> str: ...

class Vio:
    """The VIO pipeline, driven one frameset at a time.

    The default runs frontend and estimator in sequence. ``port.frontend_lag``
    overlaps this frame's frontend with the previous frame's estimator. Both
    schedules repeat bit-identically for the same input and configuration.

    The refusals are the ones :class:`OpticalFlow` makes — a value the core
    refuses is a ``ValueError``, an object of the wrong type a ``TypeError`` and
    an integer outside the parameter's own type an ``OverflowError`` — never a
    Rust panic.
    """

    def __init__(
        self,
        calibration: Calibration,
        config: VioConfig,
        *,
        threads: int = 1,
        max_keypoints: int | None = None,
        backend: BackendRequest = "auto",
        ba_backend: GbaRequest = "follow",
    ) -> None:
        """Build the pipeline for one rig; JSON files arrive through ``from_json``.

        ``ba_backend`` is where :meth:`finish_map`'s global BA runs its PCG:
        ``"follow"`` (the default) takes the frontend's lane, the GPU only when
        the back-end probe (a small GBA round trip) also passes, otherwise the CPU
        with one warning; ``"cpu"`` or ``"gpu"`` choose it outright, for benches
        and tests, and ``"gpu"`` raises ``ValueError`` when it cannot run.

        ``backend`` chooses the frontend through :func:`select_backend`. The
        default ``"auto"`` runs the pyramid, patch build and KLT tracker through
        CubeCL on this host's GPU when it can, and otherwise the CPU port with
        one warning that names the reason. ``"gpu"`` raises ``ValueError``
        instead of falling back: in a core built without the ``gpu-wgpu`` cargo
        feature, and on a host with no adapter, a driver that will not
        initialise, or a device that fails the storage or subgroup probe. None
        of them is a ``PanicException``: a failure the probe catches is a
        ``ValueError`` too, with the runtime's own panic message left on
        stderr. ``"cpu"`` always runs the CPU port.

        ``threads`` is **inert on the GPU lane**: only the CPU patch tracker
        reads it and the GPU tracker holds no work pool. It is accepted rather
        than refused on the GPU lane so one call site can select either lane.

        Raises ``ValueError`` on everything :class:`OpticalFlow` refuses, and on
        a config asking for a path this port does not have:
        ``vio_linearization_type`` other than ``ABS_QR``, ``vio_sqrt_marg``
        false, or ``vio_enforce_realtime``, which Offline mode cannot honour.
        """

    @property
    def camera_count(self) -> int: ...
    def bearings(self, camera: int, pixels: Float32[ndarray, "n 2"]) -> Float64[ndarray, "n 3"]:
        """Unit bearings of camera ``camera``'s pixels in camera 0's frame (rotation only); NaN where a pixel does not unproject."""
    @property
    def tracked_resolution(self) -> list[tuple[int, int]]:
        """``(width, height)`` of the frames each camera is tracked on: the calibration's, or half of it when every camera
        is wider than 2048 px (keypoints are then in the reduced pixels)."""
    @property
    def rolling_shutter_frames(self) -> tuple[int, int] | None:
        """``(corrected, no_coverage)``: frames the rolling-shutter correction moved, and frames left raw because the gyro
        did not cover their time; None when no camera states a readout."""
    @property
    def gpu(self) -> bool:
        """Whether the frontend runs on the GPU."""
    @property
    def backend(self) -> Lane:
        """The lane this pipeline runs: ``"cpu"`` or ``"gpu"``."""
    @property
    def gba_lane(self) -> Lane:
        """Where the global BA's PCG runs: ``"cpu"`` or ``"gpu"``."""
    def push_imu(self, t_ns: int, gyro: Sequence[float], accel: Sequence[float]) -> None:
        """Add one uncalibrated IMU sample.

        Raises ``ValueError`` unless ``t_ns`` strictly follows the last sample
        and every component is finite.
        """

    def push_imu_batch(
        self,
        t_ns: Int64[ndarray, " n_samples"],
        gyro: Float64[ndarray, "n_samples 3"],
        accel: Float64[ndarray, "n_samples 3"],
    ) -> None:
        """Add a batch of samples, all or nothing.

        Raises what :meth:`push_imu` raises, for any sample of the batch, and
        keeps none of it when it does: the estimator is left where it was, so the
        batch can be corrected and pushed again.
        """

    def track(
        self, t_ns: int, images: Sequence[UInt8[ndarray, "h w"]], masks: Sequence[UInt8[ndarray, "mh mw"] | None] | None = None
    ) -> VioResult:
        """Process one frameset of ``camera_count`` C-contiguous ``(h, w)`` uint8 images.

        ``masks``, when given, holds one entry per camera: None or a C-contiguous ``uint8`` segmentation mask at any
        resolution, nonzero = no features there, read through its scale to the camera's tracked image (the mask layer
        contract). The frontend neither detects in nor keeps a track in a detection cell that holds a masked pixel.
        ``masks=None`` masks nothing.

        With ``port.frontend_lag`` enabled, the first call returns ``Buffered``.
        Later calls return the previous frame's estimate with its own ``t_ns``.
        Call ``flush()`` to receive the final pose at the end of the stream.

        Raises ``ValueError`` on a bad dtype, rank or layout, on the wrong number
        of images, unless every image is the size the calibration gives its
        camera, and unless ``t_ns`` is strictly after the last accepted frameset.
        On a GPU lane a device that dies mid-run raises ``ValueError`` here as
        well, never a ``PanicException``.
        """

    def flush(self) -> VioResult | None:
        """Drain the final lagged pose exactly once and finish any deferred solve."""

    def snapshot(self) -> VioSnapshot | None:
        """The last measured frame, or None before the first one.

        With frontend lag its timestamp precedes ``flow_frame()``; frontend
        timers still describe the most recent input images.
        """

    def flow_frame(self) -> FlowFrame | None:
        """The keypoints the frontend tracked on the last accepted frameset, or None before the first."""

    def bootstrap_report(self) -> BootstrapReport | None:
        """The inertial bootstrap's attempts so far (``config.vio_bootstrap``), or None on the stereo start."""

    @property
    def frontend_timings_ms(self) -> dict[str, float]:
        """What the frontend lane cost on the last frameset that ran it, in ms: ``imu``, ``pyramid``, ``detect``,
        ``track`` and ``stereo``. Exists before the estimator has a state, unlike a snapshot's timings."""

    def finish_map(self, run_gba: bool) -> MapResult | None:
        """The map's keyframes at the end of a sequence, after a global BA on the map when ``run_gba`` (PR d2); None when ``backend.enabled`` is off.

        Takes every stamp first. The window is untouched, so the poses :meth:`track` returned stay the raw VIO; the
        map keeps the GBA's poses. Call it after :meth:`flush`. A GBA that fails (a camera model it cannot project,
        the solver's refusal) is reported in :attr:`MapResult.gba_failure`, and the keyframes are then the map's after
        loops and PGO.
        """

    def map_delta(self, since: int) -> MapDelta | None:
        """What changed in the map since revision ``since`` (0: everything): every keyframe pose and the loops, and the
        points of only the keyframes a write moved since. Costs those keyframes' hosted points, not the whole map.

        None with ``backend.enabled`` off. Never waits for the back end: in Realtime it is the newest delta the back-end
        thread posted, taken from the revision the reader held then (a superset of what changed since ``since``), and
        the call asks for a new one from ``since``; :meth:`flush` makes it current.
        """

    def map_snapshot(self) -> MapSnapshot | None:
        """A copy of the persistent keyframe map and the back end's counters at one moment; None when ``backend.enabled`` is off (the default).

        Never waits for the back end. Offline it is current up to the last solved keyframe. In Realtime
        (``backend.mode``) it is the newest copy the back-end thread posted, which can trail the stamps it still
        holds; :meth:`flush` makes it current. The first Realtime call asks the thread to post copies from then on.
        """

    def __repr__(self) -> str: ...

class VioConfig:
    """VIO configuration loaded from the package's JSON inputs."""

    def __init__(self) -> None:
        """Default VIO configuration."""

    @staticmethod
    def from_json(text: str) -> VioConfig:
        """Read a VIO config JSON file; keys it omits keep their default."""

    def to_json(self) -> str:
        """Write the config back in the JSON format slam-rs reads, ``value0`` wrapper and all."""

    optical_flow_image_safe_radius: float
    """Circular mask that hides a fisheye's black corners, in pixels; 0 disables it."""
    vio_bootstrap: str
    """``config.vio_bootstrap``: ``"stereo"`` (default) or ``"inertial"``; anything else raises ``ValueError``."""

    @property
    def backend_enabled(self) -> bool:
        """``backend.enabled``: whether :class:`Vio` builds the persistent keyframe map (off by default)."""

    @property
    def backend_mode(self) -> Literal["offline", "realtime"]:
        """``backend.mode``: ``offline`` steps the map inline (scoring, gates), ``realtime`` on its own thread (live)."""

    def with_backend(self, mode: Literal["off", "offline", "realtime"]) -> VioConfig:
        """The same configuration with the back end off (``backend.mode`` kept) or on in ``mode``; every other key unchanged.

        Raises ``ValueError`` on any other mode.
        """

    def __repr__(self) -> str: ...

class Calibration:
    """Camera-IMU calibration: extrinsics, intrinsics and the noise model."""

    @staticmethod
    def from_json(text: str) -> Calibration:
        """Read a calibration JSON file."""

    @staticmethod
    def from_catalog(cameras: Sequence[CameraCalib], imu: ImuCalib) -> Calibration:
        """Build the calibration from the feed's dataclasses; ``imu.imu_T_body`` is unused."""

    def to_json(self) -> str:
        """Write the calibration back in the JSON format slam-rs reads, ``value0`` wrapper and all."""

    @property
    def camera_count(self) -> int: ...
    @property
    def resolution(self) -> list[tuple[int, int]]:
        """Each camera's ``(width, height)`` in pixels, in rig order."""

    @property
    def camera_models(self) -> list[Literal["pinhole", "kb4", "pinhole-radtan8", "ds", "eucm", "ucm"]]:
        """Camera model names in rig order."""

    @property
    def intrinsics(self) -> list[list[float]]:
        """fx, fy, cx, cy, then model-specific parameters per camera."""

    @property
    def imu_T_cam(self) -> list[Float64[ndarray, "4 4"]]:
        """Copied camera-to-IMU matrices in rig order."""

    @property
    def gravity_ms2(self) -> float:
        """Gravity magnitude, m/s²: the recording's, or 9.81 when it states none."""

    @property
    def readout_s(self) -> list[float | None]:
        """Rolling-shutter readout per camera, seconds; None where the camera states none."""

    @property
    def imu_update_rate(self) -> float:
        """IMU rate in Hz."""

    @property
    def gyro_noise_std(self) -> list[float]:
        """Gyroscope noise density per axis."""

    @property
    def accel_noise_std(self) -> list[float]:
        """Accelerometer noise density per axis."""

    @property
    def gyro_bias_std(self) -> list[float]:
        """Gyroscope bias random walk per axis."""

    @property
    def accel_bias_std(self) -> list[float]:
        """Accelerometer bias random walk per axis."""

    @property
    def cam_time_offset_ns(self) -> int:
        """Camera clock offset in nanoseconds."""

    def __repr__(self) -> str: ...

class FlowFrame:
    """What one :meth:`OpticalFlow.process` call produced, copied out of the frontend."""

    @property
    def t_ns(self) -> int: ...
    @property
    def camera_count(self) -> int: ...
    @property
    def cell_size(self) -> int:
        """``optical_flow_detection_grid_size``: the occupancy cell's side in pixels."""

    @property
    def cell_origin(self) -> tuple[int, int]:
        """``(x_start, y_start)``: the top-left corner of cell ``(0, 0)`` in pixels."""

    def ids(self, camera: int) -> Int64[ndarray, " n_tracks"]:
        """One camera's keypoint ids, ascending."""

    def positions(self, camera: int) -> Float32[ndarray, "n_tracks 2"]:
        """One camera's keypoint positions in pixels."""

    def transforms(self, camera: int) -> Float32[ndarray, "n_tracks 2 3"]:
        """One camera's 2x3 warps ``[[m00, m01, tx], [m10, m11, ty]]``."""

    def occupancy(self, camera: int) -> Int32[ndarray, "rows columns"]:
        """One camera's occupancy counts over camera 0's detection grid."""

    def num_new(self, camera: int) -> int:
        """Ids one camera gained on this frameset: detections plus stereo matches."""

    def num_tracks(self, camera: int) -> int:
        """Keypoints one camera carries."""

    def __repr__(self) -> str: ...

class OpticalFlow:
    """Frame-to-frame optical flow, driven one frameset at a time.

    Pattern 51 only, which is what every shipped config asks for; another
    ``optical_flow_pattern`` raises ``ValueError`` rather than tracking with the
    wrong pattern.

    A value the core refuses is a ``ValueError``, an object of the wrong type a
    ``TypeError``, an integer outside the parameter's own type an
    ``OverflowError`` and a camera past the end of the rig an ``IndexError``.
    None of them is a Rust panic, which would arrive as a
    ``pyo3_runtime.PanicException`` that ``except Exception`` does not catch.
    """

    def __init__(
        self,
        calibration: Calibration,
        config: VioConfig,
        *,
        threads: int = 1,
        max_keypoints: int | None = None,
    ) -> None:
        """Build a frontend for one rig; JSON files arrive through ``from_json``.

        Raises ``ValueError`` on a config the frontend cannot run — another
        pattern or flow type, a detector threshold ladder that never ends or
        never runs, more pyramid levels than the patch buffers allow — on a
        ``max_keypoints`` or ``threads`` past the core's ceiling, and on a
        calibration whose resolution over ``optical_flow_detection_grid_size``
        asks for more occupancy cells than one buffer may hold. Each of those is
        a memory or thread request rather than a plain number.
        """

    @property
    def camera_count(self) -> int: ...
    @property
    def last_keypoint_id(self) -> int: ...
    @property
    def t_ns(self) -> int | None:
        """Timestamp of the last accepted frameset, or None before the first."""

    def process(self, t_ns: int, images: Sequence[UInt8[ndarray, "h w"]]) -> FlowFrame:
        """Track and detect on one frameset of ``camera_count`` C-contiguous images.

        Raises ``ValueError`` on a bad dtype, rank or layout, on the wrong number
        of images, unless every image is the size the calibration gives its
        camera, and unless ``t_ns`` is strictly after the last accepted frameset.
        Any ``int64`` is a timestamp, negative ones included; a refused frameset
        leaves the frontend exactly as the last accepted one did.
        """

    def __repr__(self) -> str: ...


def catalog_camera_calib(index: int, statics: CameraStatics, downscale: int) -> CameraCalib: ...
def catalog_imu_calib(
    calibration: ImuCalibration, imu_t_body: Float64[ndarray, "4 4"], applied_time_shift_ns: int, gravity_ms2: float | None
) -> ImuCalib: ...
def catalog_frame_nearest_anchor(times: Int64[ndarray, " n_frames"], cursor: int, anchor_t_ns: int, tolerance_ns: int) -> tuple[int | None, int]: ...
def catalog_match_framesets(camera_t_ns: Sequence[Int64[ndarray, " n_frames"]], tolerance_ns: int) -> tuple[Int64[ndarray, " n_framesets"], Int64[ndarray, "n_framesets n_cameras"]]: ...
def catalog_pair_imu(gyro_t_ns: Int64[ndarray, " n_gyro"], gyro_rad_s: Float64[ndarray, "n_gyro 3"], accel_t_ns: Int64[ndarray, " n_accel"], accel_m_s2: Float64[ndarray, "n_accel 3"], interpolate: bool) -> ImuStream: ...


def catalog_rig_profile(dataset: str) -> "RigProfile":
    """Return the camera selection and clock rules owned by Rust."""
    ...
