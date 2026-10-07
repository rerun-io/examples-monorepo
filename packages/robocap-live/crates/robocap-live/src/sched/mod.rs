//! The runtime's threads and queues: source -> downsample -> {SLAM, hands} -> output, in parallel, with bounded queues.
//!
//! ```text
//!  source (A55) --framesets--> downsample (A55 pool) --+--> SLAM (A76, latest-wins) --poses--> PoseStore
//!        \--IMU (lossless)--------------------------------/                                       |
//!                                                      +--> hands (A55 + NPU) <--newest pose <= t-+
//!                                                      |        |
//!                                                      +--------+--> output (record JSONL, Rerun logger)
//! ```
//!
//! Two queue policies:
//! - **realtime** (live, or `--realtime` replay): every queue is latest-wins: when a stage is behind, the oldest waiting item
//!   is dropped and counted, so no stage ever stalls the source. SLAM takes the newest frameset (and the rate cap);
//!   pose progress follows consumption, so lag does not make hands wait an extra frameset;
//! - **lossless** (replay as fast as possible): queues block instead, every frameset goes through every stage, SLAM selects
//!   framesets by their timestamps only, and hands wait for publication of their frameset's estimate, including the final flush.
//!
//! Every stage records its wall time per frameset; a one-line summary is printed every second and a final summary at the end.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use nalgebra::Isometry3;
use serde::Serialize;

use crate::downsample::{SmallImagePool, SmallImages, small_images};
use crate::frame::NUM_CAMERAS;
use crate::hands::{HandFrameResult, HandInputs, HandTracking, HandsError};
use crate::nets::{HandNets, NetsError};
use crate::slam::{ReferencePoses, SlamConfig, SlamLane, SlamMode, SlamPose, SlamStatus};
use crate::source::FrameSource;
use crate::source::SourceError;
use kornia_staging_sensors::SourceEvent;
use kornia_staging_sensors::{Frameset};
use kornia_staging_sensors::imu::{CombinedImuSample};

mod queue;
mod record;
mod slam_stage;
mod stats;
mod system;

// Retain over four seconds at the 2 kHz IMU rate. A gap
// beyond three seconds resets SLAM, so older samples cannot bridge that world.
const LIVE_IMU_CAPACITY: usize = 8192;

use queue::{PoseStore, QueuePolicy, StageQueue, lock};
pub use record::{FrameTimings, FramesetSink, OutputRecord, RecordWriter, SinkError};
use slam_stage::slam_loop;
pub use stats::{Counters, Summary};
use stats::{STAGES, Stage, Stats};
use system::{CoreCpu, ThreadCpu};
pub use system::{
    CoreLayout, CpuFreqCap, PowerSample, detect_big_little, parse_cpu_list, pin_current_thread,
    set_uclamp_min, soc_temperature_c,
};

/// Errors of the runtime.
#[derive(Debug, thiserror::Error)]
pub enum SchedError {
    /// The source failed.
    #[error("source: {0}")]
    Source(#[from] SourceError),
    /// A stage failed.
    #[error("{stage}: {message}")]
    Stage {
        /// Which stage.
        stage: &'static str,
        /// What happened.
        message: String,
    },
    /// A CPU list did not parse.
    #[error("bad CPU list {0:?} (use e.g. 4-7, 0,2,3 or none)")]
    CpuList(String),
    /// A thread could not be started or panicked.
    #[error("thread {0}: {1}")]
    Thread(&'static str, String),
}

/// How long the stages get to end after a stop before `run` gives up on them.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

fn stage_error(stage: &'static str, error: impl std::fmt::Display) -> SchedError {
    SchedError::Stage {
        stage,
        message: error.to_string(),
    }
}

/// Where the hands stage gets its tracker and networks.
pub struct HandsStage {
    /// The tracker.
    pub tracker: Box<dyn HandTracking>,
    /// The networks.
    pub nets: Box<dyn HandNets>,
    /// Rebuilds the networks after a failure (an RKNN run timeout soft-resets the NPU): `None` = keep using the old ones.
    pub nets_factory: Option<NetsFactory>,
}

/// Builds a fresh [`HandNets`] (new runtime contexts).
pub type NetsFactory = Box<dyn FnMut() -> Result<Box<dyn HandNets>, NetsError> + Send>;

/// How to run the pipeline.
pub struct PipelineConfig {
    /// Block instead of dropping (replay as fast as possible); otherwise latest-wins everywhere (live, `--realtime`).
    pub lossless: bool,
    /// Where poses come from.
    pub slam_mode: SlamMode,
    /// SLAM rate, threads, reset gap.
    pub slam: SlamConfig,
    /// Reference poses for `SlamMode::Reference`.
    pub reference: Option<ReferencePoses>,
    /// The hands stage, when on.
    pub hands: Option<HandsStage>,
    /// How long hands wait for SLAM to reach their frameset in realtime runs.
    pub hands_wait: Duration,
    /// How long SLAM waits for the IMU to cover a frameset in realtime runs.
    pub imu_wait: Duration,
    /// Cameras whose small image is made (`None` = all present).
    pub small_cameras: Option<Vec<usize>>,
    /// Downsample worker threads.
    pub downsample_threads: usize,
    /// A76 cores for SLAM (`None` = no pinning).
    pub cpus_big: Option<Vec<usize>>,
    /// A55 cores for everything else (`None` = no pinning).
    pub cpus_little: Option<Vec<usize>>,
    /// Cores for the downsample pool (`None` = no pinning).
    pub cpus_downsample: Option<Vec<usize>>,
    /// Cores for the hands stage (`None` = no pinning).
    pub cpus_hands: Option<Vec<usize>>,
    /// `uclamp.min` for the SLAM thread and its frontend pool (`None` = the kernel's default).
    pub slam_uclamp_min: Option<u32>,
    /// `uclamp.min` for the hands thread and the threads it spawns.
    pub hands_uclamp_min: Option<u32>,
    /// Stop after this long (from the first event).
    pub duration: Option<Duration>,
    /// Print the one-line summary every second.
    pub print_every_second: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            lossless: false,
            slam_mode: SlamMode::Off,
            slam: SlamConfig::default(),
            reference: None,
            hands: None,
            hands_wait: Duration::ZERO,
            imu_wait: Duration::from_millis(50),
            small_cameras: None,
            downsample_threads: 2,
            cpus_big: None,
            cpus_little: None,
            cpus_downsample: None,
            cpus_hands: None,
            slam_uclamp_min: None,
            hands_uclamp_min: None,
            duration: None,
            print_every_second: false,
        }
    }
}

/// Final numbers of a run.
#[derive(Debug, Default, Serialize)]
pub struct RunSummary {
    /// Live clock health and recovery counters; absent for replay.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub capture: Option<crate::source::CaptureSnapshot>,
    /// Wall seconds from the first to the last frameset arrival.
    pub seconds: f64,
    /// Framesets from the source.
    pub framesets: usize,
    /// Framesets per second from the source.
    pub source_fps: f64,
    /// SLAM steps per second.
    pub slam_hz: f64,
    /// Actual SLAM lane after startup fallback; `None` when SLAM is off or uses reference poses.
    pub slam_lane: Option<SlamLane>,
    /// Actual one-frame lag setting, when SLAM runs.
    pub slam_frontend_lag: Option<bool>,
    /// Actual frontend worker count, when SLAM runs.
    pub slam_threads: Option<usize>,
    /// The run's event counters.
    #[serde(flatten)]
    pub counters: Counters,
    /// Framesets SLAM never saw (latest-wins drops).
    pub slam_dropped: u64,
    /// Hands-stage drops.
    pub hands_dropped: u64,
    /// Output-stage drops.
    pub output_dropped: u64,
    /// Downsample-stage drops.
    pub downsample_dropped: u64,
    /// Per-stage timings, ms.
    pub stages: std::collections::BTreeMap<String, Summary>,
    /// Frames per second through each stage.
    pub stage_fps: std::collections::BTreeMap<String, f64>,
    /// SoC temperature at the end, Celsius.
    pub temperature_c: Option<f64>,
    /// Mean busy % per CPU over the run.
    pub cpu_busy_pct: Vec<f64>,
    /// Mean CPU % (of one core) per thread group over the run.
    pub thread_cpu_pct: std::collections::BTreeMap<String, f64>,
    /// Power-relevant load, once per second.
    pub power: Vec<PowerSample>,
}

struct Downsampled {
    frameset: Frameset,
    small: SmallImages,
    emitted: Instant,
    downsample_ms: f64,
}

/// What the hands stage made of a frameset.
struct HandsOutcome {
    pose: Option<SlamPose>,
    hands: Option<HandFrameResult>,
    timings: FrameTimings,
}

/// A frameset on its way to the output. `outcome` is `None` when it did not go through the hands stage (there is none, or it went
/// around a busy one): the output stage then looks its pose up.
struct HandsDone {
    item: Arc<Downsampled>,
    outcome: Option<HandsOutcome>,
}

/// The queues, poses, counters and stop flag every stage shares.
struct Shared {
    capture: Option<crate::source::CaptureHealth>,
    stats: Stats,
    poses: PoseStore,
    ds: StageQueue<(Frameset, Instant)>,
    slam: StageQueue<Arc<Downsampled>>,
    hands: StageQueue<Arc<Downsampled>>,
    out: StageQueue<HandsDone>,
    stop: Arc<AtomicBool>,
    failed: Mutex<Option<SchedError>>,
}

impl Shared {
    /// Keep the run's first error and stop the run.
    fn fail(&self, error: SchedError) {
        lock(&self.failed).get_or_insert(error);
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Close every queue and the pose store: a producer blocked on a full lossless queue or a consumer waiting on an empty one
    /// wakes up, and the stages drain and end.
    fn close_all(&self) {
        self.ds.close();
        self.slam.close();
        self.hands.close();
        self.out.close();
        self.poses.close();
    }

    /// Count one event in the counter `field` picks and print `message` for the first `first` of them and then every `every`-th.
    fn count_and_report(
        &self,
        field: impl FnOnce(&mut Counters) -> &mut u64,
        first: u64,
        every: u64,
        message: impl FnOnce(u64) -> String,
    ) {
        let count = self.stats.with(|s| {
            let counter = field(&mut s.counters);
            *counter += 1;
            *counter
        });
        if count <= first || (every > 0 && count.is_multiple_of(every)) {
            eprintln!("robocap-live: {}", message(count));
        }
    }
}

/// Start a stage thread on `cpus` with `uclamp_min`. An error of the setup or of `body`, or a panic, goes to [`Shared::fail`],
/// which stops the run: the monitor then closes every queue, so no other stage waits forever on this one.
fn spawn_stage(
    name: &'static str,
    cpus: Option<Vec<usize>>,
    uclamp_min: Option<u32>,
    shared: &Arc<Shared>,
    body: impl FnOnce() -> Result<(), SchedError> + Send + 'static,
) -> Result<thread::JoinHandle<()>, SchedError> {
    let shared = shared.clone();
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let pinned = cpus.map_or(Ok(()), |cpus| pin_current_thread(&cpus));
            if let Some(min) = uclamp_min
                && let Err(error) = set_uclamp_min(min)
            {
                eprintln!("robocap-live: {name}: {error} (continuing without the frequency hint)");
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pinned.and_then(|()| body())
            })) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => shared.fail(error),
                Err(panic) => {
                    let message = panic
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| panic.downcast_ref::<String>().cloned());
                    shared.fail(SchedError::Thread(
                        name,
                        format!("panicked: {}", message.unwrap_or_default()),
                    ));
                }
            }
        })
        .map_err(|e| SchedError::Thread(name, e.to_string()))
}

/// Run the pipeline until the source ends, `stop` is set, or `duration` elapses; every frameset goes to `sinks`.
///
/// # Errors
///
/// The first stage error (the other stages are stopped and joined first).
pub fn run(
    source: Box<dyn FrameSource>,
    mut config: PipelineConfig,
    sinks: Vec<Box<dyn FramesetSink>>,
    stop: Arc<AtomicBool>,
) -> Result<RunSummary, SchedError> {
    let lossless = config.lossless;
    let policy = if lossless {
        QueuePolicy::Block
    } else {
        QueuePolicy::DropOldest
    };
    let depth = |realtime: usize| if lossless { 4 } else { realtime };
    let shared = Arc::new(Shared {
        capture: source.capture_health(),
        stats: Stats::default(),
        poses: PoseStore::new(lossless),
        ds: StageQueue::new(2, policy),
        slam: StageQueue::new(depth(1), policy),
        hands: StageQueue::new(depth(1), policy),
        out: StageQueue::new(depth(2), policy),
        stop,
        failed: Mutex::new(None),
    });
    let hands_on = config.hands.is_some();
    let mut handles = Vec::new();
    // A stage that cannot start stops the run; the monitor then ends the stages that did start.
    if let Err(error) = start_stages(&shared, source, &mut config, sinks, &mut handles) {
        shared.fail(error);
    }
    let monitored = monitor(&shared, &handles, &config, hands_on);
    let mut first_error = monitored
        .stuck
        .then(|| SchedError::Thread("stage", "did not end after the stop".into()));
    for handle in handles {
        if monitored.stuck && !handle.is_finished() {
            continue;
        }
        if handle.join().is_err() {
            first_error.get_or_insert(SchedError::Thread("stage", "panicked".into()));
        }
    }
    if let Some(error) = lock(&shared.failed).take().or(first_error) {
        return Err(error);
    }
    Ok(summary(&shared, monitored))
}

/// Spawn SLAM, hands, output, downsample and source, in that order (consumers first), into `handles`.
fn start_stages(
    shared: &Arc<Shared>,
    source: Box<dyn FrameSource>,
    config: &mut PipelineConfig,
    sinks: Vec<Box<dyn FramesetSink>>,
    handles: &mut Vec<thread::JoinHandle<()>>,
) -> Result<(), SchedError> {
    let turned_180 = source.turned_180();
    let lossless = config.lossless;
    let slam_on = config.slam_mode == SlamMode::On;
    let hands_on = config.hands.is_some();
    // Consumers wait for SLAM to pass their frameset: without a bound in a lossless run with SLAM (the run is deterministic), else
    // up to `hands_wait`.
    let pose_wait = (!lossless || !slam_on).then_some(config.hands_wait);
    // More than the 3 s reset interval at the cap's maximum IMU rate, even at 2 kHz.
    // Replay keeps the original unbounded FIFO semantics and never evicts a sample.
    let imu_tx = Arc::new(StageQueue::<CombinedImuSample>::new(if lossless { usize::MAX } else { LIVE_IMU_CAPACITY }, QueuePolicy::DropOldest));
    let imu_rx = imu_tx.clone();

    // SLAM (A76): built on its own thread after pinning, so the frontend pool inherits the A76 mask.
    if slam_on {
        let (stage, slam, imu_wait) = (shared.clone(), config.slam.clone(), config.imu_wait);
        handles.push(spawn_stage(
            "rl-slam",
            config.cpus_big.clone(),
            config.slam_uclamp_min,
            shared,
            move || {
                let result = slam_loop(&stage, &imu_rx, slam, lossless, imu_wait);
                stage.poses.close();
                stage.slam.close();
                result
            },
        )?);
    }
    if let Some(hands) = config.hands.take() {
        let stage = shared.clone();
        handles.push(spawn_stage(
            "rl-hands",
            config.cpus_hands.clone(),
            config.hands_uclamp_min,
            shared,
            move || hands_loop(&stage, hands, pose_wait, turned_180),
        )?);
    }
    let stage = shared.clone();
    handles.push(spawn_stage(
        "rl-output",
        config.cpus_little.clone(),
        None,
        shared,
        move || output_loop(&stage, sinks, lossless, pose_wait),
    )?);
    let stage = shared.clone();
    let (threads, only, slam_mode, reference) = (
        config.downsample_threads.max(1),
        config.small_cameras.clone(),
        config.slam_mode,
        config.reference.take(),
    );
    let pool_cpus = config.cpus_downsample.clone();
    handles.push(spawn_stage(
        "rl-downsample",
        config.cpus_little.clone(),
        None,
        shared,
        move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("rl-ds-{i}"))
                .start_handler(move |_| {
                    if let Some(cpus) = &pool_cpus
                        && let Err(error) = pin_current_thread(cpus)
                    {
                        eprintln!("robocap-live: rl-ds: {error} (running unpinned)");
                    }
                })
                .build()
                .map_err(|e| stage_error("downsample", e))?;
            downsample_loop(
                &stage,
                &pool,
                only.as_deref(),
                slam_mode,
                reference.as_ref(),
                hands_on,
                turned_180,
            )
        },
    )?);
    let stage = shared.clone();
    let duration = config.duration;
    handles.push(spawn_stage(
        "rl-source",
        config.cpus_little.clone(),
        None,
        shared,
        move || source_loop(&stage, source, slam_on.then_some(imu_tx), duration),
    )?);
    Ok(())
}

/// The hands stage (A55 + NPU): the tracker on each frameset, with the newest pose at or before it.
fn hands_loop(
    shared: &Shared,
    stage: HandsStage,
    pose_wait: Option<Duration>,
    turned_180: [bool; NUM_CAMERAS],
) -> Result<(), SchedError> {
    let HandsStage {
        mut tracker,
        nets,
        mut nets_factory,
    } = stage;
    let mut nets: Option<Box<dyn HandNets>> = Some(nets);
    let mut last_rebuild: Option<Instant> = None;
    while let Some(item) = shared.hands.pop() {
        if nets.is_none()
            && let Some(factory) = nets_factory.as_mut()
            && last_rebuild.is_none_or(|at| at.elapsed() >= Duration::from_secs(1))
        {
            last_rebuild = Some(Instant::now());
            match factory() {
                Ok(fresh) => {
                    shared.stats.with(|s| s.counters.nets_recreated += 1);
                    eprintln!(
                        "robocap-live: hands: networks rebuilt ({})",
                        fresh.describe()
                    );
                    nets = Some(fresh);
                }
                Err(error) => eprintln!(
                    "robocap-live: hands: rebuilding the networks failed: {error} (retrying in 1 s)"
                ),
            }
        }
        let (pose, waited) = shared.poses.pose_for(item.frameset.index, pose_wait);
        shared
            .stats
            .record(Stage::PoseWait, waited.as_secs_f64() * 1e3);
        let world_from_rig = pose.map_or_else(Isometry3::identity, |p| p.world_from_rig);
        let full: [Option<&kornia_staging_sensors::CameraFrame>; NUM_CAMERAS] =
            std::array::from_fn(|c| item.frameset.cameras[c].as_ref());
        let small: [Option<&crate::frame::Luma>; NUM_CAMERAS] =
            std::array::from_fn(|c| item.small[c].as_ref());
        let inputs = HandInputs {
            turned_180,
            index: item.frameset.index,
            t_ns: item.frameset.timestamp_ns,
            full,
            small,
        };
        let started = Instant::now();
        let result = match nets.as_mut() {
            Some(nets) => Some(tracker.step(&inputs, &world_from_rig, nets.as_mut())),
            None => {
                shared.stats.with(|s| s.counters.hands_without_nets += 1);
                None
            }
        };
        let hands_ms = started.elapsed().as_secs_f64() * 1e3;
        shared.stats.record(Stage::Hands, hands_ms);
        let index = item.frameset.index;
        let hands = match result {
            None => None,
            Some(Ok(result)) => Some(result),
            Some(Err(HandsError::Nets(error @ NetsError::Run { .. }))) => {
                // An NPU failure (e.g. an RKNN run timeout and soft reset): skip this frameset's hands, drop the contexts and
                // rebuild them before the next step; SLAM and the output never wait on this. Other errors (a bad input) are not
                // the contexts' fault and take the ordinary error path below.
                shared.count_and_report(|c| &mut c.nets_failures, 20, 100, |n| format!("hands: network failure #{n} on frameset {index}: {error}; rebuilding the networks"));
                if nets_factory.is_some() {
                    nets = None;
                    last_rebuild = None;
                }
                None
            }
            Some(Err(error)) => {
                shared.count_and_report(
                    |c| &mut c.hands_errors,
                    5,
                    0,
                    |_| format!("hands: {error}"),
                );
                None
            }
        };
        let mut timings = FrameTimings::downsampled(item.downsample_ms);
        timings.set_pose(pose.as_ref(), index, waited);
        if let Some(hands) = &hands {
            timings.set_hands(hands_ms, &hands.timings);
        }
        shared.out.push(HandsDone {
            item,
            outcome: Some(HandsOutcome {
                pose,
                hands,
                timings,
            }),
        });
    }
    shared.out.close();
    Ok(())
}

/// The output stage (A55): every frameset, in order, to the sinks.
fn output_loop(
    shared: &Shared,
    mut sinks: Vec<Box<dyn FramesetSink>>,
    lossless: bool,
    pose_wait: Option<Duration>,
) -> Result<(), SchedError> {
    let mut last_index: Option<u64> = None;
    while let Some(HandsDone { item, outcome }) = shared.out.pop() {
        if !lossless && last_index.is_some_and(|last| item.frameset.index <= last) {
            shared.stats.with(|s| s.counters.output_late += 1);
            continue;
        }
        last_index = Some(item.frameset.index);
        let HandsOutcome {
            pose,
            hands,
            mut timings,
        } = outcome.unwrap_or_else(|| {
            // It skipped the hands stage: the output stage is its pose consumer.
            let (pose, waited) = shared.poses.pose_for(item.frameset.index, pose_wait);
            let mut timings = FrameTimings::downsampled(item.downsample_ms);
            timings.set_pose(pose.as_ref(), item.frameset.index, waited);
            HandsOutcome {
                pose,
                hands: None,
                timings,
            }
        });
        timings.pipeline_ms = item.emitted.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let record = OutputRecord {
            frameset: &item.frameset,
            small: &item.small,
            pose: pose.as_ref(),
            hands: hands.as_ref(),
            timings: &timings,
        };
        for sink in sinks.iter_mut() {
            if let Err(error) = sink.frameset(&record) {
                shared.count_and_report(
                    |c| &mut c.output_errors,
                    5,
                    0,
                    |_| format!("output: {error}"),
                );
            }
        }
        shared
            .stats
            .record(Stage::Output, started.elapsed().as_secs_f64() * 1e3);
        shared
            .stats
            .record(Stage::EndToEnd, item.emitted.elapsed().as_secs_f64() * 1e3);
    }
    for sink in sinks.iter_mut() {
        if let Err(error) = sink.finish() {
            shared.fail(stage_error("output", error));
        }
    }
    Ok(())
}

/// The downsample stage (A55 pool): small images, then the fan-out to SLAM (or the reference/identity pose), hands and output.
fn downsample_loop(
    shared: &Shared,
    pool: &rayon::ThreadPool,
    only: Option<&[usize]>,
    slam_mode: SlamMode,
    reference: Option<&ReferencePoses>,
    hands_on: bool,
    turned_180: [bool; NUM_CAMERAS],
) -> Result<(), SchedError> {
    let mut images = SmallImagePool::default();
    while let Some((frameset, emitted)) = shared.ds.pop() {
        let started = Instant::now();
        let small = pool
            .install(|| small_images(&frameset, &turned_180, only, &mut images))
            .map_err(|e| stage_error("downsample", e))?;
        let downsample_ms = started.elapsed().as_secs_f64() * 1e3;
        shared.stats.record(Stage::Downsample, downsample_ms);
        let item = Arc::new(Downsampled {
            frameset,
            small,
            emitted,
            downsample_ms,
        });
        match slam_mode {
            SlamMode::On => {
                shared.slam.push(item.clone());
            }
            SlamMode::Off => shared.poses.publish(SlamPose::untracked(
                item.frameset.index,
                item.frameset.timestamp_ns,
                SlamStatus::Off,
            )),
            SlamMode::Reference => {
                if let Some(world_from_rig) = reference.and_then(|r| r.at(item.frameset.timestamp_ns)) {
                    let untracked = SlamPose::untracked(
                        item.frameset.index,
                        item.frameset.timestamp_ns,
                        SlamStatus::Reference,
                    );
                    shared.poses.publish(SlamPose {
                        world_from_rig,
                        ok: true,
                        ..untracked
                    });
                } else {
                    // An absent catalog pose is a skip, not an identity estimate of this frame. Hold the last reference.
                    shared.poses.progress(item.frameset.index);
                }
            }
        }
        if hands_on {
            // Realtime: a frameset the hands stage is too busy to take still goes to the output, without hands, so the viewer and
            // the record keep every frameset through an NPU stall or a slow acquisition.
            let mut evicted = Vec::new();
            shared.hands.push_evicting(item, &mut evicted);
            for item in evicted {
                shared.stats.with(|s| s.counters.hands_bypassed += 1);
                shared.out.push(HandsDone {
                    item,
                    outcome: None,
                });
            }
        } else {
            shared.out.push(HandsDone {
                item,
                outcome: None,
            });
        }
    }
    shared.slam.close();
    if slam_mode != SlamMode::On {
        shared.poses.close();
    }
    if hands_on {
        shared.hands.close();
    } else {
        shared.out.close();
    }
    Ok(())
}

/// The source stage (A55): framesets to the downsample queue, IMU samples to SLAM (`imu`, when SLAM runs), until the source ends,
/// a stop, or `duration` from the first event.
fn source_loop(
    shared: &Shared,
    mut source: Box<dyn FrameSource>,
    imu: Option<Arc<StageQueue<CombinedImuSample>>>,
    duration: Option<Duration>,
) -> Result<(), SchedError> {
    let mut started: Option<Instant> = None;
    let result = loop {
        if shared.stop.load(Ordering::Relaxed)
            || started
                .zip(duration)
                .is_some_and(|(at, limit)| at.elapsed() >= limit)
        {
            break Ok(());
        }
        let event = match source.next_event() {
            Ok(Some(event)) => event,
            Ok(None) => break Ok(()),
            Err(error) => break Err(SchedError::Source(error)),
        };
        started.get_or_insert_with(Instant::now);
        match event {
            SourceEvent::Imu(sample) => {
                shared.stats.with(|s| {
                    s.counters.imu += 1;
                    s.imu_window += 1;
                });
                if let Some(imu) = &imu {
                    let _ = imu.push(sample);
                    shared.stats.with(|s| s.counters.slam_imu_dropped = imu.dropped());
                }
            }
            SourceEvent::Frameset(frameset) => {
                if let Err(error) = crate::frame::require_cap_slots(&frameset) {
                    break Err(SchedError::Source(SourceError::Frame(error)));
                }
                shared.stats.record(Stage::Source, 0.0);
                shared.ds.push((frameset, Instant::now()));
            }
        }
    };
    // Close the IMU channel and the queue (also after an error), so the stages drain what arrived.
    if let Some(imu) = imu { imu.close(); }
    shared.ds.close();
    result
}

/// What the monitor measured, for the summary.
struct Monitored {
    stuck: bool,
    cpu_totals: Vec<f64>,
    thread_totals: std::collections::BTreeMap<String, f64>,
    samples: usize,
    power: Vec<PowerSample>,
}

/// One line per second until every stage finishes. A stop drains input; a failure closes all queues.
/// At the shutdown deadline, close all queues to wake remaining waiters and report stuck stages.
fn monitor(
    shared: &Shared,
    handles: &[thread::JoinHandle<()>],
    config: &PipelineConfig,
    hands_on: bool,
) -> Monitored {
    let mut thread_cpu = ThreadCpu::default();
    let mut core_cpu = CoreCpu::default();
    let _ = (thread_cpu.sample(), core_cpu.sample());
    let run_started = Instant::now();
    let mut monitored = Monitored {
        stuck: false,
        cpu_totals: Vec::new(),
        thread_totals: Default::default(),
        samples: 0,
        power: Vec::new(),
    };
    let mut last_print = Instant::now();
    let mut stop_seen: Option<Instant> = None;
    let mut aborted = false;
    while handles.iter().any(|h| !h.is_finished()) {
        thread::sleep(Duration::from_millis(50));
        if shared.stop.load(Ordering::Relaxed) {
            if !aborted && lock(&shared.failed).is_some() {
                shared.close_all();
                aborted = true;
                stop_seen = Some(Instant::now());
            }
            let seen = *stop_seen.get_or_insert_with(|| {
                // A normal stop ends input, then lets SLAM drain/flush before hands and output finish.
                // A stage failure needs the abort path to wake producers blocked behind that failed consumer.
                shared.ds.close();
                Instant::now()
            });
            if seen.elapsed() > SHUTDOWN_GRACE {
                shared.close_all();
                let alive: Vec<String> = handles
                    .iter()
                    .filter(|h| !h.is_finished())
                    .map(|h| h.thread().name().unwrap_or("?").to_owned())
                    .collect();
                eprintln!(
                    "robocap-live: stages {alive:?} did not end {} s after the stop; leaving them",
                    SHUTDOWN_GRACE.as_secs()
                );
                monitored.stuck = true;
                break;
            }
        }
        if last_print.elapsed() < Duration::from_secs(1) {
            continue;
        }
        let window = last_print.elapsed().as_secs_f64();
        last_print = Instant::now();
        let cores = core_cpu.sample();
        let threads = thread_cpu.sample();
        let power = PowerSample::read(run_started.elapsed().as_secs_f64());
        monitored.samples += 1;
        monitored.cpu_totals.resize(cores.len(), 0.0);
        for (total, busy) in monitored.cpu_totals.iter_mut().zip(&cores) {
            *total += busy;
        }
        for (name, pct) in &threads {
            *monitored.thread_totals.entry(name.clone()).or_default() += pct;
        }
        let drops = (
            shared.slam.dropped(),
            shared.hands.dropped(),
            shared.out.dropped(),
        );
        let soc_c = soc_temperature_c();
        let line = shared.stats.with(|s| {
            let rate = |stage: Stage| s.window[stage as usize].len() as f64 / window;
            let mean = |stage: Stage| Summary::of(&s.window[stage as usize]);
            let slam = mean(Stage::Slam);
            let ok_pct = if slam.count > 0 { 100.0 * s.slam_ok_window as f64 / slam.count as f64 } else { 0.0 };
            let group = |cpus: &Option<Vec<usize>>| -> String {
                cpus.as_ref()
                    .map(|cpus| {
                        let v: Vec<f64> = cpus.iter().filter_map(|&c| cores.get(c).copied()).collect();
                        format!("{:.0}%", v.iter().sum::<f64>() / v.len().max(1) as f64)
                    })
                    .unwrap_or_else(|| format!("{:.0}%", cores.iter().sum::<f64>() / cores.len().max(1) as f64))
            };
            let mut line = format!(
                "[{:6.1} s] src {:4.1}/s imu {:3.0}/s | ds {:4.1}/s {:4.1} ms | slam {:4.1} Hz {:5.1}/{:5.1} ms ok {:3.0}% drop {} | ",
                run_started.elapsed().as_secs_f64(),
                rate(Stage::Source),
                s.imu_window as f64 / window,
                rate(Stage::Downsample),
                mean(Stage::Downsample).mean,
                rate(Stage::Slam),
                slam.mean,
                slam.p95,
                ok_pct,
                drops.0,
            );
            if hands_on {
                let hands = mean(Stage::Hands);
                line += &format!("hands {:4.1}/s {:5.1} ms drop {} | ", rate(Stage::Hands), hands.mean, drops.1);
            }
            line += &format!(
                "out {:4.1}/s e2e {:5.1} ms drop {} | cpu big {} little {}",
                rate(Stage::Output),
                mean(Stage::EndToEnd).mean,
                drops.2,
                group(&config.cpus_big),
                group(&config.cpus_little),
            );
            if let Some(t) = soc_c {
                line += &format!(" {t:.1} C");
            }
            if let Some(health) = &shared.capture
                && let Ok(json) = serde_json::to_string(&health.snapshot()) {
                line += &format!("{}{json}", crate::log_markers::CAPTURE_HEALTH);
            }
            for window in s.window.iter_mut() {
                window.clear();
            }
            s.imu_window = 0;
            s.slam_ok_window = 0;
            line
        });
        if config.print_every_second {
            eprintln!("{line}");
            if !threads.is_empty() {
                let top: Vec<String> = threads
                    .iter()
                    .take(8)
                    .map(|(name, pct)| format!("{name} {pct:.0}%"))
                    .collect();
                eprintln!("           threads: {}", top.join(", "));
            }
            eprintln!("{}", power.line());
        }
        monitored.power.push(power);
    }
    monitored
}

/// The run's final numbers.
fn summary(shared: &Shared, monitored: Monitored) -> RunSummary {
    let mut summary = shared.stats.with(|s| {
        let seconds = match (s.first_frameset, s.last_frameset) {
            (Some(a), Some(b)) => (b - a).as_secs_f64(),
            _ => 0.0,
        };
        let framesets = s.total[Stage::Source as usize].len();
        let per_second = |n: usize| {
            if seconds > 0.0 {
                (n.saturating_sub(1)) as f64 / seconds
            } else {
                0.0
            }
        };
        RunSummary {
            capture: shared.capture.as_ref().map(|health| health.snapshot()),
            seconds,
            framesets,
            source_fps: per_second(framesets),
            slam_hz: per_second(s.total[Stage::Slam as usize].len()),
            slam_lane: s.slam_lane,
            slam_frontend_lag: s.slam_frontend_lag,
            slam_threads: s.slam_threads,
            counters: s.counters.clone(),
            stages: STAGES
                .iter()
                .filter(|(stage, _)| *stage != Stage::Source)
                .map(|&(stage, name)| (name.to_string(), Summary::of(&s.total[stage as usize])))
                .collect(),
            stage_fps: STAGES
                .iter()
                .map(|&(stage, name)| (name.to_string(), per_second(s.total[stage as usize].len())))
                .collect(),
            ..Default::default()
        }
    });
    summary.slam_dropped = shared.slam.dropped();
    summary.hands_dropped = shared.hands.dropped();
    summary.output_dropped = shared.out.dropped();
    summary.downsample_dropped = shared.ds.dropped();
    summary.temperature_c = soc_temperature_c();
    summary.power = monitored.power;
    if monitored.samples > 0 {
        let samples = monitored.samples as f64;
        summary.cpu_busy_pct = monitored
            .cpu_totals
            .iter()
            .map(|t| (t / samples * 10.0).round() / 10.0)
            .collect();
        summary.thread_cpu_pct = monitored
            .thread_totals
            .into_iter()
            .map(|(k, v)| (k, (v / samples * 10.0).round() / 10.0))
            .collect();
    }
    summary
}

#[cfg(test)]
mod tests;
