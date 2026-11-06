//! robocap-live command line. `anyhow` is used here only. `robocap-live --help` lists every flag; the two usual runs are
//!
//! ```text
//! robocap-live --source live --nets rknn <models dir> --viewer rerun+http://<host>:9876/proxy      # on a cap
//! robocap-live --source replay <dump dir> --nets ort <models dir> --record <results.jsonl>        # lossless replay
//! ```
//!
//! SIGINT/SIGTERM stop the source; the stages drain, the logger and the record are finished, and a live source restores the
//! devices (trigger stopped, cameras closed, IIO attributes restored) before the process exits. A second signal exits at once,
//! unless it comes within 0.5 s of the first: that is the same stop delivered twice (see `REPEAT_WINDOW_NS`).

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use robocap_live::capture::Cap;
use robocap_live::frame::{NUM_CAMERAS, SMALL_SIZE};
use robocap_live::hands::{self, HandsConfig, ScaleMode};
use robocap_live::log::video::{EncoderKind, encoder_config};
use robocap_live::log::{Logger, LoggerConfig, LoggerSink, VideoMode};
use robocap_live::nets::{HandNets, NetsError, NoNets};
use robocap_live::sched::{self, FramesetSink, HandsStage, PipelineConfig, RecordWriter};
use robocap_live::slam::{
    ReferencePoses, SlamConfig, SlamLane, SlamMode, SlamProfile, parse_override,
};
use robocap_live::source::FrameSource;
use robocap_live::source::replay::{ReplayConfig, ReplaySource, read_reference_poses};

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SlamArg {
    On,
    Off,
    Reference,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum HandsArg {
    /// The hand tracker (needs `--nets`).
    On,
    /// No hands stage.
    Off,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DetNetArg {
    /// Every DetNet camera, in `--detnet-groups` interleaved groups.
    All,
    /// One camera per frameset (ROBUST_TRACKER_CONFIG as handtrack runs it): one group per camera.
    RoundRobin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SmallArg {
    /// Every present camera.
    All,
    /// The four SLAM cameras.
    Slam,
}

/// A comma list of cameras, e.g. `0,1,5`.
#[derive(Clone, Debug)]
struct Cameras(Vec<usize>);

impl FromStr for Cameras {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let cameras = text
            .split(',')
            .map(|c| c.trim().parse::<usize>().ok().filter(|&c| c < NUM_CAMERAS))
            .collect::<Option<Vec<_>>>();
        cameras
            .map(Self)
            .ok_or_else(|| "a comma list of 0..5".into())
    }
}

/// `--scale`: live calibration or a fixed phi.
#[derive(Clone, Copy, Debug)]
enum ScaleArg {
    Auto,
    Fixed(f64),
}

impl FromStr for ScaleArg {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "auto" => Ok(Self::Auto),
            phi => phi
                .parse()
                .map(Self::Fixed)
                .map_err(|_| "expected auto or a number".into()),
        }
    }
}

/// A `--threads-*` core set: `auto` (the stage's default cores), or a CPU list (`none` = no pinning).
#[derive(Clone, Debug)]
enum Cores {
    Auto,
    List(Option<Vec<usize>>),
}

impl FromStr for Cores {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text == "auto" {
            return Ok(Self::Auto);
        }
        sched::parse_cpu_list(text)
            .map(Self::List)
            .map_err(|e| e.to_string())
    }
}

impl Cores {
    /// These cores, or `auto`'s.
    fn or_auto(&self, auto: Option<Vec<usize>>) -> Option<Vec<usize>> {
        match self {
            Self::Auto => auto,
            Self::List(cpus) => cpus.clone(),
        }
    }
}

/// A `uclamp.min` value, 0-1024; `none` (or `off`) = the kernel's default.
#[derive(Clone, Copy, Debug)]
struct Uclamp(Option<u32>);

impl FromStr for Uclamp {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "none" | "off" => Ok(Self(None)),
            value => match value.parse::<u32>() {
                Ok(min) if min <= 1024 => Ok(Self(Some(min))),
                Ok(_) => Err("at most 1024".into()),
                Err(_) => Err("expected 0-1024 or none".into()),
            },
        }
    }
}

impl std::fmt::Display for Uclamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(min) => write!(f, "{min}"),
            None => f.write_str("none"),
        }
    }
}

/// RoboCap live pipeline: cameras + IMU -> slam-rs VIO + hand tracking -> Rerun, live on Cap A or Cap B or replaying a dump.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CaptureArg {
    Copy,
    ZeroCopy,
}

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// `live`, or `replay <dump dir>` (a robocap-live-dump/1 directory).
    #[arg(long, num_args = 1..=2, value_names = ["KIND", "DIR"], required = true)]
    source: Vec<String>,
    /// Replay: pace events by their timestamps (otherwise as fast as possible, lossless).
    #[arg(long)]
    realtime: bool,
    /// Replay: restart at the end, times shifted forward.
    #[arg(long = "loop")]
    looping: bool,
    /// Replay: read the whole clip into RAM first.
    #[arg(long)]
    preload: bool,
    /// Hand networks: `none`, `rknn <models dir>` (the cap's NPU) or `ort <models dir>` (ONNX Runtime).
    #[arg(long, num_args = 1..=2, value_names = ["KIND", "DIR"])]
    nets: Vec<String>,
    /// Hands stage (default: on with nets, off without).
    #[arg(long, value_enum)]
    hands: Option<HandsArg>,
    /// Live Rerun viewer, e.g. rerun+http://<host>:9876/proxy.
    #[arg(long)]
    viewer: Option<String>,
    /// Also save the Rerun stream to this .rrd.
    #[arg(long)]
    save: Option<PathBuf>,
    /// Write one JSON line per frameset (`RecordLine` in sched/record.rs).
    #[arg(long)]
    record: Option<PathBuf>,
    /// Pose source: slam-rs VIO, identity, or the dump's reference poses.
    #[arg(long, value_enum, default_value = "on")]
    slam: SlamArg,
    /// SLAM target rate (framesets are selected by timestamp; when compute is behind, the newest wins).
    #[arg(long, default_value_t = 30.0)]
    slam_hz: f64,
    /// slam-rs VIO profile (measurements on `robocap_live::slam::SlamProfile`). `live` (default): PR #270's fast profile + the
    /// keyframe joint solve deferred beside the next frameset's frontend + 5 LM iterations, inside every accuracy band. `live30`:
    /// live + a 70 px detection grid, about 40 % less frontend work, outside the 10 s clip's band. `fast`: PR #270's.
    #[arg(long, default_value = "live")]
    slam_profile: SlamProfile,
    /// Override a slam-rs VIO configuration key after --slam-profile, e.g. `config.optical_flow_max_iterations=4` or
    /// `port.keyframe_solve_deferred=false` (repeatable).
    #[arg(long = "slam-set", value_name = "KEY=VALUE")]
    slam_set: Vec<String>,
    /// A 4-camera 640x360 Basalt calibration for SLAM. Default: the cap's factory one, chosen by the rig's `device` (live, the
    /// `--rig` file, which must be the identified cap's; in replay, the dump's `rig.json`).
    #[arg(long)]
    slam_calibration: Option<PathBuf>,
    /// SLAM frontend (aarch64: gpu, other hosts: cpu). GPU startup failure falls back to CPU.
    #[arg(long, value_enum, default_value_t = SlamLane::default())]
    slam_lane: SlamLane,
    /// SLAM workers within its two cores (1 on GPU, 2 on CPU).
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=2))]
    slam_threads: Option<u8>,
    /// Cameras the hands may use.
    #[arg(long, default_value = "0,1,2,3,4,5")]
    hand_cameras: Cameras,
    /// DetNet while a hand is untracked: on `all` DetNet cameras (in --detnet-groups groups) or one camera per frameset
    /// (`round-robin`, ROBUST_TRACKER_CONFIG as handtrack runs it; parity with s66-cams4/cams6).
    #[arg(long, value_enum, default_value = "all")]
    detnet: DetNetArg,
    /// Cameras DetNet looks at (default: --hand-cameras).
    #[arg(long)]
    detnet_cameras: Option<Cameras>,
    /// With `--detnet all`: interleaved camera groups, one per frameset (1 = every camera every frameset).
    #[arg(long, default_value_t = 1)]
    detnet_groups: usize,
    /// Hand scale: `auto` (live calibration) or a fixed phi.
    #[arg(long, default_value = "auto")]
    scale: ScaleArg,
    /// Seconds of live scale calibration with `--scale auto`.
    #[arg(long, default_value_t = 10.0)]
    scale_seconds: f64,
    /// Stop after this many seconds.
    #[arg(long)]
    duration: Option<f64>,
    /// Cores for SLAM: a list like 6-7, `none`, or `auto` (the top-capacity cores: one A76 pair = one cpufreq policy).
    #[arg(long, default_value = "auto")]
    threads_a76: Cores,
    /// Cores for the source, output and (by default) hands stages: a list like 0-3, `none`, or `auto` (the little cores).
    #[arg(long, default_value = "auto")]
    threads_a55: Cores,
    /// Cores for the downsample pool: a list, `none`, or `auto` (the other big cores, 4-5 on the RK3588: an A76 streams the
    /// 1080p planes ~6x faster than an A55).
    #[arg(long, default_value = "auto")]
    threads_ds: Cores,
    /// Cores for the hands stage: a list, `none`, or `auto` (= --threads-ds, the other A76 pair on the RK3588: hands at 26 fps
    /// with 1 drop/s there vs 21 fps with 6.5 drops/s on the A55s, Cap B realtime replay).
    #[arg(long, default_value = "auto")]
    threads_hands: Cores,
    /// Cap the A76 cores' cpufreq (`scaling_max_freq`, kHz) for the run, restored on exit (power budget on 5 V USB).
    #[arg(long)]
    max_a76_khz: Option<u32>,
    /// Cap the A55 cores' cpufreq (kHz) for the run, restored on exit.
    #[arg(long)]
    max_a55_khz: Option<u32>,
    /// `uclamp.min` (0-1024) for the SLAM threads: a frequency hint to schedutil while they run, `none` = kernel default.
    /// 1024 took SLAM from 19.8 to 24.6 Hz on a Cap B realtime replay (policy 6 at 2.2 instead of 1.76 GHz), for more power
    /// and heat.
    #[arg(long, default_value = "none")]
    slam_uclamp: Uclamp,
    /// `uclamp.min` (0-1024) for the hands thread and its helpers, `none` = kernel default.
    #[arg(long, default_value = "none")]
    hands_uclamp: Uclamp,
    /// Downsample worker threads.
    #[arg(long, default_value_t = 2)]
    downsample_threads: usize,
    /// Small images for `all` present cameras or only the four `slam` cameras.
    #[arg(long, value_enum, default_value = "all")]
    small: SmallArg,
    /// Realtime runs: how long the hands wait for SLAM to reach their frameset, ms. 0 by default: SLAM takes ~30 ms per
    /// frameset, so a 10 ms wait gave the frameset's own pose on 34 of 3150 hands steps (Cap B) and added 10 ms of latency to all.
    #[arg(long, default_value_t = 0.0)]
    hands_wait_ms: f64,
    /// Live: added to IIO timestamps to put IMU0 on the camera clock (default: DataForge's RoboCap alignment, -14.9 ms, which the
    /// catalog and the replay dumps already carry).
    #[arg(long, default_value_t = robocap_live::capture::DATAFORGE_IMU_TIME_OFFSET_NS, allow_hyphen_values = true)]
    imu_time_offset_ns: i64,
    /// Live: the cap's rig.json (its `device` must be this cap's).
    #[arg(long, default_value = robocap_live::capture::DEFAULT_RIG_PATH)]
    rig: PathBuf,
    /// Live: `zero-copy` (frames are read-only images over the capture buffers) or `copy` (each luma plane copied out).
    #[cfg(target_os = "linux")]
    #[arg(long, default_value = "zero-copy")]
    capture: CaptureArg,
    /// Video in the Rerun stream: `h264` (encoder child processes), `raw` (640x360 luma images, local viewers) or `off`.
    #[arg(long, default_value = if cfg!(target_arch = "aarch64") { "h264" } else { "raw" })]
    video: VideoMode,
    /// H.264 encoder: `mpp` (the cap's hardware mpph264enc), `x264` (ffmpeg) or `openh264` (gst).
    #[arg(long, default_value = if cfg!(target_arch = "aarch64") { "mpp" } else { "x264" })]
    encoder: EncoderKind,
    /// H.264 bit rate per camera.
    #[arg(long, default_value_t = 1_000_000)]
    video_bps: u32,
    /// Cameras whose video goes into the Rerun stream (the others are not encoded: power budget).
    #[arg(long, default_value = "0,1,2,3,4,5")]
    video_cameras: Cameras,
    /// Hand overlays on the camera panes: `fit` (the fitted hands projected through each lens, and DetNet's strongest box),
    /// `debug` (also every KeyNet view: dots coloured by confidence, the crop outline coloured and labelled by what the tracker
    /// did with it; and every DetNet answer from 0.5 up) or `verbose` (also KeyNet's relative depths and the predicted pose).
    /// Default: `debug` in a replay that does not loop, `fit` live and in a looping replay (each overlay is a chunk per frameset
    /// on the preview link).
    #[arg(long)]
    hand_overlays: Option<robocap_live::log::scene::HandOverlays>,
    /// The display asset (.rrd with the static scene + blueprint) sent at the start of each stream.
    #[arg(long)]
    display: Option<PathBuf>,
    /// Write the final summary as JSON.
    #[arg(long)]
    summary_json: Option<PathBuf>,
    /// Write per-frame capture diagnostics to this CSV (live source only; opt-in).
    #[arg(long)]
    frame_csv: Option<PathBuf>,
    /// No per-second lines.
    #[arg(long)]
    quiet: bool,
}

/// CLOCK_MONOTONIC nanoseconds of the first SIGINT/SIGTERM; 0 until one arrives.
static FIRST_SIGNAL_NS: AtomicU64 = AtomicU64::new(0);

/// A signal this soon after the first is the same stop delivered twice, not a second request: coreutils `timeout` (the caps'
/// scripts run robocap-live under it) sends what it forwards to the command and then to its process group.
const REPEAT_WINDOW_NS: u64 = 500_000_000;

extern "C" fn on_signal(_: libc::c_int) {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: clock_gettime is async-signal-safe and `now` is a valid timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) };
    let now_ns = (now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64).max(1);
    if let Err(first_ns) =
        FIRST_SIGNAL_NS.compare_exchange(0, now_ns, Ordering::SeqCst, Ordering::SeqCst)
        && now_ns - first_ns >= REPEAT_WINDOW_NS
    {
        // A second request: give up on the clean shutdown.
        // SAFETY: _exit is async-signal-safe.
        unsafe { libc::_exit(130) };
    }
}

fn install_signal_handlers() {
    for signal in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: the handler only reads the clock, touches an atomic and calls _exit, all async-signal-safe; the sigaction
        // struct is fully initialised (zeroed, then the handler set).
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as *const () as libc::sighandler_t;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
}

fn build_nets(spec: &[String]) -> Result<Option<Box<dyn HandNets>>> {
    let dir = |kind: &str| {
        spec.get(1)
            .map(PathBuf::from)
            .with_context(|| format!("--nets {kind} needs a models directory"))
    };
    match spec.first().map(String::as_str) {
        None | Some("none") => Ok(None),
        Some("rknn") => {
            let dir = dir("rknn")?;
            let nets = robocap_live::nets::rknn::RknnNets::open(&dir)
                .with_context(|| format!("load RKNN models from {}", dir.display()))?;
            Ok(Some(Box::new(nets)))
        }
        #[cfg(feature = "ort")]
        Some("ort") => {
            let dir = dir("ort")?;
            let options = robocap_live::nets::ort::OrtConfig::default();
            let nets = robocap_live::nets::ort::OrtNets::new(&dir, &options)
                .with_context(|| format!("load ONNX models from {}", dir.display()))?;
            Ok(Some(Box::new(nets)))
        }
        #[cfg(not(feature = "ort"))]
        Some("ort") => {
            bail!("--nets ort: this build has no ONNX Runtime (build with --features ort)")
        }
        Some(other) => bail!("--nets {other}: expected none, rknn <dir> or ort <dir>"),
    }
}

/// Rebuilds the `--nets` backend after a network failure (an RKNN run timeout soft-resets the NPU); `None` for `--nets none`.
fn nets_factory(spec: &[String]) -> Option<sched::NetsFactory> {
    let spec = spec.to_vec();
    matches!(spec.first().map(String::as_str), Some("rknn" | "ort")).then(|| {
        let factory: sched::NetsFactory = Box::new(move || match build_nets(&spec) {
            Ok(Some(nets)) => Ok(nets),
            Ok(None) => Err(NetsError::Load {
                what: "nets".into(),
                message: "no backend".into(),
            }),
            Err(error) => Err(NetsError::Load {
                what: format!("{spec:?}"),
                message: format!("{error:#}"),
            }),
        });
        factory
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    install_signal_handlers();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        std::thread::Builder::new()
            .name("rl-signal".into())
            .spawn(move || {
                loop {
                    if FIRST_SIGNAL_NS.load(Ordering::SeqCst) != 0 {
                        eprintln!("{} (signal)", robocap_live::log_markers::STOPPING);
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            })?;
    }

    let layout = sched::detect_big_little();
    let other_big = layout
        .as_ref()
        .map(|layout| {
            layout
                .big
                .iter()
                .copied()
                .filter(|c| !layout.fastest.contains(c))
                .collect::<Vec<_>>()
        })
        .filter(|cores| !cores.is_empty());
    let cpus_big = cli
        .threads_a76
        .or_auto(layout.as_ref().map(|layout| layout.fastest.clone()));
    let cpus_little = cli
        .threads_a55
        .or_auto(layout.as_ref().map(|layout| layout.little.clone()));
    let cpus_downsample = cli
        .threads_ds
        .or_auto(other_big.or_else(|| cpus_little.clone()));
    let cpus_hands = cli.threads_hands.or_auto(cpus_downsample.clone());
    // The frequency caps cover every core of the kind (all four A76s), not only the stage masks: hands and the downsample pool run
    // on the A76 pair SLAM does not use.
    let (all_a76, all_a55) = match &layout {
        Some(layout) => (Some(layout.big.clone()), Some(layout.little.clone())),
        None => (cpus_big.clone(), cpus_little.clone()),
    };
    let mut freq_caps = Vec::new();
    for (khz, cpus, what) in [
        (cli.max_a76_khz, &all_a76, "--max-a76-khz"),
        (cli.max_a55_khz, &all_a55, "--max-a55-khz"),
    ] {
        if let Some(khz) = khz {
            let cpus = cpus.as_ref().with_context(|| format!("{what} needs the core list (big.LITTLE detection or --threads-a76/--threads-a55)"))?;
            freq_caps.push(sched::CpuFreqCap::apply(cpus, khz)?);
        }
    }
    // Everything this thread spawns (capture threads, the stages) starts on the little cores; SLAM re-pins itself to the big ones.
    if let Some(cpus) = &cpus_little {
        sched::pin_current_thread(cpus)?;
    }

    let slam_mode = match cli.slam {
        SlamArg::On => SlamMode::On,
        SlamArg::Off => SlamMode::Off,
        SlamArg::Reference => SlamMode::Reference,
    };
    let kind = cli.source.first().map(String::as_str);
    #[cfg(target_os = "linux")]
    let diagnostic_log = if let Some(path) = &cli.frame_csv {
        anyhow::ensure!(kind == Some("live"), "--frame-csv requires --source live");
        Some(robocap_live::diagnostic::DiagnosticLog::open(path)?)
    } else {
        None
    };
    let mut reference = None;
    let realtime;
    let source: Box<dyn FrameSource> = match kind {
        Some("replay") => {
            let dir = PathBuf::from(
                cli.source
                    .get(1)
                    .context("--source replay needs a dump directory")?,
            );
            let options = ReplayConfig {
                realtime: cli.realtime,
                looping: cli.looping,
                preload: cli.preload,
            };
            let replay = ReplaySource::open(&dir, options, stop.clone())
                .with_context(|| format!("open dump {}", dir.display()))?;
            let meta = replay.meta();
            eprintln!(
                "robocap-live: replay {} ({} framesets, {:.1} s, {}){}{}",
                dir.display(),
                meta.frames,
                (meta.last_t_ns - meta.first_t_ns) as f64 / 1e9,
                meta.segment,
                if cli.realtime {
                    ", realtime"
                } else {
                    ", lossless"
                },
                if cli.looping { ", looping" } else { "" }
            );
            if slam_mode == SlamMode::Reference {
                let poses = read_reference_poses(&dir)?;
                reference = Some(ReferencePoses::new(
                    poses,
                    replay.first_t_ns(),
                    cli.looping.then(|| replay.loop_span_ns()),
                ));
            }
            realtime = cli.realtime;
            Box::new(replay)
        }
        Some("live") => {
            if slam_mode == SlamMode::Reference {
                bail!("--slam reference needs --source replay");
            }
            realtime = true;
            #[cfg(target_os = "linux")]
            {
                live_source(
                    &cli,
                    stop.clone(),
                    diagnostic_log.as_ref().map(|log| log.sink()),
                )?
            }
            #[cfg(not(target_os = "linux"))]
            {
                live_source(&cli, stop.clone())?
            }
        }
        _ => bail!("--source must be `live` or `replay <dump dir>`"),
    };

    let Cameras(hand_cameras) = cli.hand_cameras.clone();
    let detnet_cameras = cli.detnet_cameras.clone().map(|Cameras(cameras)| cameras);
    let detnet_groups = match cli.detnet {
        DetNetArg::All => cli.detnet_groups.max(1),
        DetNetArg::RoundRobin => detnet_cameras.as_ref().unwrap_or(&hand_cameras).len(),
    };
    let scale = match cli.scale {
        ScaleArg::Auto => ScaleMode::Auto {
            seconds: cli.scale_seconds,
        },
        ScaleArg::Fixed(phi) => ScaleMode::Fixed(phi),
    };
    let nets = build_nets(&cli.nets)?;
    let hands_mode = cli.hands.unwrap_or(if nets.is_some() {
        HandsArg::On
    } else {
        HandsArg::Off
    });
    let hands = match hands_mode {
        HandsArg::Off => None,
        HandsArg::On => {
            let nets = nets.unwrap_or_else(|| {
                eprintln!("robocap-live: --hands on with --nets none: the tracker runs but DetNet never sees a hand");
                Box::new(NoNets)
            });
            let config = HandsConfig {
                scale,
                cameras: hand_cameras.clone(),
                detnet_cameras: detnet_cameras.clone(),
                detnet_groups,
                // A lossless replay switches to the calibrated scale on the same frameset every run; live and realtime replays solve
                // in the background (waiting would stall the 30 fps step for the solve, 0.2-1.4 s).
                scale_wait: !realtime,
                acquire_threads: cpus_hands.as_ref().map_or(4, |cpus| cpus.len().max(1)),
                ..HandsConfig::default()
            };
            eprintln!(
                "robocap-live: hands on cameras {hand_cameras:?}, DetNet {:?} on {:?} in {} group(s), scale {:?} (wait {}), nets {}",
                cli.detnet,
                detnet_cameras.as_ref().unwrap_or(&hand_cameras),
                config.detnet_groups,
                config.scale,
                config.scale_wait,
                nets.describe()
            );
            let tracker = hands::new_tracker(source.rig(), config)?;
            Some(HandsStage {
                tracker,
                nets,
                nets_factory: nets_factory(&cli.nets),
            })
        }
    };

    let mut sinks: Vec<Box<dyn FramesetSink>> = Vec::new();
    if let Some(path) = &cli.record {
        sinks.push(Box::new(RecordWriter::create(path)?));
    }
    if cli.viewer.is_some() || cli.save.is_some() {
        let encoder = encoder_config(cli.encoder, SMALL_SIZE, 30, cli.video_bps, 30)?;
        let Cameras(video_cameras) = cli.video_cameras.clone();
        let options = LoggerConfig {
            viewer: cli.viewer.clone(),
            save: cli.save.clone(),
            video: cli.video,
            encoder,
            display: cli.display.clone(),
            video_cameras,
            hand_overlays: cli
                .hand_overlays
                .unwrap_or(if kind == Some("replay") && !cli.looping {
                    robocap_live::log::scene::HandOverlays::Debug
                } else {
                    robocap_live::log::scene::HandOverlays::Fit
                }),
            ..LoggerConfig::default()
        };
        let options_overlays = options.hand_overlays;
        let logger = Logger::new(source.rig(), options).context("start the Rerun logger")?;
        eprintln!(
            "robocap-live: rerun recording {} -> viewer {:?}, save {:?}, video {:?}, hand overlays {:?}",
            logger.recording_id(),
            cli.viewer,
            cli.save,
            cli.video,
            options_overlays
        );
        sinks.push(Box::new(LoggerSink::new(logger)));
    }

    let mut slam = SlamConfig {
        lane: cli.slam_lane,
        hz: cli.slam_hz,
        frontend_threads: cli.slam_threads.map(usize::from),
        profile: cli.slam_profile,
        overrides: cli
            .slam_set
            .iter()
            .map(|setting| parse_override(setting).with_context(|| format!("--slam-set {setting}")))
            .collect::<Result<_>>()?,
        ..SlamConfig::default()
    };
    if slam_mode == SlamMode::On {
        // Live, the rig's device is the identified cap's (the live source refuses another cap's rig).
        let device = &source.rig().device;
        let calibration_from = match &cli.slam_calibration {
            Some(path) => {
                slam.calibration = std::fs::read_to_string(path)
                    .with_context(|| format!("read --slam-calibration {}", path.display()))?;
                path.display().to_string()
            }
            None => {
                let cap = Cap::from_device(device)
                    .with_context(|| format!("no factory SLAM calibration for device {device:?} (the rig's `device`); pass --slam-calibration"))?;
                slam.calibration = cap.slam_calibration().to_string();
                format!("the {device} factory calibration")
            }
        };
        eprintln!(
            "robocap-live: slam profile {}, overrides {:?}, calibration {calibration_from}",
            cli.slam_profile.as_str(),
            slam.overrides
        );
    }
    let small_cameras = match cli.small {
        SmallArg::All => None,
        SmallArg::Slam => Some(robocap_live::frame::SLAM_CAMERAS.to_vec()),
    };
    let config = PipelineConfig {
        lossless: !realtime,
        slam_mode,
        slam,
        reference,
        hands,
        hands_wait: Duration::from_secs_f64(cli.hands_wait_ms.max(0.0) / 1e3),
        small_cameras,
        downsample_threads: cli.downsample_threads,
        cpus_big: cpus_big.clone(),
        cpus_little: cpus_little.clone(),
        cpus_downsample: cpus_downsample.clone(),
        cpus_hands: cpus_hands.clone(),
        slam_uclamp_min: cli.slam_uclamp.0,
        hands_uclamp_min: cli.hands_uclamp.0,
        duration: cli.duration.map(Duration::from_secs_f64),
        print_every_second: !cli.quiet,
        ..PipelineConfig::default()
    };
    eprintln!(
        "robocap-live: slam {:?} at {} Hz (on {:?}, uclamp {}), downsample {} threads on {:?}, hands on {:?} (uclamp {}), other stages on {:?}, {}",
        slam_mode,
        cli.slam_hz,
        cpus_big,
        cli.slam_uclamp,
        cli.downsample_threads,
        cpus_downsample,
        cpus_hands,
        cli.hands_uclamp,
        cpus_little,
        if realtime {
            "realtime (latest-wins queues)"
        } else {
            "lossless (blocking queues)"
        }
    );
    let result = sched::run(source, config, sinks, stop);
    #[cfg(target_os = "linux")]
    let diagnostic_stats = diagnostic_log.map(|log| log.finish()).transpose()?;
    let mut summary = serde_json::to_value(result?)?;
    #[cfg(target_os = "linux")]
    if let Some(stats) = diagnostic_stats {
        summary["frame_diagnostics"] = serde_json::to_value(stats)?;
    }
    let json = serde_json::to_string_pretty(&summary)?;
    let mut brief = serde_json::to_value(&summary)?;
    if let Some(object) = brief.as_object_mut() {
        object.remove("power");
    }
    eprintln!(
        "robocap-live: summary (power samples in --summary-json)\n{}",
        serde_json::to_string(&brief)?
    );
    if let Some(path) = &cli.summary_json {
        std::fs::write(path, json).with_context(|| format!("write {}", path.display()))?;
    }
    drop(freq_caps);
    Ok(())
}

#[cfg(target_os = "linux")]
fn live_source(
    cli: &Cli,
    stop: Arc<AtomicBool>,
    diagnostic: Option<robocap_live::diagnostic::DiagnosticSink>,
) -> Result<Box<dyn FrameSource>> {
    use robocap_live::source::live::{LiveConfig, LiveSource};
    let options = LiveConfig {
        diagnostic,
        rig_path: cli.rig.clone(),
        imu_time_offset_ns: cli.imu_time_offset_ns,
        capture: match cli.capture {
            CaptureArg::Copy => kornia_staging_io::v4l::mplane::CaptureMode::Copy,
            CaptureArg::ZeroCopy => kornia_staging_io::v4l::mplane::CaptureMode::ZeroCopy,
        },
        ..LiveConfig::default()
    };
    eprintln!(
        "robocap-live: live IMU time offset {} ns",
        options.imu_time_offset_ns
    );
    Ok(Box::new(LiveSource::open(options, stop)?))
}

#[cfg(not(target_os = "linux"))]
fn live_source(_: &Cli, _: Arc<AtomicBool>) -> Result<Box<dyn FrameSource>> {
    bail!("--source live runs only on the cap (Linux)")
}
