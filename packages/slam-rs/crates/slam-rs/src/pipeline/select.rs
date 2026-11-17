//! The one frontend-backend choice that every slam-rs entry point makes.
//!
//! The Rust CLI, the Python bindings and through them every Python tool ask
//! [`select_backend`] which frontend to build. `Auto` is their default: the GPU
//! frontend when this build carries it and the host's device passes the
//! capability probes, otherwise the CPU frontend with one warning that names
//! the reason. An explicit `Gpu` fails instead of falling back.
//!
//! Capability validation has one owner,
//! `kornia_staging_gpu::runtime::default_capabilities`: it validates the
//! default device once per process, and the GPU frontend's construction reads
//! the same cached result. Allocation and tracking failures happen after the
//! choice and are never a reason to fall back.
//!
//! The same call picks the global BA's PCG lane (DESIGN §7.3, Q2): by default
//! it follows the frontend. On a GPU frontend, [`probe_back_end`] adds a small
//! GBA round trip on top of the cached validation; if only that fails, the GBA
//! runs on the CPU with one warning. `--ba-backend cpu|gpu` (benches, tests)
//! chooses it outright, and an explicit `gpu` fails instead of falling back.

#[cfg(doc)]
use crate::Vio;
use crate::{Backend, VioError};

/// The backend a caller asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendRequest {
    /// The GPU frontend when [`gpu_probe`] passes, otherwise the CPU frontend
    /// with one warning on stderr. The default of every slam-rs tool.
    #[default]
    Auto,
    /// The CPU frontend.
    Cpu,
    /// The GPU frontend; [`select_backend`] returns the probe's error if it
    /// cannot run here.
    Gpu,
}

impl std::str::FromStr for BackendRequest {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(format!(
                "unknown backend {other:?}: expected auto, cpu or gpu"
            )),
        }
    }
}

/// Where the global BA's PCG runs, as a caller asks (`--ba-backend`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GbaRequest {
    /// The frontend's lane: the GPU when the frontend runs there and
    /// [`probe_back_end`] passes, otherwise the CPU. The default.
    #[default]
    Follow,
    /// The CPU.
    Cpu,
    /// The GPU; [`select_backend`] returns the probe's error if it cannot run here.
    Gpu,
}

impl std::str::FromStr for GbaRequest {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "follow" => Ok(Self::Follow),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            other => Err(format!(
                "unknown GBA backend {other:?}: expected follow, cpu or gpu"
            )),
        }
    }
}

/// The lanes of one pipeline: the frontend's and the global BA's PCG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Backends {
    /// The frontend.
    pub frontend: Backend,
    /// The global BA's PCG.
    pub gba: Backend,
}

impl Backend {
    /// The lane name the tools print and the gate records: `cpu` or `gpu`.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }
}

/// The selection rule over one capability probe.
///
/// The probe runs at most once (only for `Auto` or `Gpu`), and the fallback
/// notice is written at most once, however many callers and threads ask.
/// Generic over the probe and the notice so that tests can count both.
struct Selector<E, P, N> {
    probe: P,
    notice: N,
    result: std::sync::OnceLock<Result<(), E>>,
    noticed: std::sync::Once,
}

impl<E, P, N> Selector<E, P, N>
where
    E: Clone,
    P: Fn() -> Result<(), E>,
    N: Fn(&E),
{
    const fn new(probe: P, notice: N) -> Self {
        Self {
            probe,
            notice,
            result: std::sync::OnceLock::new(),
            noticed: std::sync::Once::new(),
        }
    }

    fn probed(&self) -> Result<(), E> {
        self.result.get_or_init(|| (self.probe)()).clone()
    }

    fn select(&self, request: BackendRequest) -> Result<Backend, E> {
        match request {
            BackendRequest::Cpu => Ok(Backend::Cpu),
            BackendRequest::Gpu => self.probed().map(|()| Backend::Gpu),
            BackendRequest::Auto => match self.probed() {
                Ok(()) => Ok(Backend::Gpu),
                Err(reason) => {
                    self.noticed.call_once(|| (self.notice)(&reason));
                    Ok(Backend::Cpu)
                }
            },
        }
    }
}

/// Why the GPU frontend cannot run in this process.
#[cfg(feature = "gpu-wgpu")]
type ProbeError = kornia_staging_gpu::runtime::GpuError;

/// Why the GPU frontend cannot run in this process: this build has none.
#[cfg(not(feature = "gpu-wgpu"))]
type ProbeError = NoGpuFeature;

/// A build without the `gpu-wgpu` feature.
#[cfg(not(feature = "gpu-wgpu"))]
#[derive(Debug, Clone, Copy)]
struct NoGpuFeature;

#[cfg(not(feature = "gpu-wgpu"))]
impl std::fmt::Display for NoGpuFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        VioError::GpuUnavailable.fmt(f)
    }
}

fn device_probe() -> Result<(), ProbeError> {
    #[cfg(feature = "gpu-wgpu")]
    {
        kornia_staging_gpu::runtime::default_capabilities().map(|_| ())
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        Err(NoGpuFeature)
    }
}

/// The back end's probe: the cached device validation, then the GBA round
/// trip (`kornia_staging_gpu::linalg::probe_pcg`, which holds its own
/// exclusive sections).
fn back_end_probe() -> Result<(), ProbeError> {
    #[cfg(feature = "gpu-wgpu")]
    {
        kornia_staging_gpu::runtime::default_capabilities()?;
        kornia_staging_gpu::linalg::probe_pcg(&kornia_staging_gpu::runtime::gpu_client()?)
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        Err(NoGpuFeature)
    }
}

fn into_vio_error(error: ProbeError) -> VioError {
    #[cfg(feature = "gpu-wgpu")]
    {
        VioError::Frontend(error.into())
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        let NoGpuFeature = error;
        VioError::GpuUnavailable
    }
}

fn write_fallback_notice(reason: &ProbeError) {
    eprintln!(
        "slam-rs: WARNING: tracking on the CPU because the GPU frontend cannot run here: {}. \
         Pass `--backend cpu` to choose the CPU lane without this warning.",
        into_vio_error(*reason)
    );
}

fn write_gba_notice(reason: &ProbeError) {
    eprintln!(
        "slam-rs: WARNING: the global BA runs on the CPU because its GPU lane cannot run here: {}. \
         Pass `--ba-backend cpu` to choose the CPU lane without this warning.",
        into_vio_error(*reason)
    );
}

/// The selector type over the real device probe and stderr notice.
type ProcessSelector = Selector<ProbeError, fn() -> Result<(), ProbeError>, fn(&ProbeError)>;

/// The process's selector over the real device probe.
static SELECTOR: ProcessSelector = Selector::new(device_probe, write_fallback_notice);

/// The process's selector for the GBA lane, over the back end's probe.
static GBA_SELECTOR: ProcessSelector = Selector::new(back_end_probe, write_gba_notice);

/// Both lanes over two selectors: the frontend's choice, then the GBA's,
/// which follows the frontend unless asked outright.
fn select_pair<E, P, N, Q, M>(
    frontend: &Selector<E, P, N>,
    gba: &Selector<E, Q, M>,
    request: BackendRequest,
    gba_request: GbaRequest,
) -> Result<Backends, E>
where
    E: Clone,
    P: Fn() -> Result<(), E>,
    N: Fn(&E),
    Q: Fn() -> Result<(), E>,
    M: Fn(&E),
{
    let frontend: Backend = frontend.select(request)?;
    Ok(Backends {
        frontend,
        gba: gba_lane(gba, frontend, gba_request)?,
    })
}

/// The GBA lane for a chosen frontend.
fn gba_lane<E, Q, M>(
    gba: &Selector<E, Q, M>,
    frontend: Backend,
    request: GbaRequest,
) -> Result<Backend, E>
where
    E: Clone,
    Q: Fn() -> Result<(), E>,
    M: Fn(&E),
{
    gba.select(match (request, frontend) {
        (GbaRequest::Follow, Backend::Gpu) => BackendRequest::Auto,
        (GbaRequest::Follow, Backend::Cpu) | (GbaRequest::Cpu, _) => BackendRequest::Cpu,
        (GbaRequest::Gpu, _) => BackendRequest::Gpu,
    })
}

/// The GBA lane that follows a frontend chosen without [`select_backend`]
/// ([`Vio::with_backend`], the RoboCap paths): the GPU when the frontend runs
/// there and [`probe_back_end`] passes, otherwise the CPU (one warning).
pub fn follow_frontend(frontend: Backend) -> Backend {
    gba_lane(&GBA_SELECTOR, frontend, GbaRequest::Follow).unwrap_or(Backend::Cpu)
}

/// Check that the GPU frontend can run on this host.
///
/// Reads the default device's cached capability validation (storage and
/// subgroup probes, run once per process).
///
/// # Errors
///
/// [`VioError::GpuUnavailable`] in a build without the `gpu-wgpu` feature;
/// otherwise the device error: no adapter, a client that panicked, or a device
/// that fails the storage or subgroup probe.
pub fn gpu_probe() -> Result<(), VioError> {
    SELECTOR.probed().map_err(into_vio_error)
}

/// Check that the global BA's PCG can run on the GPU here: the cached device
/// validation (it does not probe the device again), then a GBA round trip
/// (8 poses, 200 landmark blocks, 3 CG iterations within 1e-4 of the CPU
/// lane). Runs once per process.
///
/// # Errors
///
/// As [`gpu_probe`], or the round trip's disagreement.
pub fn probe_back_end() -> Result<(), VioError> {
    GBA_SELECTOR.probed().map_err(into_vio_error)
}

/// Resolve a [`BackendRequest`] and a [`GbaRequest`] to the lanes that
/// [`Vio::with_backends`] builds.
///
/// `Auto` never fails: when [`gpu_probe`] fails it returns [`Backend::Cpu`] and
/// writes one warning to stderr, once per process. `Cpu` never probes. The GBA
/// lane follows the frontend by default; on a GPU frontend whose
/// [`probe_back_end`] fails, the GBA runs on the CPU with one warning.
///
/// # Errors
///
/// For `Gpu` only: the error of [`gpu_probe`]; for [`GbaRequest::Gpu`]: the
/// error of [`probe_back_end`].
pub fn select_backend(request: BackendRequest, gba: GbaRequest) -> Result<Backends, VioError> {
    select_pair(&SELECTOR, &GBA_SELECTOR, request, gba).map_err(into_vio_error)
}

/// How many device capability probes this process has run (0 in a build
/// without the GPU feature). The unit test tier asserts it stays at 0.
pub fn device_probes_run() -> usize {
    #[cfg(feature = "gpu-core")]
    {
        kornia_staging_gpu::runtime::capability_probes_run()
    }
    #[cfg(not(feature = "gpu-core"))]
    {
        0
    }
}

/// How many GBA round trips ([`probe_back_end`]) this process has run, counted
/// apart from [`device_probes_run`] (0 in a build without the GPU feature).
pub fn gba_probes_run() -> usize {
    #[cfg(feature = "gpu-core")]
    {
        kornia_staging_gpu::linalg::pcg_round_trips_run()
    }
    #[cfg(not(feature = "gpu-core"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendRequest, Backends, GbaRequest, Selector, select_backend, select_pair};
    use crate::Backend;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The failure classes a real probe reports, as a fake can.
    const FAILURES: [&str; 4] = [
        "wgpu found no vulkan adapter",
        "building the wgpu client panicked",
        "this runtime does not store 16-bit elements",
        "GPU KLT requires working power-of-two subgroups",
    ];

    /// A selector over a fake probe, counting probe runs and notices.
    fn counting(
        outcome: Result<(), &'static str>,
        probes: &AtomicUsize,
        notices: &AtomicUsize,
    ) -> Selector<&'static str, impl Fn() -> Result<(), &'static str>, impl Fn(&&'static str)> {
        Selector::new(
            move || {
                probes.fetch_add(1, Ordering::SeqCst);
                outcome
            },
            move |_: &&'static str| {
                notices.fetch_add(1, Ordering::SeqCst);
            },
        )
    }

    #[test]
    fn requests_parse_from_their_cli_names() {
        assert_eq!("auto".parse(), Ok(BackendRequest::Auto));
        assert_eq!("cpu".parse(), Ok(BackendRequest::Cpu));
        assert_eq!("gpu".parse(), Ok(BackendRequest::Gpu));
        assert!("cuda".parse::<BackendRequest>().is_err());
        assert_eq!(BackendRequest::default(), BackendRequest::Auto);
    }

    #[test]
    fn a_passing_probe_selects_the_gpu_once_and_says_nothing() {
        let (probes, notices) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let selector = counting(Ok(()), &probes, &notices);
        for _ in 0..3 {
            assert_eq!(selector.select(BackendRequest::Auto), Ok(Backend::Gpu));
            assert_eq!(selector.select(BackendRequest::Gpu), Ok(Backend::Gpu));
            assert_eq!(selector.select(BackendRequest::Cpu), Ok(Backend::Cpu));
        }
        assert_eq!(probes.load(Ordering::SeqCst), 1);
        assert_eq!(notices.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn every_failure_class_falls_back_on_auto_and_fails_gpu() {
        for failure in FAILURES {
            let (probes, notices) = (AtomicUsize::new(0), AtomicUsize::new(0));
            let selector = counting(Err(failure), &probes, &notices);
            for _ in 0..3 {
                assert_eq!(
                    selector.select(BackendRequest::Auto),
                    Ok(Backend::Cpu),
                    "{failure}"
                );
                assert_eq!(selector.select(BackendRequest::Gpu), Err(failure));
            }
            assert_eq!(probes.load(Ordering::SeqCst), 1, "{failure}");
            assert_eq!(notices.load(Ordering::SeqCst), 1, "{failure}");
        }
    }

    #[test]
    fn a_cpu_request_never_probes() {
        let (probes, notices) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let selector = counting(Err(FAILURES[0]), &probes, &notices);
        assert_eq!(selector.select(BackendRequest::Cpu), Ok(Backend::Cpu));
        assert_eq!(probes.load(Ordering::SeqCst), 0);
        assert_eq!(notices.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn concurrent_auto_requests_probe_once_and_warn_once() {
        let (probes, notices) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let selector = counting(Err(FAILURES[3]), &probes, &notices);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..16 {
                        assert_eq!(selector.select(BackendRequest::Auto), Ok(Backend::Cpu));
                    }
                });
            }
        });
        assert_eq!(probes.load(Ordering::SeqCst), 1);
        assert_eq!(notices.load(Ordering::SeqCst), 1);
    }

    /// The GBA lane over two fake probes: the frontend's and the back end's.
    fn lanes(
        frontend: Result<(), &'static str>,
        gba: Result<(), &'static str>,
        request: BackendRequest,
        gba_request: GbaRequest,
        counts: &[AtomicUsize; 4],
    ) -> Result<Backends, &'static str> {
        let [front_probes, front_notices, gba_probes, gba_notices] = counts;
        select_pair(
            &counting(frontend, front_probes, front_notices),
            &counting(gba, gba_probes, gba_notices),
            request,
            gba_request,
        )
    }

    fn zero() -> [AtomicUsize; 4] {
        std::array::from_fn(|_| AtomicUsize::new(0))
    }

    fn loads(counts: &[AtomicUsize; 4]) -> [usize; 4] {
        std::array::from_fn(|i| counts[i].load(Ordering::SeqCst))
    }

    #[test]
    fn gba_requests_parse_from_their_cli_names() {
        assert_eq!("follow".parse(), Ok(GbaRequest::Follow));
        assert_eq!("cpu".parse(), Ok(GbaRequest::Cpu));
        assert_eq!("gpu".parse(), Ok(GbaRequest::Gpu));
        assert!("auto".parse::<GbaRequest>().is_err());
        assert_eq!(GbaRequest::default(), GbaRequest::Follow);
    }

    #[test]
    fn the_gba_lane_follows_the_frontend() {
        let gpu = Backends {
            frontend: Backend::Gpu,
            gba: Backend::Gpu,
        };
        let cpu = Backends {
            frontend: Backend::Cpu,
            gba: Backend::Cpu,
        };
        let counts = zero();
        let auto = lanes(
            Ok(()),
            Ok(()),
            BackendRequest::Auto,
            GbaRequest::Follow,
            &counts,
        );
        assert_eq!(auto, Ok(gpu));
        assert_eq!(loads(&counts), [1, 0, 1, 0]);
        // A CPU frontend, chosen or fallen back to, keeps the GBA on the CPU
        // without running the back-end probe.
        let counts = zero();
        let chosen = lanes(
            Ok(()),
            Ok(()),
            BackendRequest::Cpu,
            GbaRequest::Follow,
            &counts,
        );
        assert_eq!(chosen, Ok(cpu));
        assert_eq!(loads(&counts), [0, 0, 0, 0]);
        let counts = zero();
        let fallen = lanes(
            Err("no adapter"),
            Ok(()),
            BackendRequest::Auto,
            GbaRequest::Follow,
            &counts,
        );
        assert_eq!(fallen, Ok(cpu));
        assert_eq!(loads(&counts), [1, 1, 0, 0]);
    }

    #[test]
    fn a_failing_back_end_probe_moves_only_the_gba_and_warns_once() {
        let counts = zero();
        let selected = lanes(
            Ok(()),
            Err("the GBA's device PCG disagrees"),
            BackendRequest::Auto,
            GbaRequest::Follow,
            &counts,
        );
        assert_eq!(
            selected,
            Ok(Backends {
                frontend: Backend::Gpu,
                gba: Backend::Cpu
            })
        );
        assert_eq!(loads(&counts), [1, 0, 1, 1]);
        // An explicit GPU lane fails with the probe's reason instead.
        let counts = zero();
        let explicit = lanes(
            Ok(()),
            Err("the GBA's device PCG disagrees"),
            BackendRequest::Auto,
            GbaRequest::Gpu,
            &counts,
        );
        assert_eq!(explicit, Err("the GBA's device PCG disagrees"));
        assert_eq!(loads(&counts), [1, 0, 1, 0]);
    }

    #[test]
    fn explicit_gba_lanes_ignore_the_frontend() {
        let counts = zero();
        let cpu = lanes(
            Ok(()),
            Ok(()),
            BackendRequest::Gpu,
            GbaRequest::Cpu,
            &counts,
        );
        assert_eq!(cpu.map(|b| b.gba), Ok(Backend::Cpu));
        assert_eq!(loads(&counts)[2], 0, "a CPU GBA never probes");
        let counts = zero();
        let gpu = lanes(
            Ok(()),
            Ok(()),
            BackendRequest::Cpu,
            GbaRequest::Gpu,
            &counts,
        );
        assert_eq!(
            gpu,
            Ok(Backends {
                frontend: Backend::Cpu,
                gba: Backend::Gpu
            })
        );
    }

    /// Without the GPU feature, auto falls back and an explicit GPU request fails.
    #[cfg(not(feature = "gpu-wgpu"))]
    #[test]
    fn a_cpu_only_build_falls_back_on_auto_and_refuses_gpu() {
        assert!(matches!(
            select_backend(BackendRequest::Auto, GbaRequest::Follow),
            Ok(Backends {
                frontend: Backend::Cpu,
                gba: Backend::Cpu
            })
        ));
        assert!(matches!(
            select_backend(BackendRequest::Gpu, GbaRequest::Follow),
            Err(crate::VioError::GpuUnavailable)
        ));
        assert!(matches!(
            select_backend(BackendRequest::Cpu, GbaRequest::Gpu),
            Err(crate::VioError::GpuUnavailable)
        ));
        assert_eq!(super::device_probes_run(), 0);
    }

    /// With the GPU feature, auto never fails and agrees with an explicit GPU request.
    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn auto_never_fails_and_agrees_with_gpu() {
        let auto: Result<Backends, _> = select_backend(BackendRequest::Auto, GbaRequest::Follow);
        assert!(auto.is_ok(), "auto returned {auto:?}");
        let gpu_ok: bool = select_backend(BackendRequest::Gpu, GbaRequest::Cpu).is_ok();
        assert_eq!(
            auto.as_ref().ok().map(|b| b.frontend) == Some(Backend::Gpu),
            gpu_ok
        );
        // A GPU frontend that passed keeps the GBA there too when its probe passes.
        if let Ok(backends) = auto {
            assert_eq!(
                backends.gba == Backend::Gpu,
                backends.frontend == Backend::Gpu && super::probe_back_end().is_ok()
            );
        }
    }
}
