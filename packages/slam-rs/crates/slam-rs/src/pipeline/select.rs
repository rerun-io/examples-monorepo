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

/// The selector type over the real device probe and stderr notice.
type ProcessSelector = Selector<ProbeError, fn() -> Result<(), ProbeError>, fn(&ProbeError)>;

/// The process's selector over the real device probe.
static SELECTOR: ProcessSelector = Selector::new(device_probe, write_fallback_notice);

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

/// Resolve a [`BackendRequest`] to the [`Backend`] that [`Vio::with_backend`]
/// builds.
///
/// `Auto` never fails: when [`gpu_probe`] fails it returns [`Backend::Cpu`] and
/// writes one warning to stderr, once per process. `Cpu` never probes.
///
/// # Errors
///
/// For `Gpu` only: the error of [`gpu_probe`].
pub fn select_backend(request: BackendRequest) -> Result<Backend, VioError> {
    SELECTOR.select(request).map_err(into_vio_error)
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

#[cfg(test)]
mod tests {
    use super::{BackendRequest, Selector, select_backend};
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

    /// Without the GPU feature, auto falls back and an explicit GPU request fails.
    #[cfg(not(feature = "gpu-wgpu"))]
    #[test]
    fn a_cpu_only_build_falls_back_on_auto_and_refuses_gpu() {
        assert!(matches!(
            select_backend(BackendRequest::Auto),
            Ok(Backend::Cpu)
        ));
        assert!(matches!(
            select_backend(BackendRequest::Gpu),
            Err(crate::VioError::GpuUnavailable)
        ));
        assert_eq!(super::device_probes_run(), 0);
    }

    /// With the GPU feature, auto never fails and agrees with an explicit GPU request.
    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn auto_never_fails_and_agrees_with_gpu() {
        let auto: Result<Backend, _> = select_backend(BackendRequest::Auto);
        assert!(auto.is_ok(), "auto returned {auto:?}");
        let gpu_ok: bool = select_backend(BackendRequest::Gpu).is_ok();
        assert_eq!(auto.ok() == Some(Backend::Gpu), gpu_ok);
    }
}
