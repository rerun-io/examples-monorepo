//! The one frontend-backend choice that every slam-rs entry point makes.
//!
//! The Rust CLI, the Python bindings and through them every Python tool ask
//! [`select_backend`] which frontend to build. `Auto` is their default: the GPU
//! frontend when this build carries it and the host has a usable adapter,
//! otherwise the CPU frontend with one warning that names the reason. An
//! explicit `Gpu` fails instead of falling back.

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

/// Check that the GPU frontend can run on this host.
///
/// The check builds the wgpu client and runs the storage and subgroup probes
/// that the GPU frontend needs. The result is cached for the process, as the
/// wgpu client itself is.
///
/// # Errors
///
/// [`VioError::GpuUnavailable`] in a build without the `gpu-wgpu` feature;
/// otherwise the device error: no adapter, a client that panicked, or a device
/// that fails the storage or subgroup probe.
pub fn gpu_probe() -> Result<(), VioError> {
    #[cfg(feature = "gpu-wgpu")]
    {
        static PROBE: std::sync::OnceLock<Result<(), kornia_staging_gpu::runtime::GpuError>> =
            std::sync::OnceLock::new();
        PROBE
            .get_or_init(crate::gpu::probe_device)
            .map_err(|error| VioError::Frontend(error.into()))
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        Err(VioError::GpuUnavailable)
    }
}

/// Resolve a [`BackendRequest`] to the [`Backend`] that [`Vio::with_backend`]
/// builds.
///
/// `Auto` never fails: when [`gpu_probe`] fails it returns [`Backend::Cpu`] and
/// writes one warning to stderr, once per process.
///
/// # Errors
///
/// For `Gpu` only: the error of [`gpu_probe`].
pub fn select_backend(request: BackendRequest) -> Result<Backend, VioError> {
    match request {
        BackendRequest::Cpu => Ok(Backend::Cpu),
        BackendRequest::Gpu => gpu_probe().map(|()| Backend::Gpu),
        BackendRequest::Auto => match gpu_probe() {
            Ok(()) => Ok(Backend::Gpu),
            Err(reason) => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!(
                        "slam-rs: WARNING: tracking on the CPU because the GPU frontend cannot run here: {reason}. \
                         Pass `--backend cpu` to choose the CPU lane without this warning."
                    );
                });
                Ok(Backend::Cpu)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendRequest, select_backend};
    use crate::Backend;

    #[test]
    fn requests_parse_from_their_cli_names() {
        assert_eq!("auto".parse(), Ok(BackendRequest::Auto));
        assert_eq!("cpu".parse(), Ok(BackendRequest::Cpu));
        assert_eq!("gpu".parse(), Ok(BackendRequest::Gpu));
        assert!("cuda".parse::<BackendRequest>().is_err());
        assert_eq!(BackendRequest::default(), BackendRequest::Auto);
    }

    #[test]
    fn cpu_is_always_granted() {
        assert!(matches!(
            select_backend(BackendRequest::Cpu),
            Ok(Backend::Cpu)
        ));
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
    }

    /// With the GPU feature, auto and an explicit GPU request agree with the probe.
    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn auto_follows_the_probe() {
        let probe_ok: bool = super::gpu_probe().is_ok();
        let auto: Backend = select_backend(BackendRequest::Auto).unwrap_or(Backend::Cpu);
        assert_eq!(auto == Backend::Gpu, probe_ok);
        assert_eq!(select_backend(BackendRequest::Gpu).is_ok(), probe_ok);
    }
}
