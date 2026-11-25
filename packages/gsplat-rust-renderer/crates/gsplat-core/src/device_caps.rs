//! Compute requirements shared by standalone and viewer devices.
use crate::Error;

// Every raster batch and its workgroup arrays fit this budget (256 splats of 40 bytes plus four scalars); the
// kernels.rs tests check each shader against it.
pub(crate) const REQUIRED_WORKGROUP_STORAGE_BYTES: u32 = 10_256;

fn required_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: 8,
        max_storage_textures_per_shader_stage: 3,
        max_compute_workgroup_size_y: 1,
        max_compute_workgroup_size_z: 1,
        max_compute_workgroups_per_dimension: 65_535,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_storage_size: REQUIRED_WORKGROUP_STORAGE_BYTES,
        ..wgpu::Limits::downlevel_webgl2_defaults()
    }
}

/// Validate the compute limits before creating pipelines. `SUBGROUP` is optional: without it the scan and sort kernels use their workgroup-memory variant.
pub fn check_adapter(limits: &wgpu::Limits) -> Result<(), Error> {
    if !required_limits().check_limits(limits) {
        return Err(Error::Capabilities);
    }
    Ok(())
}

/// Environment variable that withholds `SUBGROUP` from gsplat devices (CLI, native viewer, test device) when
/// set to `1`, to exercise the workgroup-memory kernels on a GPU that has subgroups.
pub const NO_SUBGROUPS_ENV: &str = "GSPLAT_NO_SUBGROUPS";

/// Whether a gsplat device may request `SUBGROUP`: not when [`NO_SUBGROUPS_ENV`] is `1`.
pub fn subgroups_allowed() -> bool {
    !withholds_subgroups(std::env::var_os(NO_SUBGROUPS_ENV).as_deref())
}

fn withholds_subgroups(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

/// Subgroups (unless [`subgroups_allowed`] says no) and timing are enabled when the adapter supports them.
pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let subgroups = if subgroups_allowed() {
        wgpu::Features::SUBGROUP
    } else {
        wgpu::Features::empty()
    };
    adapter.features() & (subgroups | wgpu::Features::TIMESTAMP_QUERY)
}

/// Add only the core's compute requirements to a caller's renderer limits.
/// Storage binding capacity follows the adapter so a large SH buffer can fit.
pub fn compute_limits(available: &wgpu::Limits, mut requested: wgpu::Limits) -> wgpu::Limits {
    requested = requested
        .or_better_values_from(&required_limits())
        .or_worse_values_from(available);
    requested.max_storage_buffer_binding_size = available.max_storage_buffer_binding_size;
    requested.max_buffer_size = requested
        .max_buffer_size
        .max(available.max_storage_buffer_binding_size)
        .min(available.max_buffer_size);
    requested
}

#[cfg(test)]
mod capability_tests {
    #[test]
    fn only_no_subgroups_1_withholds_subgroups() {
        let withholds = |v: Option<&str>| super::withholds_subgroups(v.map(std::ffi::OsStr::new));
        assert!(withholds(Some("1")));
        assert!(!withholds(None));
        assert!(!withholds(Some("0")));
        assert!(!withholds(Some("")));
        assert!(!withholds(Some("true")));
    }

    #[test]
    fn startup_accepts_missing_subgroups_and_reports_weak_limits() {
        assert!(super::check_adapter(&wgpu::Limits::default()).is_ok());
        let error = super::check_adapter(&wgpu::Limits::downlevel_webgl2_defaults()).unwrap_err();
        assert!(error.to_string().contains("eight storage buffers"));
        let mut failures = 0;
        super::required_limits().check_limits_with_fail_fn(
            &wgpu::Limits::downlevel_webgl2_defaults(),
            false,
            |_, _, _| failures += 1,
        );
        assert_eq!(failures, 8);
    }

    #[test]
    fn id_raster_requires_three_storage_textures() {
        let limits = wgpu::Limits {
            max_storage_textures_per_shader_stage: 2,
            ..wgpu::Limits::default()
        };
        assert!(super::check_adapter(&limits).is_err());
    }

    #[test]
    fn compute_limits_preserve_renderer_limits_and_fit_large_bindings() {
        let available = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 2 << 30,
            ..wgpu::Limits::default()
        };
        let requested = wgpu::Limits::downlevel_webgl2_defaults();
        let limits = super::compute_limits(&available, requested.clone());
        assert_eq!(limits.max_buffer_size, 1 << 30);
        assert_eq!(limits.max_storage_buffer_binding_size, 1 << 30);
        assert_eq!(limits.max_bind_groups, requested.max_bind_groups);
        assert_eq!(
            limits.max_texture_dimension_2d,
            requested.max_texture_dimension_2d
        );
        assert_eq!(
            limits.max_uniform_buffers_per_shader_stage,
            requested.max_uniform_buffers_per_shader_stage
        );
        assert_eq!(limits.max_storage_buffers_per_shader_stage, 8);
        assert_eq!(limits.max_storage_textures_per_shader_stage, 3);
        super::check_adapter(&limits).unwrap();
        assert!(limits.check_limits(&available));
        let weak = wgpu::Limits::downlevel_webgl2_defaults();
        assert!(super::compute_limits(&weak, weak.clone()).check_limits(&weak));
    }
}
