//! The canonical Rerun viewer: stock Rerun plus every extension in [`EXTENSIONS`].
//!
//! The patched `re_viewer` (`patches/re_viewer-0.44.0-extension-hooks.patch`) calls two hooks that every viewer goes
//! through, native, headless and web: one adjusts the device request, one runs on each new `App`. A third one, in
//! `re_viewer_context`, lets the fallbacks an extension declares in `replaces_fallbacks` be replaced without a warning.
//! [`install_hooks`] points them at [`EXTENSIONS`].

use canonical_viewer_extension::{Extension, wgpu};

/// Every extension in the canonical viewer. Adding one = its crate in Cargo.toml + one line here.
pub const EXTENSIONS: &[Extension] = &[gsplat_viewer::setup::EXTENSION];

/// Point the patched `re_viewer`'s hooks at [`EXTENSIONS`]; call once, before the viewer starts.
pub fn install_hooks() {
    let device_set = re_viewer::extension_hooks::DEVICE_DESCRIPTOR
        .set(request_device_needs)
        .is_ok();
    let setup_set = re_viewer::extension_hooks::SETUP_APP
        .set(register_extensions)
        .is_ok();
    let fallbacks_set =
        re_viewer::external::re_viewer_context::extension_hooks::INTENDED_FALLBACK_REPLACEMENT
            .set(|view, component| declares_fallback(EXTENSIONS, view.as_str(), component.as_str()))
            .is_ok();
    if !(device_set && setup_set && fallbacks_set) {
        re_log::error!(
            "canonical viewer: the re_viewer extension hooks were already set; extensions may be missing"
        );
    }
}

fn request_device_needs(
    adapter: &wgpu::Adapter,
    rerun_descriptor: wgpu::DeviceDescriptor<'static>,
) -> wgpu::DeviceDescriptor<'static> {
    with_device_needs(
        EXTENSIONS,
        &adapter.features(),
        &adapter.limits(),
        rerun_descriptor,
    )
}

/// Whether an extension in `extensions` declares that it replaces the stock fallback for `component` in `view`.
fn declares_fallback(extensions: &[Extension], view: &str, component: &str) -> bool {
    extensions
        .iter()
        .any(|extension| extension.replaces_fallbacks.contains(&(view, component)))
}

fn register_extensions(app: &mut re_viewer::App) {
    for extension in EXTENSIONS {
        match (extension.register)(app) {
            Ok(()) => re_log::info!("canonical viewer: extension {} registered", extension.name),
            Err(err) => re_log::error!(
                "canonical viewer: extension {} failed to register: {err:#}",
                extension.name
            ),
        }
    }
}

/// Apply each extension's device needs, in order, to Rerun's device descriptor.
///
/// Only an extension's features and limits are taken, and only when they keep everything already requested and stay
/// within the adapter's offer; otherwise they are logged and dropped, so one extension can never break the device or
/// set device-wide policy (memory hints, label) for the others.
pub fn with_device_needs(
    extensions: &[Extension],
    offered_features: &wgpu::Features,
    offered_limits: &wgpu::Limits,
    mut descriptor: wgpu::DeviceDescriptor<'static>,
) -> wgpu::DeviceDescriptor<'static> {
    for extension in extensions {
        let Some(device_needs) = extension.device_needs else {
            continue;
        };
        let mut wanted = descriptor.clone();
        device_needs(offered_features, offered_limits, &mut wanted);
        let keeps_requested = wanted
            .required_features
            .contains(descriptor.required_features)
            && descriptor
                .required_limits
                .check_limits(&wanted.required_limits);
        let within_offer = offered_features.contains(wanted.required_features)
            && wanted.required_limits.check_limits(offered_limits);
        if keeps_requested && within_offer {
            descriptor.required_features = wanted.required_features;
            descriptor.required_limits = wanted.required_limits;
        } else {
            re_log::error!(
                "canonical viewer: extension {} asked for device features or limits that {}; ignoring its device needs",
                extension.name,
                if keeps_requested {
                    "the adapter does not offer"
                } else {
                    "drop what Rerun or an earlier extension requested"
                }
            );
        }
    }
    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extension(
        name: &'static str,
        device_needs: fn(&wgpu::Features, &wgpu::Limits, &mut wgpu::DeviceDescriptor<'static>),
    ) -> Extension {
        Extension {
            name,
            device_needs: Some(device_needs),
            replaces_fallbacks: &[],
            register: |_| Ok(()),
        }
    }

    fn rerun_descriptor() -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            ..Default::default()
        }
    }

    fn offer() -> (wgpu::Features, wgpu::Limits) {
        (
            wgpu::Features::SUBGROUP | wgpu::Features::TIMESTAMP_QUERY,
            wgpu::Limits::default(),
        )
    }

    #[test]
    fn each_replaced_fallback_has_one_owner() {
        let mut claims: Vec<_> = EXTENSIONS
            .iter()
            .flat_map(|extension| extension.replaces_fallbacks.iter())
            .collect();
        let claimed = claims.len();
        claims.sort_unstable();
        claims.dedup();
        assert_eq!(
            claims.len(),
            claimed,
            "two extensions replace the same stock fallback"
        );
    }

    #[test]
    fn only_declared_fallbacks_are_intended() {
        const OWNER: Extension = Extension {
            name: "owner",
            device_needs: None,
            replaces_fallbacks: &[("3D", "EyeControls3D:position")],
            register: |_| Ok(()),
        };
        assert!(declares_fallback(&[OWNER], "3D", "EyeControls3D:position"));
        assert!(!declares_fallback(&[OWNER], "2D", "EyeControls3D:position"));
        assert!(!declares_fallback(
            &[OWNER],
            "3D",
            "EyeControls3D:look_target"
        ));
    }

    #[test]
    fn only_features_and_limits_are_taken() {
        let (features, limits) = offer();
        let policy = extension("policy", |offered, _, wanted| {
            wanted.required_features |= *offered & wgpu::Features::SUBGROUP;
            wanted.memory_hints = wgpu::MemoryHints::MemoryUsage;
            wanted.label = Some("policy");
        });
        let merged = with_device_needs(&[policy], &features, &limits, rerun_descriptor());
        assert_eq!(merged.required_features, wgpu::Features::SUBGROUP);
        assert!(matches!(
            merged.memory_hints,
            wgpu::MemoryHints::Performance
        ));
        assert_eq!(merged.label, None);
    }

    #[test]
    fn extension_names_are_unique() {
        let mut names: Vec<_> = EXTENSIONS.iter().map(|extension| extension.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), EXTENSIONS.len());
    }

    #[test]
    fn offered_needs_compose_in_list_order() {
        let (features, limits) = offer();
        let subgroups = extension("subgroups", |offered, _, wanted| {
            wanted.required_features |= *offered & wgpu::Features::SUBGROUP;
        });
        let storage = extension("storage", |_, offered, wanted| {
            wanted.required_limits.max_compute_workgroup_storage_size =
                offered.max_compute_workgroup_storage_size;
        });
        let merged = with_device_needs(
            &[subgroups, storage],
            &features,
            &limits,
            rerun_descriptor(),
        );
        assert_eq!(merged.required_features, wgpu::Features::SUBGROUP);
        assert_eq!(
            merged.required_limits.max_compute_workgroup_storage_size,
            limits.max_compute_workgroup_storage_size
        );
    }

    #[test]
    fn needs_beyond_the_offer_are_dropped() {
        let (features, limits) = offer();
        let unoffered = extension("unoffered", |_, _, wanted| {
            wanted.required_features |= wgpu::Features::SHADER_F64;
        });
        let too_large = extension("too-large", |_, offered, wanted| {
            wanted.required_limits.max_buffer_size = offered.max_buffer_size + 1;
        });
        let merged = with_device_needs(
            &[unoffered, too_large],
            &features,
            &limits,
            rerun_descriptor(),
        );
        assert_eq!(merged.required_features, wgpu::Features::empty());
        assert_eq!(merged.required_limits, rerun_descriptor().required_limits);
    }

    #[test]
    fn needs_that_drop_earlier_requests_are_dropped() {
        let (features, limits) = offer();
        let subgroups = extension("subgroups", |_, _, wanted| {
            wanted.required_features |= wgpu::Features::SUBGROUP;
        });
        let clears = extension("clears", |_, _, wanted| {
            wanted.required_features = wgpu::Features::TIMESTAMP_QUERY;
        });
        let lowers = extension("lowers", |_, _, wanted| {
            wanted.required_limits.max_texture_dimension_2d = 1;
        });
        let merged = with_device_needs(
            &[subgroups, clears, lowers],
            &features,
            &limits,
            rerun_descriptor(),
        );
        assert_eq!(merged.required_features, wgpu::Features::SUBGROUP);
        assert_eq!(merged.required_limits, rerun_descriptor().required_limits);
    }
}
