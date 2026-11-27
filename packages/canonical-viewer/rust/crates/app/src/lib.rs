//! The canonical Rerun viewer: stock Rerun plus every extension in [`EXTENSIONS`].
//!
//! The patched `re_viewer` (`patches/re_viewer-0.44.0-extension-hooks.patch`) calls two hooks that every viewer goes
//! through, native, headless and web: one adjusts the device request, one runs on each new `App`. [`install_hooks`]
//! points them at [`EXTENSIONS`].

use canonical_viewer_extension::{Extension, wgpu};

/// Every extension in the canonical viewer. Adding one = its crate in Cargo.toml + one line here.
pub const EXTENSIONS: &[Extension] = &[];

/// Point the patched `re_viewer`'s hooks at [`EXTENSIONS`]; call once, before the viewer starts.
pub fn install_hooks() {
    let device_set = re_viewer::extension_hooks::DEVICE_DESCRIPTOR
        .set(request_device_needs)
        .is_ok();
    let setup_set = re_viewer::extension_hooks::SETUP_APP
        .set(register_extensions)
        .is_ok();
    if !(device_set && setup_set) {
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
/// An extension's change is kept only when it keeps everything already requested and stays within the adapter's
/// offer; otherwise it is logged and dropped, so one extension can never break the device for the others.
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
            descriptor = wanted;
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
