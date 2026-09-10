#![allow(clippy::unwrap_used)]

use super::*;
use crate::gpu::seam;
use crate::gpu::submission::{empty, queued_tasks, read_blocking, read_failed};
use crate::gpu::{
    GpuCornerScan, GpuPatchTracker, GpuPatches, GpuPyramid, GpuPyramidBuilder, gpu_backends,
};

/// A read that fails is a typed error at every boundary, never a panic.
///
/// The failure this is about cannot be forced from a test: a `ServerError`
/// on a download means a lost device or a staging allocation refused under
/// memory pressure, and neither is reachable from a healthy card. What is
/// testable — and what the per-frame path actually depends on — is that the
/// mapping produces a variant the stage errors carry, so the unwinding a
/// panicking read would do through the released GIL (decision D32) cannot
/// happen.
#[test]
fn a_failed_device_read_is_a_typed_error_at_every_stage() {
    let what: &str = "the tracker result";
    let error: cubecl::server::ServerError = cubecl::server::ServerError::Generic {
        reason: "the device is gone".to_owned(),
        backtrace: cubecl::backtrace::BackTrace::default(),
    };
    assert_eq!(
        read_failed(what, &error),
        GpuError::DeviceReadFailed { what }
    );

    // The three stages a download sits in each carry it, so the error
    // reaches the Python boundary as the documented `ValueError`.
    let tracker: crate::frontend::tracker::TrackerError =
        GpuError::DeviceReadFailed { what }.into();
    let pyramid: crate::pyramid::PyramidError = GpuError::DeviceReadFailed { what }.into();
    let detect: crate::frontend::detect::DetectError = GpuError::DeviceReadFailed { what }.into();
    for message in [tracker.to_string(), pyramid.to_string(), detect.to_string()] {
        assert!(
            message.contains(what),
            "the stage error dropped what failed: {message}"
        );
    }
}

/// A failed download still empties this producer's queue accounting.
///
/// Every blocking read tells [`drained`] the device's queued tasks are
/// gone, and it does so whether the download answered or returned a
/// `ServerError`: the wait consumed them, the answer did not. A count left
/// standing after a failed read would make the next stage flush for tasks
/// that are not there — a second synchronisation a frameset, on a device
/// that is slow rather than lost (D77).
///
/// The `ServerError` cannot be produced on a healthy card, so one armed
/// fault stands in for it *at the read*, which is the only place it can be
/// armed without reaching past the accounting under test.
#[test]
fn a_failed_blocking_read_still_drains_the_queue() {
    let client = gpu_client().unwrap();
    let handle: cubecl::server::Handle = empty(&client, 256);
    assert!(
        queued_tasks(&client) > 0,
        "the allocation reserved no task, so the drain below proves nothing"
    );

    let what: &str = "the tracker result";
    arm_fault_at(BLOCKING_READ);
    assert_eq!(
        read_blocking(&client, vec![handle.clone()], what, &seam::READ_TRACK).unwrap_err(),
        GpuError::DeviceReadFailed { what }
    );
    assert_eq!(
        queued_tasks(&client),
        0,
        "a failed read left this producer's tasks outstanding"
    );

    // Unarmed the same read answers and drains, so the lines above measure
    // the failure path and not a read that never ran.
    let second: cubecl::server::Handle = empty(&client, 256);
    assert!(queued_tasks(&client) > 0);
    read_blocking(&client, vec![second], what, &seam::READ_TRACK).unwrap();
    assert_eq!(queued_tasks(&client), 0);
}

/// A panic inside a stage is a typed error, not an unwind into Python.
///
/// What a lost device does to a per-frame call, on the real stage and the
/// real client: `arm_fault_at` panics where the runtime would, at the
/// top of the guarded region, and what comes back is the stage's own error
/// type. The unguarded call on either side of it is the control — the path
/// works, so the middle line is measuring the guard and not a broken build.
#[test]
fn a_panic_inside_a_stage_is_a_typed_error() {
    use crate::pyramid::PyramidBuilder;

    let mut builder: GpuPyramidBuilder<GpuRuntime> =
        GpuPyramidBuilder::new(gpu_client().unwrap(), &[[0.0, 0.0]]);
    let mut pyramid: GpuPyramid<GpuRuntime> = builder.allocate(64, 64, 2).unwrap();
    let image: crate::image::ImageU16 = crate::image::ImageU16::zeros(64, 64).unwrap();
    builder.build(0, &image, &mut pyramid).unwrap();

    arm_fault_at(GUARDED_REGION);
    let error: crate::pyramid::PyramidError = builder.build(0, &image, &mut pyramid).unwrap_err();
    assert!(
        matches!(
            error,
            crate::pyramid::PyramidError::Gpu(GpuError::DeviceLost {
                what: "pyramid build"
            })
        ),
        "a panicking stage gave {error}"
    );

    // One call, and only that one: the flag is consumed where it fires.
    builder.build(0, &image, &mut pyramid).unwrap();
}

/// A panic anywhere in the bring-up is a typed error, not an unwind.
///
/// Two guards, because the bring-up has two layers now. `gpu_backends`
/// itself is guarded from its first line to the returned backends, and a
/// fault at the top of that region — before any client exists — comes back
/// as `ClientPanicked`. Inside it, `probe_storage` and the three
/// constructors each carry their own guard, because each is also a public
/// entry a caller reaches on its own, and a fault at the probe's site comes
/// back as that guard's `DeviceLost`. Either way nothing unwinds past the
/// constructor (decision D32).
#[test]
fn a_panic_after_the_client_is_built_is_a_typed_error() {
    arm_fault_at(GUARDED_REGION);
    let outer: crate::frontend::tracker::TrackerError =
        gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0, 2).unwrap_err();
    assert!(
        matches!(
            outer,
            crate::frontend::tracker::TrackerError::Gpu(GpuError::ClientPanicked { .. })
        ),
        "a panic in the outer region gave {outer}"
    );

    arm_fault_at(STORAGE_PROBE);
    let probe: crate::frontend::tracker::TrackerError =
        gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0, 2).unwrap_err();
    assert!(
        matches!(
            probe,
            crate::frontend::tracker::TrackerError::Gpu(GpuError::DeviceLost {
                what: "the storage probe"
            })
        ),
        "a panic in the storage probe gave {probe}"
    );

    // And the same call with nothing armed builds the three backends, so
    // what the lines above measure is the guards.
    gpu_backends::<crate::frontend::patterns::Pattern51>(64, 3, 5, 4.0, 2).unwrap();
}

/// A panic in an exported constructor is a typed error, not an unwind.
///
/// Three of them touch the device before they return — the corner scanner
/// uploads the FAST ring, the patch set allocates its store and its
/// positions, the tracker allocates both transform buffers — and each is a
/// public entry a caller outside `gpu_backends` can reach. Armed at the
/// guard, each returns its own error type; unarmed, each builds, so what
/// the armed lines measure is the guard and not a broken build (decision
/// D32).
#[test]
fn a_panic_in_an_exported_constructor_is_a_typed_error() {
    use crate::frontend::patterns::Pattern51;
    use crate::frontend::tracker::TrackerError;

    let client = gpu_client().unwrap();

    arm_fault_at(GUARDED_REGION);
    let scan: GpuError = GpuCornerScan::<GpuRuntime>::new(client.clone()).unwrap_err();
    assert_eq!(
        scan,
        GpuError::DeviceLost {
            what: "corner scan setup"
        }
    );

    arm_fault_at(GUARDED_REGION);
    let patches: TrackerError =
        GpuPatches::<Pattern51, GpuRuntime>::new(client.clone(), 64, 4).unwrap_err();
    assert!(
        matches!(
            patches,
            TrackerError::Gpu(GpuError::DeviceLost {
                what: "patch allocation"
            })
        ),
        "a panicking patch allocation gave {patches}"
    );

    arm_fault_at(GUARDED_REGION);
    let tracker: TrackerError =
        GpuPatchTracker::<Pattern51, GpuRuntime>::new(client.clone(), 64, 4, 5, 4.0, 2)
            .unwrap_err();
    assert!(
        matches!(
            tracker,
            TrackerError::Gpu(GpuError::DeviceLost {
                what: "tracker allocation"
            })
        ),
        "a panicking tracker allocation gave {tracker}"
    );

    GpuCornerScan::<GpuRuntime>::new(client.clone()).unwrap();
    GpuPatches::<Pattern51, GpuRuntime>::new(client.clone(), 64, 4).unwrap();
    GpuPatchTracker::<Pattern51, GpuRuntime>::new(client, 64, 4, 5, 4.0, 2).unwrap();
}

/// A panic in the public storage probe is a typed error, not an unwind.
///
/// [`probe_storage`] is public and it allocates, launches and downloads, so
/// it is a device operation a caller reaches without going through
/// `gpu_backends` and its guard. The fault is armed at the probe's own
/// site — the same one `a_panic_after_the_client_is_built_is_a_typed_error`
/// uses through the constructor path — and here the call is direct.
#[test]
fn a_panic_in_the_public_storage_probe_is_a_typed_error() {
    let client = gpu_client().unwrap();

    arm_fault_at(STORAGE_PROBE);
    assert_eq!(
        probe_storage(&client).unwrap_err(),
        GpuError::DeviceLost {
            what: "the storage probe"
        }
    );

    // Unarmed the same probe passes on this host, so the line above is the
    // guard and not a runtime that cannot store these widths.
    probe_storage(&client).unwrap();
}

/// A panic in an exported read is a typed error, not an unwind.
///
/// The two downloads that are not on the per-frame path — a pyramid level
/// and the patch store, both of them how generic code and the tolerance
/// suite read a buffer a GPU backend owns. Neither is inside a per-frame
/// stage, so neither was covered by the stage guards.
#[test]
fn a_panic_in_an_exported_read_is_a_typed_error() {
    use crate::frontend::patterns::Pattern51;
    use crate::frontend::tracker::TrackerError;
    use crate::pyramid::{Pyramid, PyramidBuilder, PyramidError};

    let client = gpu_client().unwrap();
    let builder: GpuPyramidBuilder<GpuRuntime> =
        GpuPyramidBuilder::new(client.clone(), &[[0.0, 0.0]]);
    let pyramid: GpuPyramid<GpuRuntime> = builder.allocate(64, 64, 2).unwrap();
    let patches: GpuPatches<Pattern51, GpuRuntime> = GpuPatches::new(client, 64, 3).unwrap();
    let mut level: crate::image::ImageU16 = crate::image::ImageU16::default();

    arm_fault_at(GUARDED_REGION);
    let read: PyramidError = pyramid.copy_level_into(0, &mut level).unwrap_err();
    assert!(
        matches!(
            read,
            PyramidError::Gpu(GpuError::DeviceLost {
                what: "a pyramid level read"
            })
        ),
        "a panicking level read gave {read}"
    );

    arm_fault_at(GUARDED_REGION);
    let store: TrackerError = patches.read_store().unwrap_err();
    assert!(
        matches!(
            store,
            TrackerError::Gpu(GpuError::DeviceLost {
                what: "the patch store read"
            })
        ),
        "a panicking store read gave {store}"
    );

    // Both reads succeed unarmed, so the two lines above are the guards.
    pyramid.copy_level_into(0, &mut level).unwrap();
    patches.read_store().unwrap();
}
