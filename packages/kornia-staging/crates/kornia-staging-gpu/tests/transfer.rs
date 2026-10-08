//! Device stream ordering and typed transfer failures.
#![cfg(feature = "wgpu")]
use cubecl::prelude::*;
use kornia_staging_gpu::{
    runtime::{gpu_client, GpuError},
    transfer::*,
};

#[test]
fn lookahead_reads_the_submitted_snapshot_and_keeps_the_next_write() {
    let client = gpu_client().unwrap();
    let handle = upload(&client, u32::as_bytes(&[7, 11, 19])).unwrap();
    let bytes = read_with_lookahead(&client, vec![handle.clone()], "snapshot", || {
        client.write(
            &handle,
            cubecl::bytes::Bytes::from_elems(vec![23u32, 29, 31]),
        );
        Ok(())
    })
    .unwrap();
    assert_eq!(u32::from_bytes(&bytes[0]), &[7, 11, 19]);
    let bytes = read_buffers(&client, vec![handle], "next").unwrap();
    assert_eq!(u32::from_bytes(&bytes[0]), &[23, 29, 31]);
}

#[test]
fn exclusive_execution_preserves_borrows_errors_and_survives_a_panic() {
    let client = gpu_client().unwrap();
    let values = [3u32, 5, 8];
    let result = execute_exclusive(&client, "borrowed input", || {
        let handle = upload(&client, u32::as_bytes(&values))?;
        let read = read_buffers(&client, vec![handle], "borrowed result")?;
        Ok::<_, GpuError>(u32::from_bytes(&read[0]).to_vec())
    })
    .unwrap();
    assert_eq!(result, values);
    let error = execute_exclusive(&client, "body panic", || -> Result<(), GpuError> {
        panic!("device work failed")
    })
    .unwrap_err();
    assert_eq!(error, GpuError::DeviceLost { what: "body panic" });
    assert_eq!(
        execute_exclusive(&client, "recovery", || Ok::<_, GpuError>(13)).unwrap(),
        13
    );
    let error = GpuError::DeviceReadFailed {
        what: "typed body error",
    };
    assert_eq!(
        execute_exclusive(&client, "preserve error", || Err::<(), _>(error)).unwrap_err(),
        error
    );
}

#[test]
fn failed_lookahead_does_not_publish_partial_bytes() {
    let client = gpu_client().unwrap();
    let handle = upload(&client, &[1, 2, 3, 4]).unwrap();
    let error = GpuError::DeviceReadFailed { what: "lookahead" };
    assert_eq!(
        read_with_lookahead(&client, vec![handle], "read", || Err(error)).unwrap_err(),
        error
    );
}
