//! Compiles the V4L2 multi-planar capture shim (`src/capture/v4l2_mplane.c`, adapted from PR #270's `native/camera.c`) on Linux.
//! The shim keeps the kernel ABI (`linux/videodev2.h` structs and unions) in C; Rust owns the camera and its buffers' lifetime.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = std::env::var("TARGET")?;
    if target.contains("linux") {
        println!("cargo:rerun-if-changed=src/capture/v4l2_mplane.c");
        cc::Build::new().file("src/capture/v4l2_mplane.c").warnings_into_errors(true).compile("robocap_v4l2_mplane");
    }
    Ok(())
}
