//! Device ownership on the cap: IIO buffers (configured and restored exactly as PR #270's `IioDevice`), the frame trigger, the
//! monotonic clock, and the checks that this is a known cap (Cap A or Cap B) and that the vendor recorder no longer owns capture.
//!
//! Adapted from PR #270's `robocap-recorder/src/{device.rs, device_profile.rs, session.rs}` (271ce643).

use std::fs::{self, File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use super::iio::MotionKind;
use super::{Cap, CaptureError};
use kornia_staging_sensor_iio::{DeviceConfig, ScanProfile, read_attribute};

/// CLOCK_MONOTONIC now, nanoseconds.
///
/// # Errors
///
/// [`CaptureError::Io`] if the clock cannot be read.
pub fn monotonic_ns() -> Result<i64, CaptureError> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a writable timespec and a clock id Linux supports.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } != 0 {
        return Err(CaptureError::last_os_error(
            "clock_gettime(CLOCK_MONOTONIC)",
        ));
    }
    Ok(time.tv_sec * 1_000_000_000 + time.tv_nsec)
}

/// Refuse to touch devices anywhere but a known cap (hostname + device-tree serial, PR #270's `DeviceProfile::identify`).
///
/// # Errors
///
/// [`CaptureError::Device`] on any other machine.
pub fn require_cap() -> Result<Cap, CaptureError> {
    let hostname = fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
    let serial = fs::read_to_string("/proc/device-tree/serial-number").unwrap_or_default();
    let (hostname, serial) = (hostname.trim(), serial.trim_end_matches('\0').trim());
    Cap::identify(hostname, serial).ok_or_else(|| {
        CaptureError::Device(format!(
            "live capture runs only on Cap A or Cap B; this is {hostname:?} {serial:?}"
        ))
    })
}

/// Refuse while the vendor recorder can own capture: every `omni-specs.bin` must be a zombie (`robocap-panel handoff` stops its
/// launcher loop and kills it; the dead child stays a zombie until the loop resumes). PR #270's `session::run` check.
///
/// # Errors
///
/// [`CaptureError::Device`] when a live vendor recorder exists.
pub fn require_vendor_recorder_stopped() -> Result<(), CaptureError> {
    let entries = fs::read_dir("/proc").map_err(|source| CaptureError::Io {
        what: "read /proc".into(),
        source,
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(name) = fs::read_to_string(path.join("comm")) else {
            continue;
        };
        if name.trim() == "omni-specs.bin" {
            let status = fs::read_to_string(path.join("status")).unwrap_or_default();
            if !status
                .lines()
                .any(|line| line.starts_with("State:") && line.contains("Z (zombie)"))
            {
                return Err(CaptureError::Device(format!(
                    "the vendor recorder ({}) still owns capture; run through the panel's supervisor (robocap-panel handoff)",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

/// Read a positive finite SI-per-count scale before capture can feed IMU maths.
///
/// # Errors
/// Returns an attribute error for malformed, nonpositive or nonfinite scales.
pub fn read_scale(path: &std::path::Path) -> Result<f64, CaptureError> {
    let text = read_attribute(path)?;
    let value = text
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && *v > 0.0);
    value.ok_or_else(|| CaptureError::Attribute {
        path: path.to_owned(),
        message: format!("invalid positive finite scale: {text}"),
    })
}

/// RoboCap's IIO profile and SI scale, backed by the shared buffer owner.
pub struct ImuDevice {
    /// Shared owner; callers reuse their scan storage.
    pub owner: kornia_staging_sensor_iio::IioDevice,
    /// SI units per raw count.
    pub scale: f64,
    /// Device index in the cap profile.
    pub index: u8,
}

impl ImuDevice {
    /// Start an IMU at the cap's 200 Hz rate and restore configuration on release.
    ///
    /// # Errors
    /// Returns a capture error for an invalid scale, busy device or unsupported layout.
    pub fn start(index: u8, kind: MotionKind) -> Result<Self, CaptureError> {
        let root = PathBuf::from(format!("/sys/bus/iio/devices/iio:device{index}"));
        let path = root.join(format!("{}_scale", kind.prefix()));
        let scale = read_scale(&path)?;
        let owner = kornia_staging_sensor_iio::IioDevice::start(DeviceConfig {
            sysfs: root,
            node: PathBuf::from(format!("/dev/iio:device{index}")),
            profile: ScanProfile::new(
                kind.driver_name(),
                kind.prefix(),
                Some("in_temp"),
            )?,
            rate_hz: Some(200),
            buffer_scans: 4096,
            watermark: 1,
            read_scans: 256,
            timeout_ms: 100,
        })?;
        Ok(Self {
            owner,
            scale,
            index,
        })
    }
}

/// The cap's frame trigger (`/dev/frame_trigger`): stopped on open and on drop; started at 30 fps once every stream is ready.
/// PR #270's ioctls: `0x7401` stop, `0x40047402` set fps (u32), `0x7400` start. Never read it without polling first.
pub struct FrameTrigger(File);

const TRIGGER_STOP: libc::c_ulong = 0x7401;
const TRIGGER_SET_FPS: libc::c_ulong = 0x4004_7402;
const TRIGGER_START: libc::c_ulong = 0x7400;

impl FrameTrigger {
    /// Open the trigger and stop any previous run.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] if it cannot be opened or stopped.
    pub fn stopped() -> Result<Self, CaptureError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open("/dev/frame_trigger")
            .map_err(|source| CaptureError::Io {
                what: "open /dev/frame_trigger".into(),
                source,
            })?;
        let trigger = Self(file);
        trigger.stop()?;
        Ok(trigger)
    }

    /// Stop shared frame pulses before draining the six camera streams.
    /// # Errors
    /// Returns the driver's stop ioctl error.
    pub fn stop(&self) -> Result<(), CaptureError> {
        // SAFETY: live owned descriptor; this request has no payload.
        if unsafe { libc::ioctl(self.0.as_raw_fd(), TRIGGER_STOP as _) } != 0 {
            return Err(CaptureError::last_os_error("stop the frame trigger"));
        }
        Ok(())
    }

    /// Set 30 fps and start triggering.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Io`] if either ioctl fails.
    pub fn start(&self) -> Result<(), CaptureError> {
        let fps: u32 = 30;
        // SAFETY: the ioctl numbers and argument size (u32) are the installed driver's (PR #270); descriptor and pointer are live.
        unsafe {
            if libc::ioctl(self.0.as_raw_fd(), TRIGGER_SET_FPS as _, &fps) != 0 {
                return Err(CaptureError::last_os_error("set the frame trigger rate"));
            }
            if libc::ioctl(self.0.as_raw_fd(), TRIGGER_START as _) != 0 {
                return Err(CaptureError::last_os_error("start the frame trigger"));
            }
        }
        Ok(())
    }
}

impl Drop for FrameTrigger {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("robocap-live: frame trigger stop failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_scale_never_reaches_imu_math() -> Result<(), Box<dyn std::error::Error>> {
        let path = std::env::temp_dir().join(format!("robocap-scale-{}", std::process::id()));
        for invalid in ["NaN", "inf", "-inf", "0", "-1", "oops"] {
            fs::write(&path, invalid)?;
            assert!(read_scale(&path).is_err(), "{invalid}");
        }
        fs::write(&path, "0.001197101")?;
        assert_eq!(read_scale(&path)?, 0.001197101);
        fs::remove_file(path)?;
        Ok(())
    }
}
