//! Device ownership on the cap: IIO buffers (configured and restored exactly as PR #270's `IioDevice`), the frame trigger, the
//! monotonic clock, and the checks that this is a known cap (Cap A or Cap B) and that the vendor recorder no longer owns capture.
//!
//! Adapted from PR #270's `robocap-recorder/src/{device.rs, device_profile.rs, session.rs}` (271ce643).

use std::fs::{self, File, OpenOptions};
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use super::iio::{IioScan, IioScanLayout, MotionKind, SCAN_BYTES, read_attribute};
use super::{Cap, CaptureError};

/// CLOCK_MONOTONIC now, nanoseconds.
///
/// # Errors
///
/// [`CaptureError::Io`] if the clock cannot be read.
pub fn monotonic_ns() -> Result<i64, CaptureError> {
    let mut time = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: a writable timespec and a clock id Linux supports.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } != 0 {
        return Err(CaptureError::last_os_error("clock_gettime(CLOCK_MONOTONIC)"));
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
    Cap::identify(hostname, serial)
        .ok_or_else(|| CaptureError::Device(format!("live capture runs only on Cap A or Cap B; this is {hostname:?} {serial:?}")))
}

/// Refuse while the vendor recorder can own capture: every `omni-specs.bin` must be a zombie (the handoff script stops its
/// launcher loop and kills it; the dead child stays a zombie until the loop resumes). PR #270's `session::run` check.
///
/// # Errors
///
/// [`CaptureError::Device`] when a live vendor recorder exists.
pub fn require_vendor_recorder_stopped() -> Result<(), CaptureError> {
    let entries = fs::read_dir("/proc").map_err(|source| CaptureError::Io { what: "read /proc".into(), source })?;
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(name) = fs::read_to_string(path.join("comm")) else { continue };
        if name.trim() == "omni-specs.bin" {
            let status = fs::read_to_string(path.join("status")).unwrap_or_default();
            if !status.lines().any(|line| line.starts_with("State:") && line.contains("Z (zombie)")) {
                return Err(CaptureError::Device(format!(
                    "the vendor recorder ({}) still owns capture; run through scripts/handoff-run.sh",
                    path.display()
                )));
            }
        }
    }
    Ok(())
}

/// Exclusive owner of one IIO buffer. Saves each attribute it changes and restores them in reverse order on drop.
pub struct IioDevice {
    file: File,
    layout: IioScanLayout,
    saved: Vec<(PathBuf, String)>,
    scratch: Vec<u8>,
    /// SI units per count (sysfs `<prefix>_scale`).
    pub scale: f64,
    /// IIO device index.
    pub index: u8,
}

impl IioDevice {
    /// Configure and enable `iio:device{index}` as PR #270 does: require `buffer/enable == 0`, open `/dev/iio:deviceN`
    /// non-blocking, `current_timestamp_clock = monotonic`, enable x/y/z + timestamp + temp, `sampling_frequency = 200`,
    /// `buffer/length = 4096`, `buffer/watermark = 1`, then `buffer/enable = 1`.
    ///
    /// # Errors
    ///
    /// [`CaptureError`] when the buffer is already owned, an attribute cannot be set, or the layout is not the expected one.
    /// Attributes already changed are restored when the half-built owner drops.
    pub fn start(index: u8, kind: MotionKind) -> Result<Self, CaptureError> {
        let root = PathBuf::from(format!("/sys/bus/iio/devices/iio:device{index}"));
        let name = read_attribute(&root.join("name"))?;
        if name != kind.driver_name() {
            return Err(CaptureError::Device(format!("iio:device{index} is {name}, expected {}", kind.driver_name())));
        }
        if read_attribute(&root.join("buffer/enable"))? != "0" {
            return Err(CaptureError::Device(format!("iio:device{index}: buffer already enabled (owned by someone else)")));
        }
        let node = format!("/dev/iio:device{index}");
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(&node)
            .map_err(|source| CaptureError::Io { what: format!("open {node}"), source })?;
        let prefix = kind.prefix();
        let scale_text = read_attribute(&root.join(format!("{prefix}_scale")))?;
        let scale: f64 = scale_text
            .parse()
            .map_err(|_| CaptureError::Attribute { path: root.join(format!("{prefix}_scale")), message: format!("not a number: {scale_text}") })?;
        let placeholder = IioScanLayout { clock: String::new(), temperature: false, kind };
        let mut owner = Self { file, layout: placeholder, saved: Vec::new(), scratch: vec![0; SCAN_BYTES * 256], scale, index };
        owner.set(&root.join("current_timestamp_clock"), "monotonic")?;
        for axis in ["x", "y", "z"] {
            owner.set(&root.join(format!("scan_elements/{prefix}_{axis}_en")), "1")?;
        }
        owner.set(&root.join("scan_elements/in_timestamp_en"), "1")?;
        owner.set(&root.join("scan_elements/in_temp_en"), "1")?;
        owner.set(&root.join("sampling_frequency"), "200")?;
        owner.set(&root.join("buffer/length"), "4096")?;
        owner.set(&root.join("buffer/watermark"), "1")?;
        let layout = IioScanLayout::read(&root)?;
        if layout.clock != "monotonic" {
            return Err(CaptureError::Device(format!("iio:device{index} refused the monotonic clock ({})", layout.clock)));
        }
        owner.layout = layout;
        owner.set(&root.join("buffer/enable"), "1")?;
        Ok(owner)
    }

    fn set(&mut self, path: &Path, value: &str) -> Result<(), CaptureError> {
        let original = read_attribute(path)?;
        self.saved.push((path.to_path_buf(), original));
        fs::write(path, value).map_err(|error| CaptureError::Attribute { path: path.to_path_buf(), message: format!("write {value:?}: {error}") })
    }

    /// Wait up to `timeout_ms` for scans and decode all that are buffered (empty on a timeout).
    ///
    /// # Errors
    ///
    /// [`CaptureError`] on a poll/read fault or a partial scan.
    pub fn read_scans(&mut self, timeout_ms: i32, out: &mut Vec<IioScan>) -> Result<(), CaptureError> {
        let mut fd = libc::pollfd { fd: self.file.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one valid pollfd, borrowed for the call.
        let ready = unsafe { libc::poll(&mut fd, 1, timeout_ms) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                return Ok(());
            }
            return Err(CaptureError::Io { what: format!("poll iio:device{}", self.index), source: error });
        }
        if ready == 0 {
            return Ok(());
        }
        if fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(CaptureError::Device(format!("iio:device{} descriptor fault (revents {:#x})", self.index, fd.revents)));
        }
        let count = match self.file.read(&mut self.scratch) {
            Ok(count) => count,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
            Err(source) => return Err(CaptureError::Io { what: format!("read iio:device{}", self.index), source }),
        };
        if count == 0 || count % SCAN_BYTES != 0 {
            return Err(CaptureError::Device(format!("iio:device{}: read {count} bytes, not whole scans", self.index)));
        }
        for packet in self.scratch[..count].chunks_exact(SCAN_BYTES) {
            out.push(self.layout.decode(packet)?);
        }
        Ok(())
    }
}

impl Drop for IioDevice {
    fn drop(&mut self) {
        for (path, original) in self.saved.iter().rev() {
            if let Err(error) = fs::write(path, original) {
                eprintln!("robocap-live: IIO restore failed for {}: {error}", path.display());
            }
        }
    }
}

/// The cap's frame trigger (`/dev/frame_trigger`): stopped on open and on drop; started at 30 fps once every stream is ready.
/// PR #270's ioctls: `0x7401` stop, `0x40047402` set fps (u32), `0x7400` start. Never read it without polling first.
pub struct FrameTrigger(File);

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
            .map_err(|source| CaptureError::Io { what: "open /dev/frame_trigger".into(), source })?;
        // SAFETY: our own descriptor; the stop request has no payload.
        if unsafe { libc::ioctl(file.as_raw_fd(), 0x7401 as _) } != 0 {
            return Err(CaptureError::last_os_error("stop the frame trigger"));
        }
        Ok(Self(file))
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
            if libc::ioctl(self.0.as_raw_fd(), 0x4004_7402 as _, &fps) != 0 {
                return Err(CaptureError::last_os_error("set the frame trigger rate"));
            }
            if libc::ioctl(self.0.as_raw_fd(), 0x7400 as _) != 0 {
                return Err(CaptureError::last_os_error("start the frame trigger"));
            }
        }
        Ok(())
    }
}

impl Drop for FrameTrigger {
    fn drop(&mut self) {
        // SAFETY: a live descriptor; stop has no payload.
        if unsafe { libc::ioctl(self.0.as_raw_fd(), 0x7401 as _) } != 0 {
            eprintln!("robocap-live: frame trigger stop failed: {}", std::io::Error::last_os_error());
        }
    }
}
