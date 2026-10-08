#![allow(unsafe_code)] // FFI calls uphold the documented buffer and ownership contracts.
use crate::{read_attribute, IioError, IioScan, IioScanLayout, ScanProfile};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
};

/// Caller-selected paths, layout, sampling rate and buffer bounds.
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    /// Directory containing this device's sysfs attributes.
    pub sysfs: PathBuf,
    /// Device node to open nonblocking.
    pub node: PathBuf,
    /// Expected driver and scan encoding.
    pub profile: ScanProfile,
    /// Sampling rate in Hz; `None` preserves the existing rate.
    pub rate_hz: Option<u32>,
    /// Kernel buffer capacity in scans.
    pub buffer_scans: usize,
    /// Kernel readiness watermark in scans.
    pub watermark: usize,
    /// Maximum scans returned by one read.
    pub read_scans: usize,
    /// Poll timeout in milliseconds, fixed at construction.
    pub timeout_ms: u16,
}

/// Exclusive buffer owner. Restores attributes in reverse order on drop, including failed setup.
///
/// ```no_run
/// # use kornia_staging_sensor_iio::{DeviceConfig, IioDevice};
/// # fn capture(config: DeviceConfig) -> Result<(), kornia_staging_sensor_iio::IioError> {
/// let mut device = IioDevice::start(config)?;
/// let mut scans = Vec::new();
/// device.read_scans(&mut scans)?;
/// # Ok(()) }
/// ```
pub struct IioDevice {
    _restore: RestoreAttributes,
    file: File,
    layout: IioScanLayout,
    scratch: Vec<u8>,
    timeout_ms: i32,
}

impl IioDevice {
    /// Open an idle device, enable its selected channels and require the monotonic clock.
    ///
    /// # Arguments
    /// * `config` - paths, supported layout, rate and buffer limits.
    ///
    /// # Errors
    /// Invalid bounds, active buffer, IO or layout failures. Partial setup is restored.
    pub fn start(config: DeviceConfig) -> Result<Self, IioError> {
        if config.buffer_scans == 0
            || config.watermark == 0
            || config.watermark > config.buffer_scans
            || config.read_scans == 0
            || config.read_scans > config.buffer_scans
            || config.rate_hz == Some(0)
        {
            return Err(IioError::Invalid("invalid device configuration".into()));
        }
        let root = &config.sysfs;
        if read_attribute(&root.join("buffer/enable"))? != "0" {
            return Err(IioError::Invalid("buffer already owned".into()));
        }
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(&config.node)?;
        let mut restore = RestoreAttributes(Vec::new());
        restore.set(&root.join("current_timestamp_clock"), "monotonic")?;
        for axis in ["x", "y", "z"] {
            restore.set(
                &root.join(format!("scan_elements/{}_{axis}_en", config.profile.prefix)),
                "1",
            )?;
        }
        restore.set(&root.join("scan_elements/in_timestamp_en"), "1")?;
        if let Some(name) = &config.profile.temperature_channel {
            restore.set(&root.join(format!("scan_elements/{name}_en")), "1")?;
        }
        if let Some(rate) = config.rate_hz {
            restore.set(&root.join("sampling_frequency"), &rate.to_string())?;
        }
        restore.set(
            &root.join("buffer/length"),
            &config.buffer_scans.to_string(),
        )?;
        restore.set(
            &root.join("buffer/watermark"),
            &config.watermark.to_string(),
        )?;
        let layout = IioScanLayout::read(root, &config.profile)?;
        if layout.clock() != "monotonic" {
            return Err(IioError::Invalid("device refused monotonic clock".into()));
        }
        let size = layout
            .scan_bytes()
            .checked_mul(config.read_scans)
            .ok_or_else(|| IioError::Invalid("read buffer overflow".into()))?;
        restore.set(&root.join("buffer/enable"), "1")?;
        Ok(Self {
            _restore: restore,
            file,
            layout,
            scratch: vec![0; size],
            timeout_ms: i32::from(config.timeout_ms),
        })
    }

    /// Append available scans into reusable caller storage, using the configured timeout.
    /// Interrupted polls and timeouts append nothing; existing output is retained.
    ///
    /// # Arguments
    /// * `out` - retained output to append scans to.
    ///
    /// # Errors
    /// Descriptor failure, read failure or incomplete scan.
    pub fn read_scans(&mut self, out: &mut Vec<IioScan>) -> Result<(), IioError> {
        let mut fd = libc::pollfd {
            fd: self.file.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: valid writable pollfd borrowed for this call.
        let ready = unsafe { libc::poll(&mut fd, 1, self.timeout_ms) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            return if error.kind() == std::io::ErrorKind::Interrupted {
                Ok(())
            } else {
                Err(error.into())
            };
        }
        if ready == 0 {
            return Ok(());
        }
        if fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
            return Err(IioError::Invalid("descriptor fault".into()));
        }
        let count = match self.file.read(&mut self.scratch) {
            Ok(count) => count,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let size = self.layout.scan_bytes();
        if count == 0 || count % size != 0 {
            return Err(IioError::Invalid(format!(
                "incomplete scan read: {count} bytes"
            )));
        }
        for packet in self.scratch[..count].chunks_exact(size) {
            out.push(self.layout.decode(packet)?);
        }
        Ok(())
    }
}

struct RestoreAttributes(Vec<(PathBuf, String)>);
impl RestoreAttributes {
    fn set(&mut self, path: &Path, value: &str) -> Result<(), IioError> {
        self.0.push((path.to_owned(), read_attribute(path)?));
        fs::write(path, value).map_err(|source| IioError::Attribute {
            path: path.to_owned(),
            source,
        })
    }
}
impl Drop for RestoreAttributes {
    fn drop(&mut self) {
        for (path, value) in self.0.iter().rev() {
            if let Err(error) = fs::write(path, value) {
                eprintln!("IIO restore failed for {}: {error}", path.display());
            }
        }
    }
}
