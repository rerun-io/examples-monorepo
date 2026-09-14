use crate::{IioScan, IioScanLayout, MotionKind, SENSORS, iio::read_attribute};
use anyhow::{Context, Result, ensure};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read},
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::{Path, PathBuf},
};

/// Saves each changed attribute and restores it in reverse order on release.
/// The coordinator must first exclude the vendor capture owner.
pub struct IioDevice {
    file: File,
    layout: Option<IioScanLayout>,
    saved: Vec<(PathBuf, String)>,
    packet_size: usize,
}

impl IioDevice {
    pub fn start(index: u8) -> Result<Self> {
        let channel = SENSORS
            .iter()
            .find(|channel| channel.iio_index == index)
            .context("unknown Cap B IIO index")?;
        let root = PathBuf::from(format!("/sys/bus/iio/devices/iio:device{index}"));
        ensure!(
            read_attribute(&root.join("buffer/enable"))? == "0",
            "IIO buffer already owned"
        );
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(format!("/dev/iio:device{index}"))?;
        let mut owner = Self {
            file,
            layout: None,
            saved: Vec::new(),
            packet_size: channel.packet_size,
        };
        owner.set(&root.join("current_timestamp_clock"), "monotonic")?;
        let prefix = channel.prefix;
        for axis in ["x", "y", "z"] {
            owner.set(&root.join(format!("scan_elements/{prefix}_{axis}_en")), "1")?;
        }
        owner.set(&root.join("scan_elements/in_timestamp_en"), "1")?;
        if !matches!(channel.kind, MotionKind::Mag) {
            owner.set(&root.join("scan_elements/in_temp_en"), "1")?;
        }
        // The vendor's MAG capture path leaves its configured conversion rate
        // alone. Keep that sequence; changing the MMC register through this
        // driver is not needed to consume its buffered samples.
        if !matches!(channel.kind, MotionKind::Mag) {
            owner.set(&root.join("sampling_frequency"), "200")?;
        }
        owner.set(&root.join("buffer/length"), "4096")?;
        owner.set(&root.join("buffer/watermark"), "1")?;
        let layout = IioScanLayout::read(&root)?;
        ensure!(layout.clock() == "monotonic", "IIO refused monotonic clock");
        owner.layout = Some(layout);
        owner.set(&root.join("buffer/enable"), "1")?;
        Ok(owner)
    }

    fn set(&mut self, path: &Path, value: &str) -> Result<()> {
        let original = read_attribute(path)?;
        self.saved.push((path.to_owned(), original));
        fs::write(path, value).with_context(|| format!("set IIO attribute {}", path.display()))?;
        Ok(())
    }

    pub fn read_scans(&mut self) -> Result<Vec<IioScan>> {
        let mut fd = libc::pollfd {
            fd: self.file.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: one valid pollfd, borrowed for the duration of the call.
        let ready = unsafe { libc::poll(&mut fd, 1, 100) };
        if ready < 0 {
            return Err(io::Error::last_os_error().into());
        }
        if ready == 0 {
            return Ok(Vec::new());
        }
        ensure!(
            fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) == 0,
            "IIO descriptor fault"
        );
        let mut bytes = vec![0; self.packet_size * 256];
        let count = self.file.read(&mut bytes)?;
        ensure!(
            count > 0 && count % self.packet_size == 0,
            "incomplete IIO scan read"
        );
        let layout = self.layout.as_ref().context("IIO layout missing")?;
        bytes[..count]
            .chunks_exact(self.packet_size)
            .map(|packet| layout.decode(packet))
            .collect()
    }
}

impl Drop for IioDevice {
    fn drop(&mut self) {
        for (path, original) in self.saved.iter().rev() {
            if let Err(error) = fs::write(path, original) {
                eprintln!("IIO restore failed for {}: {error}", path.display());
            }
        }
    }
}

pub fn monotonic_ns() -> Result<i64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writable timespec and supported Linux clock ID.
    ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } == 0,
        "read monotonic clock"
    );
    Ok(time.tv_sec * 1_000_000_000 + time.tv_nsec)
}

pub struct FrameTrigger(File);
impl FrameTrigger {
    pub fn stopped() -> Result<Self> {
        let owner = Self(
            OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_CLOEXEC)
                .open("/dev/frame_trigger")?,
        );
        // SAFETY: this descriptor belongs to the capture coordinator; stop has no payload.
        ensure!(
            unsafe { libc::ioctl(owner.0.as_raw_fd(), 0x7401) } == 0,
            "stop previous frame trigger"
        );
        Ok(owner)
    }
    pub fn start(&self) -> Result<()> {
        let fps = 30_u32;
        // SAFETY: these ioctl numbers and argument sizes match the installed
        // Cap B application. The descriptor and fps pointer remain live.
        unsafe {
            ensure!(
                libc::ioctl(self.0.as_raw_fd(), 0x40047402, &fps) == 0,
                "set frame trigger rate"
            );
            ensure!(
                libc::ioctl(self.0.as_raw_fd(), 0x7400) == 0,
                "start frame trigger"
            );
        }
        Ok(())
    }
}
impl Drop for FrameTrigger {
    fn drop(&mut self) {
        // SAFETY: a live trigger descriptor; stop has no argument payload.
        if unsafe { libc::ioctl(self.0.as_raw_fd(), 0x7401) } != 0 {
            eprintln!("trigger stop failed: {}", io::Error::last_os_error());
        }
    }
}
