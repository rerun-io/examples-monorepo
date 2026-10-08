use crate::{MotionKind, SENSORS};
use anyhow::{Context, Result, ensure};
use kornia_staging_sensor_iio::{DeviceConfig, ScanProfile};
use std::{
    fs::{File, OpenOptions},
    io,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
    path::PathBuf,
};

/// Open the cap's configured IIO channel with reusable caller scan storage.
pub fn start_imu(index: u8) -> Result<kornia_staging_sensor_iio::IioDevice> {
    let channel = SENSORS
        .iter()
        .find(|c| c.iio_index == index)
        .context("unknown Cap B IIO index")?;
    let (driver, temperature_channel) = match channel.kind {
        MotionKind::Gyro => ("icm42688-gyro", Some("in_temp")),
        MotionKind::Accel => ("icm42688-accel", Some("in_temp")),
        MotionKind::Mag => ("mmc5983ma", None),
    };
    Ok(kornia_staging_sensor_iio::IioDevice::start(DeviceConfig {
        sysfs: PathBuf::from(format!("/sys/bus/iio/devices/iio:device{index}")),
        node: PathBuf::from(format!("/dev/iio:device{index}")),
        profile: ScanProfile::new(driver, channel.prefix, temperature_channel)?,
        rate_hz: if matches!(channel.kind, MotionKind::Mag) {
            None
        } else {
            Some(200)
        },
        buffer_scans: 4096,
        watermark: 1,
        read_scans: 256,
        timeout_ms: 100,
    })?)
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
