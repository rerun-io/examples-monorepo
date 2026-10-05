//! IIO buffer scan layout and decoding for the ICM42688 gyro and accelerometer (IMU0 on the cap).
//!
//! Adapted from PR #270's `robocap-recorder/src/iio.rs` (271ce643), restricted to the two channels SLAM uses (no MAG). Reading
//! the layout does not change device configuration.

use std::fs;
use std::path::Path;

use super::CaptureError;

/// The IMU channel an IIO device carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionKind {
    /// `icm42688-gyro`, `in_anglvel_*`.
    Gyro,
    /// `icm42688-accel`, `in_accel_*`.
    Accel,
}

impl MotionKind {
    /// The sysfs channel prefix.
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Gyro => "in_anglvel",
            Self::Accel => "in_accel",
        }
    }

    /// The driver name this kind requires.
    pub fn driver_name(self) -> &'static str {
        match self {
            Self::Gyro => "icm42688-gyro",
            Self::Accel => "icm42688-accel",
        }
    }
}

/// Bytes per buffered scan: x, y, z (be s16), temp (le s16), then the 8-byte-aligned le s64 timestamp.
pub const SCAN_BYTES: usize = 16;

/// One decoded scan: raw counts, no scaling, kernel timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IioScan {
    /// Raw x, y, z counts.
    pub raw: [i32; 3],
    /// Raw temperature, when that channel is enabled.
    pub temperature_raw: Option<i16>,
    /// Kernel timestamp, nanoseconds of the declared clock.
    pub timestamp_ns: i64,
}

/// Read and trim a sysfs attribute.
///
/// # Errors
///
/// [`CaptureError::Attribute`] when it cannot be read.
pub fn read_attribute(path: &Path) -> Result<String, CaptureError> {
    fs::read_to_string(path)
        .map(|text| text.trim_matches(|c: char| c.is_whitespace() || c == '\0').to_owned())
        .map_err(|error| CaptureError::Attribute { path: path.to_path_buf(), message: error.to_string() })
}

/// The validated layout of an enabled ICM42688 gyro or accel buffer.
#[derive(Clone, Debug)]
pub struct IioScanLayout {
    /// The kernel's declared timestamp clock (`current_timestamp_clock`).
    pub clock: String,
    /// Whether `in_temp` is in the scan.
    pub temperature: bool,
    /// Which channel this is.
    pub kind: MotionKind,
}

impl IioScanLayout {
    /// Read the layout of `device` (a `/sys/bus/iio/devices/iio:deviceN` directory) and check it is exactly the one this
    /// decoder handles: x/y/z `be:s16/16>>0` at indices 0..3, `in_temp` `le:s16/16>>0` at 3, `in_timestamp` `le:s64/64>>0` at 4,
    /// and no other channel enabled.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Device`] for another driver or layout; [`CaptureError::Attribute`] when sysfs cannot be read.
    pub fn read(device: &Path) -> Result<Self, CaptureError> {
        let name = read_attribute(&device.join("name"))?;
        let kind = match name.as_str() {
            "icm42688-gyro" => MotionKind::Gyro,
            "icm42688-accel" => MotionKind::Accel,
            _ => return Err(CaptureError::Device(format!("{}: unsupported IIO device {name}", device.display()))),
        };
        let clock = read_attribute(&device.join("current_timestamp_clock"))?;
        if clock.is_empty() {
            return Err(CaptureError::Device(format!("{}: no timestamp clock", device.display())));
        }
        let scan = device.join("scan_elements");
        let prefix = kind.prefix();
        let mut expected: Vec<(String, usize, &str)> =
            ["x", "y", "z"].iter().enumerate().map(|(index, axis)| (format!("{prefix}_{axis}"), index, "be:s16/16>>0")).collect();
        expected.push(("in_temp".into(), 3, "le:s16/16>>0"));
        expected.push(("in_timestamp".into(), 4, "le:s64/64>>0"));
        let mut temperature = false;
        for (channel, index, format) in &expected {
            let channel_index = read_attribute(&scan.join(format!("{channel}_index")))?;
            let channel_type = read_attribute(&scan.join(format!("{channel}_type")))?;
            if channel_index != index.to_string() || channel_type != *format {
                return Err(CaptureError::Device(format!(
                    "{}: channel {channel} is index {channel_index} type {channel_type}, expected {index} {format}",
                    device.display()
                )));
            }
            let enabled = read_attribute(&scan.join(format!("{channel}_en")))?;
            match (channel.as_str(), enabled.as_str()) {
                ("in_temp", "0" | "1") => temperature = enabled == "1",
                (_, "1") => {}
                _ => return Err(CaptureError::Device(format!("{}: required channel {channel} is not enabled", device.display()))),
            }
        }
        let entries = fs::read_dir(&scan).map_err(|error| CaptureError::Attribute { path: scan.clone(), message: error.to_string() })?;
        for entry in entries {
            let path = entry.map_err(|error| CaptureError::Attribute { path: scan.clone(), message: error.to_string() })?.path();
            let Some(file) = path.file_name().and_then(|name| name.to_str()) else { continue };
            if let Some(channel) = file.strip_suffix("_en")
                && !expected.iter().any(|(name, _, _)| name == channel)
                && read_attribute(&path)? != "0"
            {
                return Err(CaptureError::Device(format!("{}: unexpected enabled channel {channel}", device.display())));
            }
        }
        Ok(Self { clock, temperature, kind })
    }

    /// Decode one [`SCAN_BYTES`]-byte scan.
    ///
    /// # Errors
    ///
    /// [`CaptureError::Device`] when the packet is not exactly one scan.
    pub fn decode(&self, packet: &[u8]) -> Result<IioScan, CaptureError> {
        let bytes: &[u8; SCAN_BYTES] =
            packet.try_into().map_err(|_| CaptureError::Device(format!("IIO scan of {} bytes, expected {SCAN_BYTES}", packet.len())))?;
        let raw = std::array::from_fn(|axis| i32::from(i16::from_be_bytes([bytes[2 * axis], bytes[2 * axis + 1]])));
        let mut stamp = [0u8; 8];
        stamp.copy_from_slice(&bytes[8..16]);
        Ok(IioScan {
            raw,
            temperature_raw: self.temperature.then(|| i16::from_le_bytes([bytes[6], bytes[7]])),
            timestamp_ns: i64::from_le_bytes(stamp),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_device(dir: &Path, name: &str, prefix: &str, extra_enabled: bool) -> std::io::Result<()> {
        let scan = dir.join("scan_elements");
        fs::create_dir_all(&scan)?;
        fs::write(dir.join("name"), format!("{name}\n"))?;
        fs::write(dir.join("current_timestamp_clock"), "monotonic\n")?;
        for (index, axis) in ["x", "y", "z"].iter().enumerate() {
            fs::write(scan.join(format!("{prefix}_{axis}_index")), format!("{index}\n"))?;
            fs::write(scan.join(format!("{prefix}_{axis}_type")), "be:s16/16>>0\n")?;
            fs::write(scan.join(format!("{prefix}_{axis}_en")), "1\n")?;
        }
        fs::write(scan.join("in_temp_index"), "3\n")?;
        fs::write(scan.join("in_temp_type"), "le:s16/16>>0\n")?;
        fs::write(scan.join("in_temp_en"), "1\n")?;
        fs::write(scan.join("in_timestamp_index"), "4\n")?;
        fs::write(scan.join("in_timestamp_type"), "le:s64/64>>0\n")?;
        fs::write(scan.join("in_timestamp_en"), "1\n")?;
        fs::write(scan.join("in_other_en"), if extra_enabled { "1\n" } else { "0\n" })?;
        Ok(())
    }

    #[test]
    fn the_icm42688_layout_decodes_big_endian_axes_and_the_aligned_timestamp() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join(format!("robocap-live-iio-{}", std::process::id()));
        fake_device(&dir, "icm42688-gyro", "in_anglvel", false)?;
        let layout = IioScanLayout::read(&dir)?;
        assert_eq!((layout.kind, layout.clock.as_str(), layout.temperature), (MotionKind::Gyro, "monotonic", true));
        let mut packet = [0u8; SCAN_BYTES];
        packet[0..2].copy_from_slice(&(-2i16).to_be_bytes());
        packet[2..4].copy_from_slice(&300i16.to_be_bytes());
        packet[4..6].copy_from_slice(&i16::MIN.to_be_bytes());
        packet[6..8].copy_from_slice(&25i16.to_le_bytes());
        packet[8..16].copy_from_slice(&123_456_789_012i64.to_le_bytes());
        let scan = layout.decode(&packet)?;
        assert_eq!(scan, IioScan { raw: [-2, 300, -32768], temperature_raw: Some(25), timestamp_ns: 123_456_789_012 });
        assert!(layout.decode(&packet[..15]).is_err());
        fake_device(&dir, "icm42688-gyro", "in_anglvel", true)?;
        assert!(IioScanLayout::read(&dir).is_err(), "an unexpected enabled channel must be refused");
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
