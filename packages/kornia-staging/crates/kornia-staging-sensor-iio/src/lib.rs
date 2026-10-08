//! Validated IIO scan decoding and Linux buffer ownership.
#![deny(missing_docs)]

use std::{
    fs,
    path::{Path, PathBuf},
};

#[cfg(target_os = "linux")]
mod device;
#[cfg(target_os = "linux")]
pub use device::{DeviceConfig, IioDevice};

/// Scan metadata, configuration, or device IO failure.
#[derive(Debug, thiserror::Error)]
pub enum IioError {
    /// A sysfs attribute could not be read or written.
    #[error("{path}: {source}")]
    Attribute {
        /// Attribute path.
        path: PathBuf,
        /// Underlying IO failure.
        source: std::io::Error,
    },
    /// Unsupported or malformed layout/configuration.
    #[error("{0}")]
    Invalid(String),
    /// Device IO failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// App-supplied driver and scan layout. No hardware profile is selected implicitly.
#[derive(Clone, Debug)]
pub struct ScanProfile {
    /// Expected sysfs driver name.
    driver: String,
    /// Axis attribute prefix, e.g. `in_accel`.
    prefix: String,
    /// Optional application-selected temperature channel.
    temperature_channel: Option<String>,
}

impl ScanProfile {
    /// Validate a driver name and axis attribute prefix.
    /// # Arguments
    /// * `driver` - expected sysfs name.
    /// * `prefix` - axis attribute prefix without path separators.
    /// * `temperature_channel` - optional temperature attribute name without path separators.
    /// # Errors
    /// Rejects empty names or a prefix containing a path separator.
    pub fn new(
        driver: impl Into<String>,
        prefix: impl Into<String>,
        temperature_channel: Option<&str>,
    ) -> Result<Self, IioError> {
        let driver = driver.into();
        let prefix = prefix.into();
        if driver.is_empty()
            || prefix.is_empty()
            || prefix.contains('/')
            || temperature_channel.is_some_and(|name| name.is_empty() || name.contains('/'))
        {
            return Err(IioError::Invalid("invalid scan profile".into()));
        }
        Ok(Self {
            driver,
            prefix,
            temperature_channel: temperature_channel.map(str::to_owned),
        })
    }
}

/// A decoded kernel scan, without clock conversion, calibration, or SI scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IioScan {
    /// Raw signed x/y/z counts.
    pub raw: [i32; 3],
    /// Raw temperature if enabled.
    pub temperature_raw: Option<i16>,
    /// Nanoseconds in the layout's declared clock domain.
    pub timestamp_ns: i64,
}

/// Validated scan layout. Reading metadata never enables a device.
#[derive(Clone, Debug)]
pub struct IioScanLayout {
    clock: String,
    axes: [ScanChannel; 3],
    temperature: Option<ScanChannel>,
    timestamp: ScanChannel,
    scan_bytes: usize,
}

// One scalar channel from the kernel scan_elements grammar. Repeated fields do
// not fit the xyz/temperature/timestamp payload and are rejected explicitly.
#[derive(Clone, Copy, Debug, Default)]
struct ScanChannel {
    offset: usize,
    bytes: usize,
    bits: u32,
    shift: u32,
    signed: bool,
    big_endian: bool,
}
impl ScanChannel {
    fn parse(text: &str) -> Result<Self, IioError> {
        let invalid = || IioError::Invalid(format!("unsupported scan type {text}"));
        let (endian, field) = text.split_once(':').ok_or_else(invalid)?;
        let (signed, field) = match field.as_bytes().first() {
            Some(b's') => (true, &field[1..]),
            Some(b'u') => (false, &field[1..]),
            _ => return Err(invalid()),
        };
        let (bits, storage) = field.split_once('/').ok_or_else(invalid)?;
        let (storage, shift) = storage.split_once(">>").unwrap_or((storage, "0"));
        let storage = if let Some((storage, repeat)) = storage.split_once('X') {
            if repeat.parse::<usize>().map_err(|_| invalid())? != 1 {
                return Err(invalid());
            }
            storage
        } else {
            storage
        };
        let bits: u32 = bits.parse().map_err(|_| invalid())?;
        let storage: u32 = storage.parse().map_err(|_| invalid())?;
        let shift: u32 = shift.parse().map_err(|_| invalid())?;
        if !matches!(endian, "be" | "le")
            || !matches!(storage, 8 | 16 | 32 | 64)
            || bits == 0
            || bits > storage
            || shift > storage - bits
        {
            return Err(invalid());
        }
        Ok(Self {
            offset: 0,
            bytes: (storage / 8) as usize,
            bits,
            shift,
            signed,
            big_endian: endian == "be",
        })
    }

    fn decode(self, packet: &[u8]) -> i64 {
        let mut word = 0u64;
        let bytes = &packet[self.offset..self.offset + self.bytes];
        for (i, &byte) in bytes.iter().enumerate() {
            let index = if self.big_endian {
                self.bytes - 1 - i
            } else {
                i
            };
            word |= u64::from(byte) << (index * 8);
        }
        word >>= self.shift;
        if self.bits < 64 {
            word &= (1u64 << self.bits) - 1;
        }
        if self.signed {
            ((word << (64 - self.bits)) as i64) >> (64 - self.bits)
        } else {
            word as i64
        }
    }
}

/// Read an attribute, stripping whitespace and kernel NUL suffixes.
///
/// # Arguments
/// * `path` - attribute file to read.
///
/// # Errors
/// Returns [`IioError::Attribute`] when the read fails.
pub fn read_attribute(path: &Path) -> Result<String, IioError> {
    fs::read_to_string(path)
        .map(|s| {
            s.trim_matches(|c: char| c.is_whitespace() || c == '\0')
                .to_owned()
        })
        .map_err(|source| IioError::Attribute {
            path: path.to_owned(),
            source,
        })
}

impl IioScanLayout {
    /// Validate enabled channels, indices, storage widths, signs, and endianness.
    /// Uses the kernel scan-elements grammar and natural storage alignment.
    /// Repeated fields and values wider than the scalar output are rejected.
    ///
    /// ```no_run
    /// use kornia_staging_sensor_iio::{IioScanLayout, ScanProfile};
    /// let profile = ScanProfile::new("my-imu", "in_accel", Some("in_temp"))?;
    /// let layout = IioScanLayout::read(std::path::Path::new("/sys/bus/iio/devices/iio:device0"), &profile)?;
    /// let sample = layout.decode(&[0; 16])?;
    /// assert_eq!(sample.raw, [0; 3]);
    /// # Ok::<(), kornia_staging_sensor_iio::IioError>(())
    /// ```
    ///
    /// # Arguments
    /// * `device` - sysfs directory for one device.
    /// * `profile` - caller-selected driver and supported encoding.
    ///
    /// # Errors
    /// Returns an error for missing metadata, unsupported channels, or a different driver.
    pub fn read(device: &Path, profile: &ScanProfile) -> Result<Self, IioError> {
        let name = read_attribute(&device.join("name"))?;
        if name != profile.driver {
            return Err(IioError::Invalid(format!(
                "driver {name}, expected {}",
                profile.driver
            )));
        }
        let clock = read_attribute(&device.join("current_timestamp_clock"))?;
        if clock.is_empty() {
            return Err(IioError::Invalid("missing timestamp clock".into()));
        }
        let scan = device.join("scan_elements");
        let mut expected: Vec<_> = ["x", "y", "z"]
            .iter()
            .map(|axis| format!("{}_{axis}", profile.prefix))
            .collect();
        expected.push("in_timestamp".into());
        if let Some(name) = &profile.temperature_channel {
            expected.push(name.clone());
        }
        let mut channels = Vec::new();
        for (slot, name) in expected.iter().enumerate() {
            let enabled = read_attribute(&scan.join(format!("{name}_en")))?;
            if slot == 4 && enabled == "0" {
                continue;
            }
            if enabled != "1" {
                return Err(IioError::Invalid(format!(
                    "invalid enable state for {name}"
                )));
            }
            let index: usize = read_attribute(&scan.join(format!("{name}_index")))?
                .parse()
                .map_err(|_| IioError::Invalid(format!("invalid index for {name}")))?;
            let channel = ScanChannel::parse(&read_attribute(&scan.join(format!("{name}_type")))?)?;
            let output_bits = if slot == 3 {
                64
            } else if slot == 4 {
                16
            } else {
                32
            };
            if channel.bits > output_bits || (!channel.signed && channel.bits == output_bits) {
                return Err(IioError::Invalid(format!(
                    "channel {name} exceeds its signed output width"
                )));
            }
            channels.push((index, slot, channel));
        }
        for entry in fs::read_dir(&scan)? {
            let path = entry?.path();
            if let Some(channel) = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix("_en"))
            {
                if !expected.iter().any(|n| n == channel) && read_attribute(&path)? != "0" {
                    return Err(IioError::Invalid(format!(
                        "unexpected enabled channel {channel}"
                    )));
                }
            }
        }
        channels.sort_by_key(|&(index, _, _)| index);
        if channels.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(IioError::Invalid("duplicate scan index".into()));
        }
        let mut layout = Self {
            clock,
            axes: [ScanChannel::default(); 3],
            temperature: None,
            timestamp: ScanChannel::default(),
            scan_bytes: 0,
        };
        let mut alignment = 1;
        for (_, slot, mut channel) in channels {
            alignment = alignment.max(channel.bytes);
            channel.offset = layout.scan_bytes.next_multiple_of(channel.bytes);
            layout.scan_bytes = channel.offset + channel.bytes;
            match slot {
                0..=2 => layout.axes[slot] = channel,
                3 => layout.timestamp = channel,
                _ => layout.temperature = Some(channel),
            }
        }
        layout.scan_bytes = layout.scan_bytes.next_multiple_of(alignment);
        Ok(layout)
    }

    /// Declared kernel clock; never silently treated as the camera clock.
    pub fn clock(&self) -> &str {
        &self.clock
    }

    /// Bytes per aligned scan.
    pub fn scan_bytes(&self) -> usize {
        self.scan_bytes
    }

    /// Decode exactly one scan, discarding padding and sign-extending valid bits.
    ///
    /// # Arguments
    /// * `packet` - one complete scan in the validated layout.
    ///
    /// # Errors
    /// Returns an error for a partial scan or extra bytes.
    pub fn decode(&self, packet: &[u8]) -> Result<IioScan, IioError> {
        let length = self.scan_bytes();
        if packet.len() != length {
            return Err(IioError::Invalid(format!(
                "expected {length} bytes, got {}",
                packet.len()
            )));
        }
        Ok(IioScan {
            raw: self.axes.map(|channel| channel.decode(packet) as i32),
            temperature_raw: self
                .temperature
                .map(|channel| channel.decode(packet) as i16),
            timestamp_ns: self.timestamp.decode(packet),
        })
    }
}

#[cfg(test)]
mod tests;
