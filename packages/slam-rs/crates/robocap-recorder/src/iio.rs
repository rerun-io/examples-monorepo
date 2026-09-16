use std::{fs, path::Path};

use anyhow::{Context, Result, ensure};

/// A decoded kernel scan, without clock conversion, calibration, or SI scaling.
#[derive(Debug)]
pub struct IioScan {
    pub raw: [i32; 3],
    pub temperature_raw: Option<i16>,
    pub timestamp_ns: i64,
}

/// Validated layout of an enabled Cap B ICM42688 or MMC5983MA buffer.
/// Reading metadata does not change device configuration or enable capture.
pub struct IioScanLayout {
    clock: String,
    temperature: bool,
    magnetometer: bool,
}

pub(crate) fn read_attribute(path: &Path) -> Result<String> {
    Ok(fs::read_to_string(path)
        .with_context(|| format!("read IIO attribute {}", path.display()))?
        .trim_matches(|character: char| character.is_whitespace() || character == '\0')
        .to_owned())
}

impl IioScanLayout {
    pub fn read(device: &Path) -> Result<Self> {
        let name = read_attribute(&device.join("name"))?;
        let prefix = match name.as_str() {
            "icm42688-gyro" => "in_anglvel",
            "icm42688-accel" => "in_accel",
            "mmc5983ma" => "in_magn",
            _ => anyhow::bail!("unsupported IIO device {name}"),
        };
        let clock = read_attribute(&device.join("current_timestamp_clock"))?;
        ensure!(!clock.is_empty(), "IIO timestamp clock is missing");
        let scan = device.join("scan_elements");
        let magnetometer = name == "mmc5983ma";
        let mut expected = Vec::new();
        for (index, axis) in ["x", "y", "z"].into_iter().enumerate() {
            expected.push((
                format!("{prefix}_{axis}"),
                index,
                if magnetometer {
                    "be:s18/32>>0"
                } else {
                    "be:s16/16>>0"
                },
            ));
        }
        if !magnetometer {
            expected.push(("in_temp".into(), 3, "le:s16/16>>0"));
        }
        expected.push((
            "in_timestamp".into(),
            if magnetometer { 3 } else { 4 },
            "le:s64/64>>0",
        ));
        let mut temperature = false;
        for (name, index, kind) in &expected {
            ensure!(
                read_attribute(&scan.join(format!("{name}_index")))? == index.to_string()
                    && read_attribute(&scan.join(format!("{name}_type")))? == *kind,
                "unsupported IIO scan format for {name}"
            );
            let enabled = read_attribute(&scan.join(format!("{name}_en")))?;
            ensure!(enabled == "0" || enabled == "1", "invalid IIO enable state");
            if name == "in_temp" {
                temperature = enabled == "1";
            } else {
                ensure!(enabled == "1", "required IIO channel {name} is disabled");
            }
        }
        for entry in fs::read_dir(&scan)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("invalid IIO channel name")?;
            if let Some(channel) = name.strip_suffix("_en") {
                ensure!(
                    expected.iter().any(|(name, _, _)| name == channel)
                        || read_attribute(&path)? == "0",
                    "unexpected enabled IIO channel {channel}"
                );
            }
        }
        Ok(Self {
            clock,
            temperature,
            magnetometer,
        })
    }

    /// The kernel's declared clock, never silently treated as a camera clock.
    pub fn clock(&self) -> &str {
        &self.clock
    }

    pub fn decode(&self, packet: &[u8]) -> Result<IioScan> {
        // Linux IIO aligns the timestamp to eight bytes, with or without temp.
        let length = if self.magnetometer { 24 } else { 16 };
        ensure!(
            packet.len() == length,
            "expected a complete {length}-byte IIO scan"
        );
        let raw = std::array::from_fn(|axis| {
            if self.magnetometer {
                let offset = axis * 4;
                let word = i32::from_be_bytes([
                    packet[offset],
                    packet[offset + 1],
                    packet[offset + 2],
                    packet[offset + 3],
                ]);
                // Discard unused high bits, then sign-extend the declared 18 bits.
                (word << 14) >> 14
            } else {
                i32::from(i16::from_be_bytes([packet[axis * 2], packet[axis * 2 + 1]]))
            }
        });
        Ok(IioScan {
            raw,
            temperature_raw: self
                .temperature
                .then(|| i16::from_le_bytes([packet[6], packet[7]])),
            timestamp_ns: i64::from_le_bytes(packet[length - 8..length].try_into()?),
        })
    }
}
