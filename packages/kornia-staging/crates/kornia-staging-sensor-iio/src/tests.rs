use std::fs;

use super::*;

fn fake_sysfs(
    root: &std::path::Path,
    driver: &str,
    channels: &[(&str, u8, &str)],
    enabled: bool,
) -> std::io::Result<()> {
    fs::create_dir(root.join("scan_elements"))?;
    fs::write(root.join("name"), driver)?;
    for &(name, index, kind) in channels {
        fs::write(
            root.join(format!("scan_elements/{name}_index")),
            index.to_string(),
        )?;
        fs::write(root.join(format!("scan_elements/{name}_type")), kind)?;
        fs::write(
            root.join(format!("scan_elements/{name}_en")),
            if enabled { "1" } else { "0" },
        )?;
    }
    Ok(())
}

#[test]
fn profiles_reject_invalid_driver_and_attribute_paths() {
    for (driver, prefix) in [("", "in_accel"), ("gyro", ""), ("gyro", "../in_accel")] {
        assert!(ScanProfile::new(driver, prefix, Some("in_temp")).is_err());
    }
}

#[test]
fn buffered_imu_preserves_signed_counts_temperature_and_the_declared_clock(
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fake_sysfs(
        root,
        "icm42688-gyro",
        &[
            ("in_anglvel_x", 0, "be:s16/16>>0"),
            ("in_anglvel_y", 1, "be:s16/16>>0"),
            ("in_anglvel_z", 2, "be:s16/16>>0"),
            ("in_temp", 3, "le:s16/16>>0"),
            ("in_timestamp", 4, "le:s64/64>>0"),
        ],
        true,
    )?;
    // This NUL suffix is present in metadata returned by Cap B.
    fs::write(root.join("current_timestamp_clock"), "realtime\n\0")?;
    let layout = IioScanLayout::read(
        root,
        &ScanProfile::new("icm42688-gyro", "in_anglvel", Some("in_temp"))?,
    )?;
    let bytes = [
        0x80, 0x00, 0x7f, 0xff, 0xff, 0xff, 0xfe, 0xff, 1, 0, 0, 0, 0, 0, 0x20, 0,
    ];
    let sample = layout.decode(&bytes)?;
    assert_eq!(layout.clock(), "realtime");
    assert_eq!(sample.raw, [-32768, 32767, -1]);
    assert_eq!(sample.temperature_raw, Some(-2));
    assert_eq!(sample.timestamp_ns, 9_007_199_254_740_993);
    assert!(layout.decode(&bytes[..15]).is_err());
    fs::write(root.join("scan_elements/in_timestamp_en"), "0")?;
    assert!(IioScanLayout::read(
        root,
        &ScanProfile::new("icm42688-gyro", "in_anglvel", Some("in_temp"))?
    )
    .is_err());
    Ok(())
}

#[test]
fn magnetometer_obeys_signed_18_bit_scan_metadata_and_timestamp_padding(
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fake_sysfs(
        root,
        "mmc5983ma",
        &[
            ("in_magn_x", 0, "be:s18/32>>0"),
            ("in_magn_y", 1, "be:s18/32>>0"),
            ("in_magn_z", 2, "be:s18/32>>0"),
            ("in_timestamp", 3, "le:s64/64>>0"),
        ],
        true,
    )?;
    fs::write(root.join("current_timestamp_clock"), "monotonic\n")?;
    let layout = IioScanLayout::read(root, &ScanProfile::new("mmc5983ma", "in_magn", None)?)?;
    // Valid bits encode minimum, maximum, and -1. Upper bits and timestamp
    // alignment bytes are deliberately nonzero and must not become data.
    let packet = [
        0xfe, 0xfa, 0, 0, 0, 1, 0xff, 0xff, 0, 3, 0xff, 0xff, 0xaa, 0xbb, 0xcc, 0xdd, 0x7b, 0, 0,
        0, 0, 0, 0, 0,
    ];
    let sample = layout.decode(&packet)?;
    assert_eq!(sample.raw, [-131072, 131071, -1]);
    assert_eq!(sample.timestamp_ns, 123);
    assert_eq!(sample.temperature_raw, None);
    assert_eq!(layout.clock(), "monotonic");
    fs::write(root.join("scan_elements/in_magn_x_type"), "be:s33/32>>0")?;
    assert!(IioScanLayout::read(root, &ScanProfile::new("mmc5983ma", "in_magn", None)?).is_err());
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn partial_setup_restores_every_changed_attribute() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let root = dir.path();
    fake_sysfs(
        root,
        "test-gyro",
        &[
            ("in_anglvel_x", 0, "be:s16/16>>0"),
            ("in_anglvel_y", 1, "be:s16/16>>0"),
        ],
        false,
    )?;
    fs::create_dir(root.join("buffer"))?;
    for (name, value) in [
        ("buffer/enable", "0"),
        ("current_timestamp_clock", "realtime"),
    ] {
        fs::write(root.join(name), value)?;
    }
    let node = root.join("device");
    fs::write(&node, [])?;
    let result = IioDevice::start(DeviceConfig {
        sysfs: root.to_owned(),
        node,
        profile: ScanProfile::new("test-gyro", "in_anglvel", Some("in_temp"))?,
        rate_hz: Some(200),
        buffer_scans: 4096,
        watermark: 1,
        read_scans: 256,
        timeout_ms: 100,
    });
    assert!(
        matches!(result, Err(IioError::Attribute { .. })),
        "missing z enable must fail after x/y setup"
    );
    assert_eq!(
        read_attribute(&root.join("current_timestamp_clock"))?,
        "realtime"
    );
    assert_eq!(
        read_attribute(&root.join("scan_elements/in_anglvel_x_en"))?,
        "0"
    );
    assert_eq!(
        read_attribute(&root.join("scan_elements/in_anglvel_y_en"))?,
        "0"
    );
    assert_eq!(read_attribute(&root.join("buffer/enable"))?, "0");
    Ok(())
}

#[cfg(target_os = "linux")]
#[test]
fn configured_device_reads_into_reused_output_and_restores_on_drop(
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let root = dir.path();
    fs::create_dir(root.join("buffer"))?;
    let profile = ScanProfile::new("test", "in_accel", Some("in_temp"))?;
    for (name, value) in [
        ("current_timestamp_clock", "realtime"),
        ("sampling_frequency", "100"),
        ("buffer/enable", "0"),
        ("buffer/length", "64"),
        ("buffer/watermark", "8"),
    ] {
        fs::write(root.join(name), value)?;
    }
    fake_sysfs(
        root,
        "test",
        &[
            ("in_accel_x", 0, "be:s16/16>>0"),
            ("in_accel_y", 1, "be:s16/16>>0"),
            ("in_accel_z", 2, "be:s16/16>>0"),
            ("in_temp", 3, "le:s16/16>>0"),
            ("in_timestamp", 4, "le:s64/64>>0"),
        ],
        false,
    )?;
    let node = root.join("device");
    fs::write(&node, [0u8; 32])?;
    let config = DeviceConfig {
        sysfs: root.to_owned(),
        node,
        profile,
        rate_hz: Some(200),
        buffer_scans: 256,
        watermark: 1,
        read_scans: 2,
        timeout_ms: 100,
    };
    {
        let mut owner = IioDevice::start(config.clone())?;
        assert_eq!(read_attribute(&root.join("buffer/enable"))?, "1");
        let mut out = vec![IioScan {
            raw: [1, 2, 3],
            temperature_raw: None,
            timestamp_ns: -1,
        }];
        owner.read_scans(&mut out)?;
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].timestamp_ns, -1);
        assert_eq!(out[1].raw, [0, 0, 0]);
        assert!(
            owner.read_scans(&mut out).is_err(),
            "EOF is an incomplete device read"
        );
    }
    for (name, value) in [
        ("current_timestamp_clock", "realtime"),
        ("sampling_frequency", "100"),
        ("buffer/enable", "0"),
        ("buffer/length", "64"),
        ("buffer/watermark", "8"),
        ("scan_elements/in_temp_en", "0"),
    ] {
        assert_eq!(read_attribute(&root.join(name))?, value);
    }
    fs::write(root.join("scan_elements/in_other_en"), "1")?;
    assert!(
        IioDevice::start(config.clone()).is_err(),
        "unexpected enabled channel must fail late setup"
    );
    assert_eq!(read_attribute(&root.join("sampling_frequency"))?, "100");
    assert_eq!(read_attribute(&root.join("buffer/length"))?, "64");
    assert_eq!(
        read_attribute(&root.join("current_timestamp_clock"))?,
        "realtime"
    );
    fs::write(root.join("scan_elements/in_other_en"), "0")?;
    fs::write(root.join("scan_elements/in_accel_x_index"), "2")?;
    assert!(
        IioDevice::start(config).is_err(),
        "duplicate index must fail"
    );
    assert_eq!(read_attribute(&root.join("buffer/enable"))?, "0");
    Ok(())
}

#[test]
fn metadata_drives_endianness_shift_index_order_and_alignment(
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fake_sysfs(
        root,
        "generic",
        &[
            ("in_accel_x", 4, "le:s12/16>>4"),
            ("in_accel_y", 1, "be:u8/8"),
            ("in_accel_z", 8, "be:s20/32X1>>3"),
            ("in_timestamp", 0, "be:s64/64>>0"),
        ],
        true,
    )?;
    fs::write(root.join("current_timestamp_clock"), "monotonic")?;
    let layout = IioScanLayout::read(root, &ScanProfile::new("generic", "in_accel", None)?)?;
    assert_eq!(layout.scan_bytes(), 16);
    let mut packet = [0u8; 16];
    packet[..8].copy_from_slice(&123i64.to_be_bytes());
    packet[8] = 255;
    packet[10..12].copy_from_slice(&(-32i16).to_le_bytes());
    packet[12..16].copy_from_slice(&(0x7f_fff8u32).to_be_bytes());
    let sample = layout.decode(&packet)?;
    assert_eq!(sample.raw, [-2, 255, -1]);
    assert_eq!(sample.timestamp_ns, 123);
    for bad in [
        "be:s0/16>>0",
        "be:s17/16>>0",
        "le:s16/16>>1",
        "le:s8/24>>0",
        "xx:s8/8>>0",
        "be:s8/8X2>>0",
    ] {
        fs::write(root.join("scan_elements/in_accel_x_type"), bad)?;
        assert!(
            IioScanLayout::read(root, &ScanProfile::new("generic", "in_accel", None)?).is_err(),
            "{bad}"
        );
    }
    Ok(())
}
