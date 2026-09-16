use std::fs;

use robocap_recorder::IioScanLayout;

#[test]
fn buffered_imu_preserves_signed_counts_temperature_and_the_declared_clock() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("scan_elements"))?;
    fs::write(root.join("name"), "icm42688-gyro\n")?;
    // This NUL suffix is present in the metadata returned by Cap B.
    fs::write(root.join("current_timestamp_clock"), "realtime\n\0")?;
    for (name, index, kind) in [
        ("in_anglvel_x", 0, "be:s16/16>>0"),
        ("in_anglvel_y", 1, "be:s16/16>>0"),
        ("in_anglvel_z", 2, "be:s16/16>>0"),
        ("in_temp", 3, "le:s16/16>>0"),
        ("in_timestamp", 4, "le:s64/64>>0"),
    ] {
        fs::write(
            root.join(format!("scan_elements/{name}_index")),
            index.to_string(),
        )?;
        fs::write(root.join(format!("scan_elements/{name}_type")), kind)?;
        fs::write(root.join(format!("scan_elements/{name}_en")), "1")?;
    }
    let layout = IioScanLayout::read(root)?;
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
    assert!(IioScanLayout::read(root).is_err());
    Ok(())
}

#[test]
fn magnetometer_obeys_signed_18_bit_scan_metadata_and_timestamp_padding() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path();
    fs::create_dir(root.join("scan_elements"))?;
    fs::write(root.join("name"), "mmc5983ma\n")?;
    fs::write(root.join("current_timestamp_clock"), "monotonic\n")?;
    for (name, index, kind) in [
        ("in_magn_x", 0, "be:s18/32>>0"),
        ("in_magn_y", 1, "be:s18/32>>0"),
        ("in_magn_z", 2, "be:s18/32>>0"),
        ("in_timestamp", 3, "le:s64/64>>0"),
    ] {
        fs::write(
            root.join(format!("scan_elements/{name}_index")),
            index.to_string(),
        )?;
        fs::write(root.join(format!("scan_elements/{name}_type")), kind)?;
        fs::write(root.join(format!("scan_elements/{name}_en")), "1")?;
    }
    let layout = IioScanLayout::read(root)?;
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
    fs::write(root.join("scan_elements/in_magn_x_type"), "be:u18/32>>0")?;
    assert!(IioScanLayout::read(root).is_err());
    Ok(())
}
