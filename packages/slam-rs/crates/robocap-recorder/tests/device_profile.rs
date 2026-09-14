use robocap_recorder::DeviceProfile;

#[test]
fn capture_identity_requires_a_known_matching_cap_and_marks_calibration_origin()
-> anyhow::Result<()> {
    let a = DeviceProfile::identify("robocap_f403b0", "f408193e6447b3b0")?;
    assert_eq!(a.serial(), "f408193e6447b3b0");
    assert!(!a.calibration().placeholder);
    assert_eq!(a.calibration().device_serial, a.serial());
    let b = DeviceProfile::identify("robocap_fe62fa", "fe6fede545c972fa")?;
    assert!(b.calibration().placeholder);
    assert_eq!(b.calibration().device_serial, a.serial());
    for cap in [a, b] {
        assert_eq!(
            cap.camera_paths(),
            [
                "/dev/video75",
                "/dev/video111",
                "/dev/video84",
                "/dev/video66",
                "/dev/video102",
                "/dev/video93"
            ]
        );
    }
    assert!(DeviceProfile::identify("robocap_f403b0", "fe6fede545c972fa").is_err());
    assert!(DeviceProfile::identify("robocap_fe62fa", "f408193e6447b3b0").is_err());
    assert!(DeviceProfile::identify("other", "unknown").is_err());
    Ok(())
}
